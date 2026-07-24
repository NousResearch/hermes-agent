"""Tor daemon lifecycle manager.

Manages a Tor subprocess with user-provided bridge configuration,
lyrebird (obfs4proxy successor) pluggable transports, and SOCKS5 proxy.

Key design decisions:
  - Lyrebird path is ABSOLUTE — raw tor.exe doesn't understand
    Tor Browser's ${pt_path} variable.
  - GeoIP files are required (bundled in Expert Bundle).
  - Bootstrap detection uses a background thread to read stdout
    because select.select() only works on sockets on Windows.
  - Process management uses terminate() on Windows, SIGINT on Linux.
"""
import os
import queue
import secrets
import shlex
import signal
import subprocess
from darkloom.policy import NetworkChannel, authorize
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import quote

import httpx

from darkloom.constants import (
    DEFAULT_SOCKS_PORT,
    DEFAULT_CONTROL_PORT,
    TOR_DATA_DIR,
    TOR_BINARY_DIR,
    get_lyrebird_path,
    get_geoip_paths,
)
from darkloom.bridges import Bridge
from darkloom.secure_files import atomic_private_write, private_lock

from darkloom.privacy import get_logger

logger = get_logger(__name__)


def authenticated_socks_proxy_url(socks_port: int = DEFAULT_SOCKS_PORT) -> str:
    """Return a fresh SOCKS URL whose authentication isolates its streams."""
    username = quote(secrets.token_urlsafe(24), safe="")
    password = quote(secrets.token_urlsafe(32), safe="")
    return f"socks5://{username}:{password}@127.0.0.1:{socks_port}"


# Template with absolute paths filled at generation time.
# No ${pt_path} — we resolve paths ourselves.
TORRC_TEMPLATE = """\
# darkloom generated torrc — {generated_at}
# DO NOT EDIT MANUALLY — regenerated on each start.

# Bind locally and use SOCKS credentials as Tor circuit-isolation tokens.
SOCKSPort 127.0.0.1:{socks_port} IsolateSOCKSAuth
{control_endpoint}
DataDirectory {data_dir}
Log notice stdout
RunAsDaemon 0
AvoidDiskWrites 1
CookieAuthentication 1
CookieAuthFile {cookie_path}
CookieAuthFileGroupReadable 0
GeoIPFile {geoip_path}
GeoIPv6File {geoip6_path}

# Pluggable transports — lyrebird handles obfs2/3/4, meek, snowflake, scramblesuit, webtunnel
{transport_plugins}

# User-provided bridges
{bridge_section}
"""


class TorDaemonError(Exception):
    """Raised when Tor daemon operations fail."""


@dataclass(frozen=True)
class IsolationIdentity:
    """Boundaries which must not be linkable through a SOCKS circuit."""

    conversation_id: str
    agent_id: str
    subagent_id: str | None = None
    platform_account: str | None = None
    browser_context: str | None = None
    sensitive_task: str | None = None

    def __post_init__(self) -> None:
        if not self.conversation_id.strip() or not self.agent_id.strip():
            raise ValueError("conversation_id and agent_id are required for SOCKS isolation")


class SocksCredential:
    """A single-use SOCKS authentication lease with best-effort zeroization."""

    def __init__(self, identity: IsolationIdentity):
        self.identity = identity
        self._username = bytearray(secrets.token_urlsafe(24), "ascii")
        self._password = bytearray(secrets.token_urlsafe(32), "ascii")
        self._discarded = False

    def authentication(self) -> tuple[str, str]:
        if self._discarded:
            raise TorDaemonError("SOCKS credential has already been discarded")
        return self._username.decode("ascii"), self._password.decode("ascii")

    def discard(self) -> None:
        for secret in (self._username, self._password):
            secret[:] = b"\x00" * len(secret)
        self._discarded = True

    @property
    def discarded(self) -> bool:
        return self._discarded


class TorDaemon:
    """Manages a Tor daemon subprocess with bridge configuration."""

    def __init__(
        self,
        tor_binary: Path,
        data_dir: Path | None = None,
        bridges: list[str] | None = None,
        socks_port: int = DEFAULT_SOCKS_PORT,
        control_port: int = DEFAULT_CONTROL_PORT,
        tor_binary_dir: Path | None = None,
    ):
        if not tor_binary.exists():
            raise TorDaemonError(f"Tor binary not found: {tor_binary}")

        self.tor_binary = tor_binary
        self.data_dir = data_dir or TOR_DATA_DIR
        self.bridges = bridges or []
        self.socks_port = socks_port
        self.control_port = control_port
        self.tor_binary_dir = tor_binary_dir or TOR_BINARY_DIR

        self._process: Optional[subprocess.Popen] = None
        self._torrc_path = self.data_dir / "torrc"
        self.control_socket_path = self.data_dir / "control.sock"
        self.cookie_path = self.data_dir / "control_auth_cookie"
        self._start_time: Optional[float] = None
        self._credential_lock = threading.Lock()
        self._active_credentials: set[SocksCredential] = set()

    # ── Public API ─────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def socks_proxy_url(self) -> str:
        """Return the listener address; do not export it as ``ALL_PROXY``."""
        return f"socks5://127.0.0.1:{self.socks_port}"

    def issue_socks_credential(self, identity: IsolationIdentity) -> SocksCredential:
        """Issue a fresh, non-reusable credential for one isolation boundary."""
        if not isinstance(identity, IsolationIdentity):
            raise TorDaemonError("an explicit IsolationIdentity is required")
        credential = SocksCredential(identity)
        with self._credential_lock:
            self._active_credentials.add(credential)
        return credential

    def discard_socks_credential(self, credential: SocksCredential) -> None:
        """Revoke a credential lease and erase its locally held secret bytes."""
        with self._credential_lock:
            if credential not in self._active_credentials:
                raise TorDaemonError("SOCKS credential is not active")
            self._active_credentials.remove(credential)
        credential.discard()

    @contextmanager
    def isolated_client(
        self, identity: IsolationIdentity | None, **client_kwargs
    ) -> Iterator[httpx.Client]:
        """Yield a request-scoped client, then close it and discard its identity.

        Every entry creates random credentials, including repeated entries for
        the same identity.  ``trust_env=False`` prevents environment-wide proxy
        configuration from replacing or bypassing the isolation session.
        """
        if identity is None:
            raise TorDaemonError("anonymous SOCKS clients are forbidden; assign an identity")
        controlled_options = {"proxy", "trust_env", "mounts"}.intersection(client_kwargs)
        if controlled_options:
            names = ", ".join(sorted(controlled_options))
            raise TorDaemonError(f"isolated clients control routing settings: {names}")
        credential = self.issue_socks_credential(identity)
        username, password = credential.authentication()
        proxy = f"socks5://{username}:{password}@127.0.0.1:{self.socks_port}"
        try:
            with httpx.Client(proxy=proxy, trust_env=False, **client_kwargs) as client:
                yield client
        finally:
            self.discard_socks_credential(credential)

    @property
    def uptime_seconds(self) -> float | None:
        if self._start_time is None:
            return None
        return time.time() - self._start_time

    def start(self, timeout: float = 60.0) -> None:
        """Start the Tor daemon. Blocks until bootstrapped or timeout.

        Raises TorDaemonError if Tor exits prematurely or fails to bootstrap.
        """
        if self.is_running:
            logger.info("Tor daemon already running (PID %s)", self._process.pid)
            return

        self._write_torrc()
        self._verify_prerequisites()

        cmd = [str(self.tor_binary), "-f", str(self._torrc_path)]
        logger.info("Starting Tor with redacted command arguments")

        authorize(NetworkChannel.TOR_BOOTSTRAP)
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._start_time = time.time()

        # Use a thread to read stdout because select.select()
        # only works on sockets on Windows, not pipes.
        line_queue: queue.Queue = queue.Queue()
        stop_reader = threading.Event()

        def reader():
            """Read Tor stdout line by line into the queue."""
            try:
                assert self._process and self._process.stdout
                for line in iter(self._process.stdout.readline, ""):
                    if stop_reader.is_set():
                        break
                    line_queue.put(line)
            except (ValueError, OSError):
                pass
            finally:
                line_queue.put(None)  # Sentinel: reader done

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()

        # Wait for bootstrap
        deadline = time.time() + timeout
        bootstrapped = False
        last_line = ""

        try:
            while time.time() < deadline:
                # Check if process died
                if self._process.poll() is not None:
                    # Drain remaining lines
                    stop_reader.set()
                    remaining = []
                    while not line_queue.empty():
                        try:
                            remaining.append(line_queue.get_nowait())
                        except queue.Empty:
                            break
                    raise TorDaemonError(
                        f"Tor exited prematurely (code {self._process.returncode})."
                    )

                # Non-blocking read from queue
                try:
                    line = line_queue.get(timeout=0.1)
                    if line is None:  # Reader sentinel
                        break
                    line = line.rstrip()
                    if line:
                        logger.debug("Tor log event received")
                        last_line = line
                        if "Bootstrapped 100%" in line:
                            bootstrapped = True
                            break
                except queue.Empty:
                    pass

        finally:
            stop_reader.set()

        if not bootstrapped:
            self.stop()
            raise TorDaemonError(
                f"Tor failed to bootstrap within {timeout}s; "
                "check connectivity or replace the bridge configuration."
            )

        logger.info(
            "Tor daemon running (PID %s, SOCKS5 %s, uptime %.1fs)",
            self._process.pid,
            self.socks_proxy_url,
            self.uptime_seconds,
        )

        # Tor normally creates this as 0600.  Enforce that invariant rather
        # than relying on the process umask, since it grants control of Tor.
        if self.cookie_path.exists():
            self.cookie_path.chmod(0o600)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the Tor daemon gracefully."""
        if not self._process:
            self._discard_all_credentials()
            return

        pid = self._process.pid
        logger.info("Stopping Tor daemon (PID %s)...", pid)

        if self._process.poll() is None:
            # Graceful shutdown: SIGINT on Linux, terminate() on Windows
            if os.name == "nt":
                self._process.terminate()
            else:
                self._process.send_signal(signal.SIGINT)

            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Tor did not exit gracefully, force-killing")
                self._process.kill()
                self._process.wait(timeout=5)

        self._process = None
        self._start_time = None
        self._discard_all_credentials()
        logger.info("Tor daemon stopped")

    def health_check(self) -> bool:
        """Complete SOCKS5 method negotiation with the managed daemon."""
        if not self.is_running:
            return False

        authorize(NetworkChannel.TOR_CONTROL, local_only=True)
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(("127.0.0.1", self.socks_port))
            sock.sendall(b"\x05\x01\x00")
            result = sock.recv(2) == b"\x05\x00"
            sock.close()
            return result
        except Exception:
            return False

    def process_health(self) -> bool:
        """Verify this object still owns the exact live Tor subprocess."""
        if not self.is_running or self._process is None or self._process.pid <= 0:
            return False
        if os.name != "nt":
            try:
                return Path(f"/proc/{self._process.pid}/exe").resolve() == self.tor_binary.resolve()
            except OSError:
                return False
        return True

    def bootstrap_status(self) -> tuple[int | None, str | None]:
        """Read authenticated bootstrap progress from Tor's ControlPort."""
        if not self.process_health():
            return None, "managed Tor process is not healthy"
        try:
            import re
            import socket
            cookie = (self.data_dir / "control_auth_cookie").read_bytes().hex().encode()
            with socket.create_connection(("127.0.0.1", self.control_port), timeout=3) as sock:
                sock.sendall(b"AUTHENTICATE " + cookie + b"\r\n")
                if not sock.recv(1024).startswith(b"250"):
                    return None, "control authentication failed"
                sock.sendall(b"GETINFO status/bootstrap-phase\r\nQUIT\r\n")
                response = sock.recv(4096).decode("utf-8", "replace")
            match = re.search(r"PROGRESS=(\d+)", response)
            return (int(match.group(1)), None) if match else (None, "missing bootstrap progress")
        except Exception as exc:
            return None, str(exc)

    def signal_newnym(self) -> bool:
        """Authenticate to the owned daemon and request fresh circuits."""
        if not self.process_health():
            return False
        try:
            import socket
            cookie = (self.data_dir / "control_auth_cookie").read_bytes().hex().encode()
            with socket.create_connection(("127.0.0.1", self.control_port), timeout=5) as sock:
                sock.sendall(b"AUTHENTICATE " + cookie + b"\r\n")
                if not sock.recv(1024).startswith(b"250"):
                    return False
                sock.sendall(b"SIGNAL NEWNYM\r\n")
                return sock.recv(1024).startswith(b"250")
        except Exception:
            return False

    # ── Context manager ────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        return False

    # ── Internals ──────────────────────────────────────────────

    def _discard_all_credentials(self) -> None:
        """Discard every outstanding lease during daemon shutdown."""
        with self._credential_lock:
            credentials = tuple(self._active_credentials)
            self._active_credentials.clear()
        for credential in credentials:
            credential.discard()

    def _verify_prerequisites(self):
        """Check that required files exist before starting Tor."""
        lyrebird = get_lyrebird_path(self.tor_binary_dir)
        if not lyrebird.exists():
            raise TorDaemonError(
                f"Lyrebird (pluggable transport) not found at {lyrebird}.\n"
                f"The Tor Expert Bundle should include lyrebird. "
                f"Re-download with: darkloom download"
            )

        geoip, geoip6 = get_geoip_paths(self.tor_binary_dir)
        if not geoip.exists():
            logger.warning("GeoIP database not found at %s — country-based routing disabled", geoip)
        if not geoip6.exists():
            logger.warning("GeoIPv6 database not found at %s", geoip6)

    def _build_torrc(self) -> str:
        """Generate torrc content with absolute paths and user bridges."""
        lyrebird_path = get_lyrebird_path(self.tor_binary_dir)
        geoip_path, geoip6_path = get_geoip_paths(self.tor_binary_dir)

        # Transport plugins — use absolute path to lyrebird
        # lyrebird handles: obfs2, obfs3, obfs4, meek_lite, scramblesuit, webtunnel, snowflake
        transport_plugins = (
            f"ClientTransportPlugin obfs2,obfs3,obfs4,meek_lite,scramblesuit,webtunnel,snowflake "
            f"exec {lyrebird_path}"
        )

        # Bridge section
        if self.bridges:
            bridge_lines = "UseBridges 1\n"
            bridge_lines += "\n".join(f"Bridge {b}" for b in self.bridges)
        else:
            bridge_lines = (
                "# No bridges configured — Tor will use public relays.\n"
                "# To add bridges, save them to ~/.hermes/tor/bridges.txt\n"
                "# and restart. Get bridges from @GetBridgesBot on Telegram."
            )

        if os.name == "nt":
            # Never expose the control transport beyond this host.
            control_endpoint = f"ControlPort 127.0.0.1:{self.control_port}"
        else:
            # Avoid a TCP listener altogether where Tor supports Unix sockets.
            control_endpoint = f'ControlSocket "{self.control_socket_path}"'

        return TORRC_TEMPLATE.format(
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            socks_port=self.socks_port,
            control_endpoint=control_endpoint,
            data_dir=self.data_dir,
            cookie_path=self.cookie_path,
            geoip_path=geoip_path,
            geoip6_path=geoip6_path,
            transport_plugins=transport_plugins,
            bridge_section=bridge_lines,
        )

    def _write_torrc(self):
        """Write torrc to disk with secure permissions."""
        from darkloom.secure_files import private_lock, atomic_private_write

        torrc_content = self._build_torrc()
        with private_lock(self._torrc_path):
            atomic_private_write(self._torrc_path, torrc_content)
        logger.debug("torrc written (%d bytes) to %s", len(torrc_content), self._torrc_path)
