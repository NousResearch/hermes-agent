"""
WhatsApp platform adapter.

WhatsApp integration is more complex than Telegram/Discord because:
- No official bot API for personal accounts
- Business API requires Meta Business verification
- Most solutions use web-based automation

This adapter supports multiple backends:
1. WhatsApp Business API (requires Meta verification)
2. whatsapp-web.js (via Node.js subprocess) - for personal accounts
3. Baileys (via Node.js subprocess) - alternative for personal accounts

For simplicity, we'll implement a generic interface that can work
with different backends via a bridge pattern.
"""

import asyncio
import logging
import os
import platform
import re
import signal
import subprocess

_IS_WINDOWS = platform.system() == "Windows"
from pathlib import Path
from typing import Dict, Optional, Any

from hermes_cli._subprocess_compat import windows_detach_popen_kwargs
from hermes_constants import (
    find_node_executable,
    get_hermes_dir,
    with_hermes_node_path,
)

logger = logging.getLogger(__name__)

# Inbound owner-typed WhatsApp text is prefixed at MessageEvent construction so
# transcripts stay disambiguated even if downstream plugins fail before silent_ingest.
_OWNER_REPLY_PREFIX = "[owner reply] "


def _listener_pids_on_port(port: int) -> list:
    """PIDs of processes *listening* on ``port`` (POSIX) — never clients.

    This must match only LISTEN sockets. A bare ``lsof -i :PORT`` (or
    ``fuser PORT/tcp``) also returns *clients* whose connection merely involves
    that port number — e.g. a browser with a tab open on a local dev server
    sharing the port. SIGTERMing those closed the user's browser at irregular
    intervals. Restricting to LISTEN state frees the port for a new bridge
    without ever touching an unrelated client.
    """
    pids: list = []
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            try:
                pids.append(int(line))
            except ValueError:
                pass
        if pids:
            return pids
    except FileNotFoundError:
        pass  # lsof not installed — fall through to ss
    # Fallback: ss (iproute2, present on virtually every modern Linux).
    try:
        result = subprocess.run(
            ["ss", "-ltnHp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=5,
        )
        for m in re.finditer(r"pid=(\d+)", result.stdout):
            pids.append(int(m.group(1)))
    except FileNotFoundError:
        pass
    return pids


def _kill_port_process(port: int) -> None:
    """Kill any process *listening* on the given TCP port (a stale bridge)."""
    try:
        if _IS_WINDOWS:
            from hermes_cli._subprocess_compat import windows_hide_flags

            # Use netstat to find the PID bound to this port, then taskkill
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=5,
                creationflags=windows_hide_flags(),
            )
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[3] == "LISTENING":
                    local_addr = parts[1]
                    if local_addr.endswith(f":{port}"):
                        try:
                            subprocess.run(
                                ["taskkill", "/PID", parts[4], "/F"],
                                capture_output=True, timeout=5,
                                creationflags=windows_hide_flags(),
                            )
                        except subprocess.SubprocessError:
                            pass
        else:
            # POSIX: only ever signal a process LISTENING on the port. A client
            # whose connection happens to involve this port number (a browser
            # tab on a local dev server, etc.) must never be killed.
            for pid in _listener_pids_on_port(port):
                try:
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
    except Exception:
        pass


def _bridge_pid_is_ours(pid: int, session_path: Path, expected_start) -> bool:
    """True only if ``pid`` is alive AND still our node bridge for this session.

    The PID is read from a file written by a previous run.  Once that process
    exits and is reaped the kernel can recycle the number onto an unrelated
    process — observed in the wild landing on a desktop browser's main process,
    which a bare-liveness ``os.kill`` then SIGTERMed, closing the whole browser
    at irregular intervals (every time the flapping bridge restarted).

    Identity is confirmed two ways: the kernel start time captured when we wrote
    the pidfile (definitive), and — for legacy pidfiles with no baseline — the
    command line, which must contain ``node`` and this session's unique path.
    A recycled PID (different start time / different cmdline) is never ours.
    """
    from gateway.status import _pid_exists
    if not _pid_exists(pid):
        return False
    if expected_start is not None:
        from gateway.status import get_process_start_time
        # A matching (pid, start time) pair uniquely identifies the process.
        return get_process_start_time(pid) == expected_start
    # Legacy pidfile (no recorded start time): fall back to a command-line
    # signature so a recycled PID is still never signalled.  If we cannot read
    # the cmdline we refuse to kill rather than risk a stranger.
    from gateway.status import _read_process_cmdline
    cmdline = _read_process_cmdline(pid)
    if not cmdline:
        return False
    return ("node" in cmdline) and (str(session_path) in cmdline)


def _kill_stale_bridge_by_pidfile(session_path: Path) -> None:
    """Kill a bridge process recorded in a PID file from a previous run.

    The bridge writes ``bridge.pid`` into the session directory when it
    starts.  If the gateway crashed without a clean shutdown the old bridge
    process becomes orphaned — this helper finds and kills it.

    Critically, the recorded PID is re-validated against the live process
    (:func:`_bridge_pid_is_ours`) before any signal, so a recycled PID that now
    names an unrelated process (e.g. the user's browser) is never killed.
    """
    pid_file = session_path / "bridge.pid"
    if not pid_file.exists():
        return
    pid = None
    recorded_start = None
    try:
        # Format: line 1 = pid, optional line 2 = kernel start time. Legacy
        # files written before the guard existed have only the pid.
        lines = pid_file.read_text().split("\n")
        pid = int(lines[0].strip())
        if len(lines) > 1 and lines[1].strip():
            recorded_start = int(lines[1].strip())
    except (ValueError, OSError, TypeError, IndexError):
        try:
            pid_file.unlink()
        except OSError:
            pass
        return
    if _bridge_pid_is_ours(pid, session_path, recorded_start):
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info("[whatsapp] Killed stale bridge PID %d from pidfile", pid)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    else:
        from gateway.status import _pid_exists
        if _pid_exists(pid):
            logger.warning(
                "[whatsapp] Not killing pidfile PID %d: it is no longer the "
                "bridge (recycled onto an unrelated process); skipping to avoid "
                "killing a stranger.", pid,
            )
    try:
        pid_file.unlink()
    except OSError:
        pass


def _write_bridge_pidfile(session_path: Path, pid: int) -> None:
    """Write the bridge PID (and its kernel start time) for later cleanup.

    The start time on line 2 lets a future run prove the PID still names this
    exact process before signalling it, so a recycled PID can never be killed
    as a "stale bridge". Older single-line files remain readable.
    """
    try:
        from gateway.status import get_process_start_time
        start = get_process_start_time(pid)
        text = str(pid) if start is None else "{}\n{}".format(pid, start)
        (session_path / "bridge.pid").write_text(text)
    except OSError:
        pass


def _terminate_bridge_process(proc, *, force: bool = False) -> None:
    """Terminate the bridge process using process-tree semantics where possible."""
    if _IS_WINDOWS:
        cmd = ["taskkill", "/PID", str(proc.pid), "/T"]
        if force:
            cmd.append("/F")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            if force:
                proc.kill()
            else:
                proc.terminate()
            return

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise OSError(details or f"taskkill failed for PID {proc.pid}")
        return

    import psutil
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        if force:
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
        else:
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            parent.terminate()
    except psutil.NoSuchProcess:
        return

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin
from gateway.whatsapp_identity import to_whatsapp_jid
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SUPPORTED_DOCUMENT_TYPES,
    cache_image_from_url,
    cache_audio_from_url,
)
from utils import env_int


def _is_allowed_bridge_path(url: str) -> bool:
    """Return True only when an absolute path from the bridge resolves inside a
    known Hermes media cache directory.

    The Baileys bridge is a local subprocess that downloads inbound media and
    hands back absolute file paths. A compromised or buggy bridge could hand
    back an arbitrary path (e.g. ``/etc/passwd``) which would otherwise be
    attached verbatim and sent to the model. Resolve the path (following any
    symlinks) and require it to live under one of the real cache roots — this
    covers both the canonical ``cache/<kind>`` layout and the legacy
    ``<kind>_cache`` layout that ``get_hermes_dir`` may return.
    """
    try:
        resolved = Path(url).resolve()
    except (OSError, ValueError):
        return False
    # Resolve the cache roots per-call via the getters (not the import-time
    # constants) so this validator follows the active profile override; under a
    # profile override the inbound bridge writes media into that profile's
    # cache, which the frozen constants would not match.
    from gateway.platforms.base import (
        get_audio_cache_dir,
        get_document_cache_dir,
        get_image_cache_dir,
        get_video_cache_dir,
    )

    for root in (
        get_image_cache_dir(),
        get_audio_cache_dir(),
        get_video_cache_dir(),
        get_document_cache_dir(),
    ):
        try:
            if resolved.is_relative_to(Path(root).resolve()):
                return True
        except (OSError, ValueError):
            continue
    return False


def _file_content_hash(path: Path) -> str:
    """Return the first 16 hex chars of the SHA-256 of *path*'s contents.

    Used for the bridge staleness handshake: bridge.js reports its own
    source hash in ``/health`` (``scriptHash``), and the adapter compares
    it against the hash of bridge.js currently on disk.  A mismatch means
    a long-lived bridge process is serving code from before an update.
    Returns ``""`` when the file can't be read.
    """
    import hashlib
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def check_whatsapp_requirements() -> bool:
    """
    Check if WhatsApp dependencies are available.
    
    WhatsApp requires a Node.js bridge for most implementations.
    """
    # Prefer Hermes-managed Node/npm so Windows installs are not broken by a
    # bad or elevation-triggering system Node on PATH.
    _node = find_node_executable("node")
    if not _node:
        return False
    try:
        result = subprocess.run(
            [_node, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


class WhatsAppAdapter(WhatsAppBehaviorMixin, BasePlatformAdapter):
    """
    WhatsApp adapter.
    
    This implementation uses a simple HTTP bridge pattern where:
    1. A Node.js process runs the WhatsApp Web client
    2. Messages are forwarded via HTTP/IPC to this Python adapter
    3. Responses are sent back through the bridge
    
    The actual Node.js bridge implementation can vary:
    - whatsapp-web.js based
    - Baileys based
    - Business API based
    
    Configuration:
    - bridge_script: Path to the Node.js bridge script
    - bridge_port: Port for HTTP communication (default: 3000)
    - session_path: Path to store WhatsApp session data
    - dm_policy: "open" | "allowlist" | "disabled" | "pairing" — how DMs are handled (default: "pairing")
    - allow_from: List of sender IDs allowed in DMs (when dm_policy="allowlist")
    - group_policy: "open" | "allowlist" | "disabled" | "pairing" — which groups are processed (default: "pairing")
    - group_allow_from: List of group JIDs allowed (when group_policy="allowlist")

    Behavior (gating, mention parsing, markdown conversion, chunking) is
    provided by ``WhatsAppBehaviorMixin`` so the Cloud API adapter can
    share it. Only transport-specific code lives here.
    """

    # Default bridge location resolved via shared helper
    _DEFAULT_BRIDGE_DIR = None  # resolved in __init__
    splits_long_messages = True  # send() chunks via truncate_message()

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WHATSAPP)
        # Use shared helper for bridge directory resolution (handles read-only install tree)
        if WhatsAppAdapter._DEFAULT_BRIDGE_DIR is None:
            from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
            WhatsAppAdapter._DEFAULT_BRIDGE_DIR = resolve_whatsapp_bridge_dir()
        self._bridge_process: Optional[subprocess.Popen] = None
        self._bridge_port: int = config.extra.get("bridge_port", 3000)
        self._bridge_script: Optional[str] = config.extra.get(
            "bridge_script",
            str(self._DEFAULT_BRIDGE_DIR / "bridge.js"),
        )
        self._session_path: Path = Path(config.extra.get(
            "session_path",
            get_hermes_dir("platforms/whatsapp/session", "whatsapp/session")
        ))
        self._reply_prefix: Optional[str] = config.extra.get("reply_prefix")
        self._dm_policy = str(config.extra.get("dm_policy") or os.getenv("WHATSAPP_DM_POLICY", "pairing")).strip().lower()
        self._allow_from = self._coerce_allow_list(config.extra.get("allow_from") or config.extra.get("allowFrom"))
        self._group_policy = str(config.extra.get("group_policy") or os.getenv("WHATSAPP_GROUP_POLICY", "pairing")).strip().lower()
        self._group_allow_from = self._coerce_allow_list(config.extra.get("group_allow_from") or config.extra.get("groupAllowFrom"))
        self._mention_patterns = self._compile_mention_patterns()
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._bridge_log_fh = None
        self._bridge_log: Optional[Path] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._http_session: Optional["aiohttp.ClientSession"] = None
        # Set to True by disconnect() before we SIGTERM our child bridge so
        # _check_managed_bridge_exit() can distinguish an intentional
        # shutdown-time exit (returncode -15 / -2 / 0) from a real crash.
        # Without this, every graceful gateway shutdown/restart would log
        # "Fatal whatsapp adapter error" plus dispatch a fatal-error
        # notification before the normal "✓ whatsapp disconnected" fires.
        self._shutting_down: bool = False

        # Text debounce batching (mirrors Telegram adapter pattern).
        # WhatsApp often delivers multiple messages in rapid succession
        # (e.g. forwarded batches, paste-splits) — without debounce each
        # message triggers a separate agent invocation, wasting tokens and
        # flooding the user with reply fragments.  Default 5s delay /
        # 10s split delay are conservative for WhatsApp's delivery cadence.
        # Tunable via config.yaml under
        # ``gateway.platforms.whatsapp.extra.text_batch_delay_seconds`` /
        # ``text_batch_split_delay_seconds``.
        self._text_batch_delay_seconds = self._coerce_float_extra(
            "text_batch_delay_seconds", 5.0
        )
        self._text_batch_split_delay_seconds = self._coerce_float_extra(
            "text_batch_split_delay_seconds", 10.0
        )
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}

    def _coerce_float_extra(self, key: str, default: float) -> float:
        """Read a float from ``config.extra``, guarding against bad/non-finite values.

        The result is fed directly to ``asyncio.sleep()``, so NaN/Inf and
        unparseable values fall back to ``default``.
        """
        import math

        value = self.config.extra.get(key) if getattr(self.config, "extra", None) else None
        if value is None:
            return float(default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(parsed) or parsed < 0:
            return float(default)
        return parsed

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """
        Start the WhatsApp bridge.
        
        This launches the Node.js bridge process and waits for it to be ready.
        """
        if not check_whatsapp_requirements():
            logger.warning("[%s] Node.js not found. WhatsApp requires Node.js.", self.name)
            self._set_fatal_error(
                "whatsapp_node_missing",
                "Node.js is not installed — install Node.js and re-run `hermes gateway`.",
                retryable=False,
            )
            return False
        
        bridge_path = Path(self._bridge_script)
        if not bridge_path.exists():
            logger.warning("[%s] Bridge script not found: %s", self.name, bridge_path)
            self._set_fatal_error(
                "whatsapp_bridge_missing",
                f"WhatsApp bridge script missing at {bridge_path}.",
                retryable=False,
            )
            return False

        # Pre-flight: skip the 30s bridge bootstrap entirely if the user
        # never finished pairing.  Without creds.json the bridge prints
        # QR codes to its log file and never reaches status:connected,
        # so every gateway restart paid the 30s timeout + queued WhatsApp
        # for indefinite retries.  Mark non-retryable so the user gets a
        # clear pairing message instead of the watcher
        # silently hammering an unconfigured platform.
        creds_path = self._session_path / "creds.json"
        if not creds_path.exists():
            logger.warning(
                "[%s] WhatsApp is enabled but not paired (no creds.json at %s). "
                "Pair from the dashboard or run `hermes whatsapp`; remove "
                "WHATSAPP_ENABLED from your .env to disable.",
                self.name, creds_path,
            )
            self._set_fatal_error(
                "whatsapp_not_paired",
                "WhatsApp enabled but not paired — pair from the dashboard or run `hermes whatsapp`.",
                retryable=False,
            )
            return False

        logger.info("[%s] Bridge found at %s", self.name, bridge_path)
        
        # Acquire scoped lock to prevent duplicate sessions
        lock_acquired = False
        try:
            if not self._acquire_platform_lock('whatsapp-session', str(self._session_path), 'WhatsApp session'):
                return False
            lock_acquired = True
        except Exception as e:
            logger.warning("[%s] Could not acquire session lock (non-fatal): %s", self.name, e)

        try:
            # Auto-install npm dependencies when node_modules is missing OR
            # package.json changed since the last install (e.g. after
            # `hermes update` bumps the Baileys pin).  The stamp file records
            # the package.json hash of the last successful install.
            bridge_dir = bridge_path.parent
            _pkg_json = bridge_dir / "package.json"
            _dep_stamp = bridge_dir / "node_modules" / ".hermes-pkg-hash"
            _pkg_hash = _file_content_hash(_pkg_json)
            _deps_fresh = False
            if (bridge_dir / "node_modules").exists():
                try:
                    _deps_fresh = (_dep_stamp.read_text().strip() == _pkg_hash) and bool(_pkg_hash)
                except OSError:
                    _deps_fresh = False
            if not _deps_fresh:
                print(f"[{self.name}] Installing WhatsApp bridge dependencies...")
                # Resolve npm path so Windows uses npm.cmd from the
                # Hermes-managed portable Node before falling back to PATH.
                _npm_bin = find_node_executable("npm") or "npm"
                try:
                    # Read timeout from environment variable, default to 300 seconds (5 minutes)
                    # to accommodate slower systems like Unraid NAS
                    npm_install_timeout = env_int("WHATSAPP_NPM_INSTALL_TIMEOUT", 300)
                    install_result = subprocess.run(
                        [_npm_bin, "install", "--silent"],
                        cwd=str(bridge_dir),
                        capture_output=True,
                        text=True,
                        timeout=npm_install_timeout,
                        env=with_hermes_node_path(),
                    )
                    if install_result.returncode != 0:
                        print(f"[{self.name}] npm install failed: {install_result.stderr}")
                        return False
                    print(f"[{self.name}] Dependencies installed")
                    if _pkg_hash:
                        try:
                            _dep_stamp.write_text(_pkg_hash)
                        except OSError:
                            pass  # Stamp is an optimization; install still succeeded
                except Exception as e:
                    print(f"[{self.name}] Failed to install dependencies: {e}")
                    return False

            # Ensure session directory exists
            self._session_path.mkdir(parents=True, exist_ok=True)
            
            # Check if bridge is already running and connected
            import aiohttp
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{self._bridge_port}/health",
                        timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            bridge_status = data.get("status", "unknown")
                            if bridge_status == "connected":
                                # Staleness handshake: only reuse a running
                                # bridge if it is serving the same bridge.js
                                # that is on disk right now.  A long-lived
                                # bridge survives gateway restarts AND
                                # `hermes update`, so without this check it
                                # keeps serving pre-update code forever
                                # (e.g. no inbound media download).  Old
                                # bridges that don't report scriptHash are
                                # treated as stale by definition.
                                running_hash = data.get("scriptHash", "")
                                disk_hash = _file_content_hash(bridge_path)
                                if running_hash and disk_hash and running_hash == disk_hash:
                                    print(f"[{self.name}] Using existing bridge (status: {bridge_status})")
                                    self._mark_connected()
                                    self._bridge_process = None  # Not managed by us
                                    self._http_session = aiohttp.ClientSession()
                                    self._poll_task = asyncio.create_task(self._poll_messages())
                                    return True
                                print(
                                    f"[{self.name}] Running bridge is stale "
                                    f"(running={running_hash or 'unversioned'}, disk={disk_hash}), restarting"
                                )
                            else:
                                print(f"[{self.name}] Bridge found but not connected (status: {bridge_status}), restarting")
            except Exception:
                pass  # Bridge not running, start a new one
            
            # Kill any orphaned bridge from a previous gateway run
            _kill_stale_bridge_by_pidfile(self._session_path)
            _kill_port_process(self._bridge_port)
            await asyncio.sleep(1)
            
            # Start the bridge process in its own process group.
            # Route output to a log file so QR codes, errors, and reconnection
            # messages are preserved for troubleshooting.
            whatsapp_mode = os.getenv("WHATSAPP_MODE", "self-chat")
            self._bridge_log = self._session_path.parent / "bridge.log"
            bridge_log_fh = open(self._bridge_log, "a", encoding="utf-8")
            self._bridge_log_fh = bridge_log_fh

            # Build bridge subprocess environment.
            # Pass WHATSAPP_REPLY_PREFIX from config.yaml so the Node bridge
            # can use it without the user needing to set a separate env var.
            # with_hermes_node_path() copies os.environ when called with no arg.
            bridge_env = with_hermes_node_path()
            if self._reply_prefix is not None:
                bridge_env["WHATSAPP_REPLY_PREFIX"] = self._reply_prefix
            # Pass the profile-aware cache directories so the bridge writes
            # media where the Python side reads it.  Without these the bridge
            # hardcodes ~/.hermes/{image,audio,document}_cache, which diverges
            # under HERMES_HOME overrides, profiles, and the new cache/ layout.
            from gateway.platforms.base import (
                get_audio_cache_dir as _get_audio_dir,
                get_document_cache_dir as _get_doc_dir,
                get_image_cache_dir as _get_img_dir,
            )
            bridge_env["HERMES_IMAGE_CACHE_DIR"] = str(_get_img_dir())
            bridge_env["HERMES_AUDIO_CACHE_DIR"] = str(_get_audio_dir())
            bridge_env["HERMES_DOCUMENT_CACHE_DIR"] = str(_get_doc_dir())

            self._bridge_process = subprocess.Popen(
                [
                    find_node_executable("node") or "node",
                    str(bridge_path),
                    "--port", str(self._bridge_port),
                    "--session", str(self._session_path),
                    "--mode", whatsapp_mode,
                ],
                stdout=bridge_log_fh,
                stderr=bridge_log_fh,
                env=bridge_env,
                **windows_detach_popen_kwargs(),
            )
            _write_bridge_pidfile(self._session_path, self._bridge_process.pid)
            
            # Wait for the bridge to connect to WhatsApp.
            # Phase 1: wait for the HTTP server to come up (up to 15s).
            # Phase 2: wait for WhatsApp status: connected (up to 15s more).
            import aiohttp
            http_ready = False
            data = {}
            for attempt in range(15):
                await asyncio.sleep(1)
                if self._bridge_process.poll() is not None:
                    print(f"[{self.name}] Bridge process died (exit code {self._bridge_process.returncode})")
                    print(f"[{self.name}] Check log: {self._bridge_log}")
                    self._close_bridge_log()
                    return False
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"http://127.0.0.1:{self._bridge_port}/health",
                            timeout=aiohttp.ClientTimeout(total=2)
                        ) as resp:
                            if resp.status == 200:
                                http_ready = True
                                data = await resp.json()
                                if data.get("status") == "connected":
                                    print(f"[{self.name}] Bridge ready (status: connected)")
                                    break
                except Exception:
                    continue

            if not http_ready:
                print(f"[{self.name}] Bridge HTTP server did not start in 15s")
                print(f"[{self.name}] Check log: {self._bridge_log}")
                self._close_bridge_log()
                return False
            
            # Phase 2: HTTP is up but WhatsApp may still be connecting.
            # Give it more time to authenticate with saved credentials.
            if data.get("status") != "connected":
                print(f"[{self.name}] Bridge HTTP ready, waiting for WhatsApp connection...")
                for attempt in range(15):
                    await asyncio.sleep(1)
                    if self._bridge_process.poll() is not None:
                        print(f"[{self.name}] Bridge process died during connection")
                        print(f"[{self.name}] Check log: {self._bridge_log}")
                        self._close_bridge_log()
                        return False
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                f"http://127.0.0.1:{self._bridge_port}/health",
                                timeout=aiohttp.ClientTimeout(total=2)
                            ) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    if data.get("status") == "connected":
                                        print(f"[{self.name}] Bridge ready (status: connected)")
                                        break
                    except Exception:
                        continue
                else:
                    # Still not connected — warn but proceed (bridge may
                    # auto-reconnect later, e.g. after a code 515 restart).
                    print(f"[{self.name}] ⚠ WhatsApp not connected after 30s")
                    print(f"[{self.name}]   Bridge log: {self._bridge_log}")
                    print(f"[{self.name}]   If session expired, re-pair: hermes whatsapp")
            
            # Create a persistent HTTP session for all bridge communication
            self._http_session = aiohttp.ClientSession()

            # Start message polling task
            self._poll_task = asyncio.create_task(self._poll_messages())
            
            self._mark_connected()
            print(f"[{self.name}] Bridge started on port {self._bridge_port}")
            return True
            
        except Exception as e:
            logger.error("[%s] Failed to start bridge: %s", self.name, e, exc_info=True)
            return False
        finally:
            if not self._running:
                if lock_acquired:
                    self._release_platform_lock()
                self._close_bridge_log()
    
    def _close_bridge_log(self) -> None:
        """Close the bridge log file handle if open."""
        if self._bridge_log_fh:
            try:
                self._bridge_log_fh.close()
            except Exception:
                pass
            self._bridge_log_fh = None

    async def _check_managed_bridge_exit(self) -> Optional[str]:
        """Return a fatal error message if the managed bridge child exited."""
        if self._bridge_process is None:
            return None

        returncode = self._bridge_process.poll()
        if returncode is None:
            return None

        # Planned shutdown: disconnect() sets _shutting_down before it sends
        # SIGTERM to the bridge, so a returncode of -15 (SIGTERM), -2 (SIGINT),
        # or 0 (clean exit) at that point is expected, not a crash. Treat it
        # as informational and skip the fatal-error path.
        # getattr-with-default keeps tests that construct the adapter via
        # ``WhatsAppAdapter.__new__`` (bypassing __init__) working without
        # every _make_adapter() helper having to seed the attribute.
        if getattr(self, "_shutting_down", False) and returncode in {0, -2, -15}:
            logger.info(
                "[%s] Bridge exited during shutdown (code %d).",
                self.name,
                returncode,
            )
            return None

        message = f"WhatsApp bridge process exited unexpectedly (code {returncode})."
        if not self.has_fatal_error:
            logger.error("[%s] %s", self.name, message)
            self._set_fatal_error("whatsapp_bridge_exited", message, retryable=True)
            self._close_bridge_log()
            await self._notify_fatal_error()
        return self.fatal_error_message or message

    async def disconnect(self) -> None:
        """Stop the WhatsApp bridge and clean up any orphaned processes."""
        # Flip the shutdown flag BEFORE signalling the child so the exit-check
        # path (which runs from other tasks like send() and the poll loop)
        # doesn't race us and report the intentional termination as fatal.
        self._shutting_down = True
        if self._bridge_process:
            try:
                try:
                    _terminate_bridge_process(self._bridge_process, force=False)
                except (ProcessLookupError, PermissionError):
                    self._bridge_process.terminate()
                await asyncio.sleep(1)
                if self._bridge_process.poll() is None:
                    try:
                        _terminate_bridge_process(self._bridge_process, force=True)
                    except (ProcessLookupError, PermissionError):
                        self._bridge_process.kill()
            except Exception as e:
                print(f"[{self.name}] Error stopping bridge: {e}")
        else:
            # Bridge was not started by us, don't kill it
            print(f"[{self.name}] Disconnecting (external bridge left running)")

        # Clean up PID file
        try:
            (self._session_path / "bridge.pid").unlink(missing_ok=True)
        except OSError:
            pass

        # Cancel the poll task explicitly
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
        self._poll_task = None

        # Close the persistent HTTP session
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        self._http_session = None

        self._release_platform_lock()

        self._mark_disconnected()
        self._bridge_process = None
        self._close_bridge_log()
        print(f"[{self.name}] Disconnected")
    
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Send a message via the WhatsApp bridge.

        Formats markdown for WhatsApp, splits long messages into chunks
        that preserve code block boundaries, and sends each chunk sequentially.
        """
        if not self._running or not self._http_session:
            return SendResult(success=False, error="Not connected")
        bridge_exit = await self._check_managed_bridge_exit()
        if bridge_exit:
            return SendResult(success=False, error=bridge_exit)

        if not content or not content.strip():
            return SendResult(success=True, message_id=None)

        chat_id = to_whatsapp_jid(chat_id)

        try:
            import aiohttp

            # Format and chunk the message
            formatted = self.format_message(content)
            chunks = self.truncate_message(formatted, self._outgoing_chunk_limit())

            sent_message_ids: list[str] = []
            last_message_id = None
            for idx, chunk in enumerate(chunks):
                payload: Dict[str, Any] = {
                    "chatId": chat_id,
                    "message": chunk,
                }
                if reply_to and idx == 0:
                    # Only reply-to on the first text chunk, even if the bridge
                    # response omits a parseable message id.
                    payload["replyTo"] = reply_to

                async with self._http_session.post(
                    f"http://127.0.0.1:{self._bridge_port}/send",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        last_message_id = data.get("messageId")
                        if last_message_id:
                            sent_message_ids.append(str(last_message_id))
                    else:
                        error = await resp.text()
                        return SendResult(success=False, error=error)

                # Small delay between chunks to avoid rate limiting
                if len(chunks) > 1:
                    await asyncio.sleep(0.3)

            return SendResult(
                success=True,
                message_id=last_message_id,
                continuation_message_ids=tuple(sent_message_ids[:-1]),
                raw_response={"message_ids": sent_message_ids},
            )
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Edit a previously sent message via the WhatsApp bridge."""
        if not self._running or not self._http_session:
            return SendResult(success=False, error="Not connected")
        bridge_exit = await self._check_managed_bridge_exit()
        if bridge_exit:
            return SendResult(success=False, error=bridge_exit)
        try:
            import aiohttp
            async with self._http_session.post(
                f"http://127.0.0.1:{self._bridge_port}/edit",
                json={
                    "chatId": to_whatsapp_jid(chat_id),
                    "messageId": message_id,
                    "message": content,
                },
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    return SendResult(success=True, message_id=message_id)
                else:
                    error = await resp.text()
                    return SendResult(success=False, error=error)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def _send_media_to_bridge(
        self,
        chat_id: str,
        file_path: str,
        media_type: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> SendResult:
        """Send any media file via bridge /send-media endpoint."""
        if not self._running or not self._http_session:
            return SendResult(success=False, error="Not connected")
        bridge_exit = await self._check_managed_bridge_exit()
        if bridge_exit:
            return SendResult(success=False, error=bridge_exit)
        try:
            import aiohttp

            if not os.path.exists(file_path):
                return SendResult(success=False, error=f"File not found: {file_path}")

            payload: Dict[str, Any] = {
                "chatId": to_whatsapp_jid(chat_id),
                "filePath": file_path,
                "mediaType": media_type,
            }
            if caption:
                payload["caption"] = caption
            if file_name:
                payload["fileName"] = file_name

            async with self._http_session.post(
                f"http://127.0.0.1:{self._bridge_port}/send-media",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return SendResult(
                        success=True,
                        message_id=data.get("messageId"),
                        raw_response=data,
                    )
                else:
                    error = await resp.text()
                    return SendResult(success=False, error=error)

        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_poll(
        self,
        chat_id: str,
        question: str,
        options: list[str],
        *,
        selectable_count: int = 1,
    ) -> SendResult:
        """Send a native WhatsApp poll via the Baileys bridge.

        This is a low-level transport primitive only. Gateway approval UX must
        remain gateway-owned and add text fallback plus explicit confirmation
        semantics before approval prompts are ever mapped onto polls.
        """
        if not self._running or not self._http_session:
            return SendResult(success=False, error="Not connected")
        bridge_exit = await self._check_managed_bridge_exit()
        if bridge_exit:
            return SendResult(success=False, error=bridge_exit)
        try:
            import aiohttp

            payload: Dict[str, Any] = {
                "chatId": to_whatsapp_jid(chat_id),
                "question": question,
                "options": list(options or []),
                "selectableCount": selectable_count,
            }
            async with self._http_session.post(
                f"http://127.0.0.1:{self._bridge_port}/send-poll",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return SendResult(
                        success=True,
                        message_id=data.get("messageId"),
                        raw_response=data,
                    )
                error = await resp.text()
                return SendResult(success=False, error=error)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Render multiple-choice clarify as a native WhatsApp poll.

        The gateway registers the pending clarify before calling this method.
        When Baileys later emits a poll_update with the selected option as
        message text, the normal clarify text-intercept resolves the pending
        question and the blocked agent continues. Open-ended clarifies use the
        text fallback so the user's next typed message is captured.
        """
        clean_choices = [str(choice).strip() for choice in (choices or []) if str(choice).strip()]
        if 2 <= len(clean_choices) <= 12:
            result = await self.send_poll(
                chat_id,
                str(question or "").strip(),
                clean_choices,
                selectable_count=1,
            )
            if result.success:
                return result
            logger.warning(
                "[%s] Native WhatsApp clarify poll failed; falling back to text: %s",
                self.name,
                result.error,
            )
        return await super().send_clarify(
            chat_id=chat_id,
            question=question,
            choices=choices,
            clarify_id=clarify_id,
            session_key=session_key,
            metadata=metadata,
        )

    async def send_location(
        self,
        chat_id: str,
        latitude: float,
        longitude: float,
        *,
        name: Optional[str] = None,
        address: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a native WhatsApp location pin via the Baileys bridge."""
        if not self._running or not self._http_session:
            return SendResult(success=False, error="Not connected")
        bridge_exit = await self._check_managed_bridge_exit()
        if bridge_exit:
            return SendResult(success=False, error=bridge_exit)
        try:
            import aiohttp

            payload: Dict[str, Any] = {
                "chatId": to_whatsapp_jid(chat_id),
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
            if name:
                payload["name"] = name
            if address:
                payload["address"] = address
            async with self._http_session.post(
                f"http://127.0.0.1:{self._bridge_port}/send-location",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return SendResult(
                        success=True,
                        message_id=data.get("messageId"),
                        raw_response=data,
                    )
                error = await resp.text()
                return SendResult(success=False, error=error)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Download image URL to cache, send natively via bridge.

        ``metadata`` is accepted to honor the base-class contract — the
        batch sender ``send_multiple_images`` passes it through to every
        send path. The bridge media call doesn't use it, matching the
        sibling overrides (send_video / send_voice / send_document).
        """
        try:
            local_path = await cache_image_from_url(image_url)
            return await self._send_media_to_bridge(chat_id, local_path, "image", caption)
        except Exception:
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file natively via bridge."""
        return await self._send_media_to_bridge(chat_id, image_path, "image", caption)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video natively via bridge — plays inline in WhatsApp."""
        return await self._send_media_to_bridge(chat_id, video_path, "video", caption)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an audio file as a WhatsApp voice message via bridge."""
        return await self._send_media_to_bridge(chat_id, audio_path, "audio", caption)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a document/file as a downloadable attachment via bridge."""
        return await self._send_media_to_bridge(
            chat_id, file_path, "document", caption,
            file_name or os.path.basename(file_path),
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send typing indicator via bridge."""
        if not self._running or not self._http_session:
            return
        if await self._check_managed_bridge_exit():
            return
        
        try:
            import aiohttp

            # Must wrap in `async with` — a bare `await session.post(...)`
            # leaves the response object alive until GC, holding its TCP
            # socket in CLOSE_WAIT. See #18451.
            async with self._http_session.post(
                f"http://127.0.0.1:{self._bridge_port}/typing",
                json={"chatId": to_whatsapp_jid(chat_id)},
                timeout=aiohttp.ClientTimeout(total=5)
            ):
                pass
        except Exception:
            pass  # Ignore typing indicator failures
    
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a WhatsApp chat."""
        if not self._running or not self._http_session:
            return {"name": "Unknown", "type": "dm"}
        if await self._check_managed_bridge_exit():
            return {"name": chat_id, "type": "dm"}
        
        try:
            import aiohttp

            async with self._http_session.get(
                f"http://127.0.0.1:{self._bridge_port}/chat/{to_whatsapp_jid(chat_id)}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "name": data.get("name", chat_id),
                        "type": "group" if data.get("isGroup") else "dm",
                        "participants": data.get("participants", []),
                    }
        except Exception as e:
            logger.debug("Could not get WhatsApp chat info for %s: %s", chat_id, e)
        
        return {"name": chat_id, "type": "dm"}
    
    async def _poll_messages(self) -> None:
        """Poll the bridge for incoming messages."""
        import aiohttp

        while self._running:
            if not self._http_session:
                break
            bridge_exit = await self._check_managed_bridge_exit()
            if bridge_exit:
                print(f"[{self.name}] {bridge_exit}")
                break
            try:
                async with self._http_session.get(
                    f"http://127.0.0.1:{self._bridge_port}/messages",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        messages = await resp.json()
                        for msg_data in messages:
                            event = await self._build_message_event(msg_data)
                            if event:
                                if event.message_type == MessageType.TEXT:
                                    self._enqueue_text_event(event)
                                else:
                                    await self.handle_message(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                bridge_exit = await self._check_managed_bridge_exit()
                if bridge_exit:
                    print(f"[{self.name}] {bridge_exit}")
                    break
                print(f"[{self.name}] Poll error: {e}")
                await asyncio.sleep(5)
            
            await asyncio.sleep(1)  # Poll interval

    # ── Text debounce batching ──────────────────────────────────────

    _SPLIT_THRESHOLD = 6000  # WhatsApp supports ~65K chars; generous threshold

    def _text_batch_key(self, event: MessageEvent) -> str:
        """Session-scoped key for text message batching."""
        from gateway.session import build_session_key
        return build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Buffer a text event and reset the flush timer.

        When WhatsApp delivers rapid-fire messages (e.g. forwarded
        batches), this concatenates them and waits for a short quiet
        period before dispatching the combined message.
        """
        key = self._text_batch_key(event)
        existing = self._pending_text_batches.get(key)
        chunk_len = len(event.text or "")
        if existing is None:
            event._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            self._pending_text_batches[key] = event
        else:
            if event.text:
                existing.text = f"{existing.text}\n{event.text}" if existing.text else event.text
            existing._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            if event.media_urls:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)

        prior_task = self._pending_text_batch_tasks.get(key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_text_batch_tasks[key] = asyncio.create_task(
            self._flush_text_batch(key)
        )

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for quiet period then dispatch aggregated text."""
        current_task = asyncio.current_task()
        try:
            pending = self._pending_text_batches.get(key)
            last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
            if last_len >= self._SPLIT_THRESHOLD:
                delay = self._text_batch_split_delay_seconds
            else:
                delay = self._text_batch_delay_seconds
            await asyncio.sleep(delay)
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    async def _build_message_event(self, data: Dict[str, Any]) -> Optional[MessageEvent]:
        """Build a MessageEvent from bridge message data, downloading images to cache."""
        try:
            if not self._should_process_message(data):
                return None

            # Determine message type
            msg_type = MessageType.TEXT
            media_type = str(data.get("mediaType", "") or "")
            if media_type in {"location", "live_location"}:
                msg_type = MessageType.LOCATION
            elif media_type == "sticker":
                msg_type = MessageType.STICKER
            elif data.get("hasMedia"):
                if "image" in media_type:
                    msg_type = MessageType.PHOTO
                elif "video" in media_type:
                    msg_type = MessageType.VIDEO
                elif "ptt" in media_type:  # ptt = WhatsApp voice note
                    msg_type = MessageType.VOICE
                elif "audio" in media_type:
                    msg_type = MessageType.AUDIO
                else:
                    msg_type = MessageType.DOCUMENT
            
            # Determine chat type
            is_group = data.get("isGroup", False)
            chat_type = "group" if is_group else "dm"
            
            # Build source
            source = self.build_source(
                chat_id=data.get("chatId", ""),
                chat_name=data.get("chatName"),
                chat_type=chat_type,
                user_id=data.get("senderId"),
                user_name=data.get("senderName"),
            )
            
            # Download media URLs to the local cache so agent tools
            # can access them reliably regardless of URL expiration.
            raw_urls = data.get("mediaUrls", [])
            cached_urls = []
            media_types = []
            for url in raw_urls:
                bridge_mime = str(data.get("mime") or "").strip()
                if msg_type == MessageType.PHOTO and url.startswith(("http://", "https://")):
                    try:
                        cached_path = await cache_image_from_url(url, ext=".jpg")
                        cached_urls.append(cached_path)
                        media_types.append(bridge_mime or "image/jpeg")
                        print(f"[{self.name}] Cached user image: {cached_path}", flush=True)
                    except Exception as e:
                        print(f"[{self.name}] Failed to cache image: {e}", flush=True)
                        cached_urls.append(url)
                        media_types.append(bridge_mime or "image/jpeg")
                elif msg_type == MessageType.PHOTO and os.path.isabs(url):
                    # Local file path — bridge already downloaded the image
                    if _is_allowed_bridge_path(url):
                        cached_urls.append(url)
                        media_types.append("image/jpeg")
                        print(f"[{self.name}] Using bridge-cached image: {url}", flush=True)
                    else:
                        print(f"[{self.name}] Rejected bridge image path outside cache dir: {url}", flush=True)
                elif msg_type == MessageType.VOICE and url.startswith(("http://", "https://")):
                    try:
                        cached_path = await cache_audio_from_url(url, ext=".ogg")
                        cached_urls.append(cached_path)
                        media_types.append(bridge_mime or ("audio/ogg" if msg_type == MessageType.VOICE else "audio/mpeg"))
                        print(f"[{self.name}] Cached user audio: {cached_path}", flush=True)
                    except Exception as e:
                        print(f"[{self.name}] Failed to cache audio: {e}", flush=True)
                        cached_urls.append(url)
                        media_types.append(bridge_mime or ("audio/ogg" if msg_type == MessageType.VOICE else "audio/mpeg"))
                elif msg_type in {MessageType.VOICE, MessageType.AUDIO} and os.path.isabs(url):
                    # Local file path — bridge already downloaded the audio
                    if _is_allowed_bridge_path(url):
                        cached_urls.append(url)
                        media_types.append("audio/ogg")
                        print(f"[{self.name}] Using bridge-cached audio: {url}", flush=True)
                    else:
                        print(f"[{self.name}] Rejected bridge audio path outside cache dir: {url}", flush=True)
                elif msg_type == MessageType.DOCUMENT and os.path.isabs(url):
                    # Local file path — bridge already downloaded the document
                    if _is_allowed_bridge_path(url):
                        cached_urls.append(url)
                        ext = Path(url).suffix.lower()
                        mime = SUPPORTED_DOCUMENT_TYPES.get(ext, "application/octet-stream")
                        media_types.append(mime)
                        print(f"[{self.name}] Using bridge-cached document: {url}", flush=True)
                    else:
                        print(f"[{self.name}] Rejected bridge document path outside cache dir: {url}", flush=True)
                elif msg_type == MessageType.VIDEO and os.path.isabs(url):
                    if _is_allowed_bridge_path(url):
                        cached_urls.append(url)
                        media_types.append("video/mp4")
                        print(f"[{self.name}] Using bridge-cached video: {url}", flush=True)
                    else:
                        print(f"[{self.name}] Rejected bridge video path outside cache dir: {url}", flush=True)
                else:
                    cached_urls.append(url)
                    media_types.append("unknown")

            # For text-readable documents, inject file content directly into
            # the message text so the agent can read it inline.
            # Cap at 100KB to match Telegram/Discord/Slack behaviour.
            body = data.get("body", "")
            if data.get("isGroup"):
                body = self._clean_bot_mention_text(body, data)

            # If this is a reply, keep the quoted message in structured fields
            # only. GatewayRunner._prepare_inbound_message_text owns rendering
            # the `[Replying to: ...]` pointer for every platform; pre-rendering
            # it here makes WhatsApp replies show the quote twice.
            quoted_text = str(data.get("quotedText") or "").strip()
            reply_to_text = quoted_text or None
            reply_to_message_id = None
            reply_to_author_id = None
            reply_to_is_own_message = False
            if data.get("hasQuotedMessage"):
                raw_reply_id = data.get("quotedMessageId")
                if raw_reply_id is not None:
                    reply_to_message_id = str(raw_reply_id)
                quoted_participant = self._normalize_whatsapp_id(data.get("quotedParticipant"))
                if quoted_participant:
                    reply_to_author_id = quoted_participant
                reply_to_is_own_message = self._message_is_reply_to_bot(data)
            MAX_TEXT_INJECT_BYTES = 100 * 1024
            if msg_type == MessageType.DOCUMENT and cached_urls:
                for doc_path in cached_urls:
                    ext = Path(doc_path).suffix.lower()
                    if ext in {".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".log", ".py", ".js", ".ts", ".html", ".css"}:
                        try:
                            file_size = Path(doc_path).stat().st_size
                            if file_size > MAX_TEXT_INJECT_BYTES:
                                print(f"[{self.name}] Skipping text injection for {doc_path} ({file_size} bytes > {MAX_TEXT_INJECT_BYTES})", flush=True)
                                continue
                            content = Path(doc_path).read_text(encoding="utf-8", errors="replace")
                            fname = Path(doc_path).name
                            # Remove the doc_<hex>_ prefix for display
                            display_name = fname
                            if "_" in fname:
                                parts = fname.split("_", 2)
                                if len(parts) >= 3:
                                    display_name = parts[2]
                            injection = f"[Content of {display_name}]:\n{content}"
                            if body:
                                body = f"{injection}\n\n{body}"
                            else:
                                body = injection
                            print(f"[{self.name}] Injected text content from: {doc_path}", flush=True)
                        except Exception as e:
                            print(f"[{self.name}] Failed to read document text: {e}", flush=True)

            metadata: Dict[str, Any] = {}
            # The bridge sets ``fromOwner: true`` on inbound fromMe messages
            # that look owner-typed (linked-device send, not echoed from our
            # own /send).  Surfaced under a platform-prefixed key so plugins
            # can detect "owner just replied in this customer chat" without
            # having to peek at raw_message.  We also prefix ``MessageEvent.text``
            # with ``[owner reply] `` here so the marker survives any downstream
            # failure (e.g. handover-rule errors that bypass silent_ingest).
            # Gated by ``WHATSAPP_FORWARD_OWNER_MESSAGES`` at the bridge layer;
            # metadata + text tagging are unconditional when the flag is present
            # so a future producer can set it without adapter changes.
            if data.get("fromOwner"):
                metadata["whatsapp_from_owner"] = True
                if not body.startswith(_OWNER_REPLY_PREFIX):
                    body = f"{_OWNER_REPLY_PREFIX}{body}"

            return MessageEvent(
                text=body,
                message_type=msg_type,
                source=source,
                raw_message=data,
                message_id=data.get("messageId"),
                media_urls=cached_urls,
                media_types=media_types,
                metadata=metadata,
            )
        except Exception as e:
            print(f"[{self.name}] Error building event: {e}")
            return None


# ──────────────────────────────────────────────────────────────────────────
# Plugin migration glue (#41112 / #3823)
#
# Added when the WhatsApp adapter moved from gateway/platforms/whatsapp.py into
# this bundled plugin. Mirrors the Discord (#24356) / Slack migrations: a
# register(ctx) entry point plus hook implementations that replace the
# per-platform core touchpoints (the Platform.WHATSAPP elif in gateway/run.py,
# the whatsapp_cfg YAML→env block + _PLATFORM_CONNECTED_CHECKERS entry in
# gateway/config.py, the _setup_whatsapp wizard + _PLATFORMS["whatsapp"] static
# dict in hermes_cli/gateway.py, and the _send_whatsapp dispatch in
# tools/send_message_tool.py).  WhatsApp auth is handled by the Node.js bridge,
# so is_connected is always True (matches the legacy checker).
# ──────────────────────────────────────────────────────────────────────────


_WA_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_WA_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_WA_AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac"}


def _bridge_media_type(file_path: str, is_voice: bool, force_document: bool) -> str:
    """Map a local media file to the bridge /send-media ``mediaType``.

    Returns one of ``image`` | ``video`` | ``audio`` | ``document`` so the
    Baileys bridge renders the right native WhatsApp message kind. Voice notes
    and audio files route to ``audio``; ``force_document`` (the [[as_document]]
    directive) forces every file to ``document`` regardless of extension.
    """
    if force_document:
        return "document"
    ext = os.path.splitext(file_path)[1].lower()
    if is_voice or ext in _WA_AUDIO_EXTS:
        return "audio"
    if ext in _WA_IMAGE_EXTS:
        return "image"
    if ext in _WA_VIDEO_EXTS:
        return "video"
    return "document"


async def _standalone_send(
    pconfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """Out-of-process WhatsApp delivery via the local bridge HTTP API.

    Implements the standalone_sender_fn contract so deliver=whatsapp cron jobs
    succeed when cron runs separately from the gateway. Replaces the legacy
    _send_whatsapp helper.
    """
    extra = getattr(pconfig, "extra", {}) or {}
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        bridge_port = extra.get("bridge_port", 3000)
        normalized_chat_id = to_whatsapp_jid(chat_id)
        media = media_files or []
        text = message or ""
        last_message_id = None
        async with aiohttp.ClientSession() as session:
            # 1) Text first (skip the /send call when this chunk is media-only).
            if text.strip():
                async with session.post(
                    f"http://localhost:{bridge_port}/send",
                    json={"chatId": normalized_chat_id, "message": text},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        return {"error": f"WhatsApp bridge error ({resp.status}): {body}"}
                    data = await resp.json()
                    last_message_id = data.get("messageId")

            # 2) Each media file as a native attachment via /send-media. The
            # bridge maps mediaType -> image/video/audio/document message kinds
            # so PNG/JPEG/WebP/GIF arrive as inline images, MP4 as a video
            # bubble, and ogg/opus as a voice note — not a file/document.
            for media_path, is_voice in media:
                if not os.path.exists(media_path):
                    return {"error": f"WhatsApp media file not found: {media_path}"}
                media_type = _bridge_media_type(media_path, is_voice, force_document)
                payload: Dict[str, Any] = {
                    "chatId": normalized_chat_id,
                    "filePath": media_path,
                    "mediaType": media_type,
                }
                if media_type == "document":
                    payload["fileName"] = os.path.basename(media_path)
                async with session.post(
                    f"http://localhost:{bridge_port}/send-media",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        return {"error": f"WhatsApp media error ({resp.status}): {body}"}
                    data = await resp.json()
                    last_message_id = data.get("messageId") or last_message_id

        return {
            "success": True,
            "platform": "whatsapp",
            "chat_id": normalized_chat_id,
            "message_id": last_message_id,
        }
    except Exception as e:
        return {"error": f"WhatsApp send failed: {e}"}


def interactive_setup() -> None:
    """Guide the user through WhatsApp setup.

    Replaces the central _setup_whatsapp in hermes_cli/gateway.py and the
    static _PLATFORMS["whatsapp"] dict. CLI helpers are lazy-imported so the
    plugin's module-load surface stays minimal.
    """
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli.cli_output import (
        prompt,
        prompt_yes_no,
        print_header,
        print_info,
        print_success,
    )

    print_header("WhatsApp")
    print_info("WhatsApp uses a local Node.js bridge (WhatsApp Web client).")
    print_info("Start the bridge separately; the gateway connects to it over HTTP.")
    existing = get_env_value("WHATSAPP_ENABLED")
    if existing and existing.lower() in {"true", "1", "yes"}:
        print_info("WhatsApp: already enabled")
        if not prompt_yes_no("Reconfigure WhatsApp?", False):
            return

    if prompt_yes_no("Enable WhatsApp?", True):
        save_env_value("WHATSAPP_ENABLED", "true")
        print_success("WhatsApp enabled")
    else:
        save_env_value("WHATSAPP_ENABLED", "false")
        print_info("WhatsApp left disabled")
        return

    allowed_users = prompt(
        "Allowed user IDs (comma-separated, leave empty for no allowlist)"
    )
    if allowed_users:
        save_env_value("WHATSAPP_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("WhatsApp allowlist configured")

    home_channel = prompt("Home chat ID for cron delivery (leave empty to skip)")
    if home_channel:
        save_env_value("WHATSAPP_HOME_CHANNEL", home_channel.strip())


def _apply_yaml_config(yaml_cfg: dict, whatsapp_cfg: dict) -> dict | None:
    """Translate config.yaml whatsapp: keys into WHATSAPP_* env vars.

    Implements the apply_yaml_config_fn contract (#24849). Mirrors the legacy
    whatsapp_cfg block from gateway/config.py::load_gateway_config(). Env vars
    take precedence over YAML. Returns None — everything flows through env.
    """
    import json as _json
    if "require_mention" in whatsapp_cfg and not os.getenv("WHATSAPP_REQUIRE_MENTION"):
        os.environ["WHATSAPP_REQUIRE_MENTION"] = str(whatsapp_cfg["require_mention"]).lower()
    if "mention_patterns" in whatsapp_cfg and not os.getenv("WHATSAPP_MENTION_PATTERNS"):
        os.environ["WHATSAPP_MENTION_PATTERNS"] = _json.dumps(whatsapp_cfg["mention_patterns"])
    frc = whatsapp_cfg.get("free_response_chats")
    if frc is not None and not os.getenv("WHATSAPP_FREE_RESPONSE_CHATS"):
        if isinstance(frc, list):
            frc = ",".join(str(v) for v in frc)
        os.environ["WHATSAPP_FREE_RESPONSE_CHATS"] = str(frc)
    if "dm_policy" in whatsapp_cfg and not os.getenv("WHATSAPP_DM_POLICY"):
        os.environ["WHATSAPP_DM_POLICY"] = str(whatsapp_cfg["dm_policy"]).lower()
    af = whatsapp_cfg.get("allow_from")
    if af is not None and not os.getenv("WHATSAPP_ALLOWED_USERS"):
        if isinstance(af, list):
            af = ",".join(str(v) for v in af)
        os.environ["WHATSAPP_ALLOWED_USERS"] = str(af)
    if "group_policy" in whatsapp_cfg and not os.getenv("WHATSAPP_GROUP_POLICY"):
        os.environ["WHATSAPP_GROUP_POLICY"] = str(whatsapp_cfg["group_policy"]).lower()
    gaf = whatsapp_cfg.get("group_allow_from")
    if gaf is not None and not os.getenv("WHATSAPP_GROUP_ALLOWED_USERS"):
        if isinstance(gaf, list):
            gaf = ",".join(str(v) for v in gaf)
        os.environ["WHATSAPP_GROUP_ALLOWED_USERS"] = str(gaf)
    return None


def _is_connected(config) -> bool:
    """WhatsApp is considered connected when the user has explicitly enabled it
    via ``WHATSAPP_ENABLED`` (or the YAML-bridged equivalent on the config).

    Auth itself is handled by the external Node.js bridge — we can't verify the
    bridge token here — so the opt-in flag is the connection signal. The legacy
    built-in path keyed off ``WHATSAPP_ENABLED`` in both the connected-platforms
    check and the setup-status display; returning an unconditional True here
    would make WhatsApp always show as "configured" in ``hermes setup`` even
    when the user never enabled it. #41112.
    """
    extra = getattr(config, "extra", {}) or {}
    if config is not None and getattr(config, "enabled", False) and extra:
        # An explicitly-enabled PlatformConfig with seeded extras (e.g. from
        # YAML) counts as configured.
        return True
    # Read via hermes_cli.gateway.get_env_value (not os.getenv) so setup-status
    # callers that patch get_env_value — and the gateway connected-platforms
    # check — observe the same value. Matches the discord/slack plugin pattern.
    import hermes_cli.gateway as gateway_mod
    val = (gateway_mod.get_env_value("WHATSAPP_ENABLED") or "").strip().lower()
    return val in {"true", "1", "yes"}


def _build_adapter(config):
    """Factory wrapper that constructs WhatsAppAdapter from a PlatformConfig."""
    return WhatsAppAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="whatsapp",
        label="WhatsApp",
        adapter_factory=_build_adapter,
        check_fn=check_whatsapp_requirements,
        is_connected=_is_connected,
        required_env=["WHATSAPP_ENABLED"],
        install_hint="WhatsApp requires a Node.js bridge — see the WhatsApp messaging docs",
        setup_fn=interactive_setup,
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="WHATSAPP_ALLOWED_USERS",
        allow_all_env="WHATSAPP_ALLOW_ALL_USERS",
        cron_deliver_env_var="WHATSAPP_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=4096,
        emoji="💬",
        allow_update_command=True,
    )
