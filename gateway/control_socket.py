"""
Gateway control socket — the gateway-owned local coordination surface.

Migration step 1 of the #92091 design: every other process on the machine
(the updater, `hermes serve`/dashboard, the Desktop app) currently discovers
gateway identity/state by scanning the process table and string-matching argv
or by reading ``gateway_state.json`` (which can outlive its writer). This
module gives the gateway an OWNED contract instead: a local-only socket the
gateway process creates at startup and removes on clean shutdown, answering
versioned JSON verbs. A connectable socket with a well-formed ``identify``
answer IS liveness — no PID-reuse heuristics.

v1 verbs (observation only — no behavior change for the gateway):

- ``identify`` → pid, profile label, hermes_home, code_sha/code_version
  (the #91283 stamps, now queryable live), supervisor kind, served profiles,
  start_time, protocol version.
- ``status``   → the live runtime-status payload (what ``gateway_state.json``
  holds today, but answered by the process itself, race-free).

Transport:

- POSIX: Unix domain socket at ``$HERMES_HOME/gateway.sock``. When the home
  path is too long for ``sun_path`` (~104 bytes on macOS/BSD), the socket is
  bound in the system temp dir and a pointer file
  ``$HERMES_HOME/gateway.sock.path`` records the real location; clients
  follow the pointer transparently.
- Windows: named pipe ``\\\\.\\pipe\\hermes-gateway-<home-hash>`` served via
  the proactor event loop. Same trust model (per-user namespace).

Never a TCP port. Filesystem/pipe ACLs are the auth boundary — the same
trust model as ``gateway_state.json`` today.

v1 wire contract: ONE request per connection — a single JSON line in, a
single JSON line out, then the server closes. Clients must not rely on
keep-alive or pipelining. Verb handlers may touch disk (they run in an
executor server-side) but must stay fast; the client budget is small.

Consumers (``hermes update --plan`` inventory, the post-update fleet version
matrix) PREFER the socket when it answers and fall back to the existing
state-file/scan layer when it doesn't — old gateways mid-upgrade and crashed
processes keep working exactly as before. The scan layer is demoted, not
deleted.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import socket
import stat
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CONTROL_PROTOCOL_VERSION = 1

_SOCKET_FILENAME = "gateway.sock"
_POINTER_FILENAME = "gateway.sock.path"
_IS_WINDOWS = sys.platform == "win32"

# Practical sun_path limit: 104 on macOS/BSD, 108 on Linux. Stay under the
# smaller bound with margin for the NUL terminator.
_MAX_UNIX_PATH = 100

# Requests and responses are single JSON lines. Bound them so a misbehaving
# peer can't balloon gateway memory.
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_RESPONSE_BYTES = 512 * 1024

_DEFAULT_CLIENT_TIMEOUT = 2.0


# ---------------------------------------------------------------------------
# Path resolution (shared by server and client)
# ---------------------------------------------------------------------------

def _home_hash(home: Path) -> str:
    canonical = os.path.normcase(str(Path(home).expanduser().resolve(strict=False)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def windows_pipe_name(home: Path) -> str:
    """Per-HERMES_HOME named pipe path (Windows transport)."""
    return rf"\\.\pipe\hermes-gateway-{_home_hash(home)}"


def _pointer_path(home: Path) -> Path:
    return Path(home) / _POINTER_FILENAME


def _default_socket_path(home: Path) -> Path:
    return Path(home) / _SOCKET_FILENAME


def _fallback_socket_path(home: Path) -> Path:
    """Short temp-dir path for homes whose direct socket path exceeds sun_path.

    Prefers ``tempfile.gettempdir()``; when even that yields a too-long path
    (deep $TMPDIR), falls back to ``/tmp`` on POSIX. If nothing fits, the
    tempdir candidate is returned anyway — bind will fail non-fatally and
    consumers use the scan layer.
    """
    name = f"hermes-gw-{_home_hash(home)}.sock"
    candidates = [Path(tempfile.gettempdir()) / name]
    if not _IS_WINDOWS:
        candidates.append(Path("/tmp") / name)
    for candidate in candidates:
        if len(str(candidate).encode("utf-8")) <= _MAX_UNIX_PATH:
            return candidate
    return candidates[0]


def resolve_server_socket_path(home: Path) -> tuple[Path, Optional[Path]]:
    """Where the server should bind, plus the pointer file to write (or None).

    Returns ``(bind_path, pointer_file)``. ``pointer_file`` is non-None only
    when the direct in-home path is too long and the temp-dir fallback is in
    use — the server must then persist the real location for clients.
    """
    direct = _default_socket_path(home)
    if len(str(direct).encode("utf-8")) <= _MAX_UNIX_PATH:
        return direct, None
    return _fallback_socket_path(home), _pointer_path(home)


def _current_euid() -> Optional[int]:
    """The effective uid, or None where the platform has no such concept.

    ``os.geteuid`` is absent on Windows, so it is resolved through
    ``getattr`` rather than referenced directly — the Windows-footgun lint
    flags a bare call even inside a platform-gated branch, and looking it up
    removes the hazard instead of annotating it.
    """
    getter = getattr(os, "geteuid", None)
    return getter() if getter is not None else None


def _peer_euid(sock: "socket.socket") -> Optional[int]:
    """The connected peer's euid, where the platform reports it.

    The path check is a filter, not a guarantee: a socket can be swapped
    between the ``stat`` and the ``connect``. ``SO_PEERCRED`` answers for the
    process actually on the other end, which closes that race regardless of
    filesystem timing.

    Linux-only in practice — and that is precisely the platform carrying the
    exposure, because a shared ``/tmp`` is what makes the fallback path
    reachable by another account at all (macOS hands each user a private
    ``gettempdir()``). BSD/macOS report peer credentials through
    ``LOCAL_PEERCRED`` with a different structure; returning None there
    leaves the path check as the sole filter rather than guessing at a
    layout.
    """
    opt = getattr(socket, "SO_PEERCRED", None)
    if opt is None:
        return None
    try:
        raw = sock.getsockopt(socket.SOL_SOCKET, opt, struct.calcsize("3i"))
        _pid, uid, _gid = struct.unpack("3i", raw)
        return uid
    except (OSError, struct.error):
        return None


def _is_trustworthy_socket(candidate: Path) -> bool:
    """True when *candidate* is a socket this user owns, with no group/other access.

    The module's trust model is that filesystem ACLs are the auth boundary.
    That holds for ``$HERMES_HOME/gateway.sock`` — the home directory is the
    user's. It does NOT hold for the temp-dir fallback: on Linux
    ``tempfile.gettempdir()`` is usually the shared, world-writable ``/tmp``,
    and the filename is ``hermes-gw-<sha256(home)[:16]>.sock`` — unsalted, so
    anyone who can guess the home path can compute it and bind there first.
    A client that connects to whatever exists would then take its answers
    from that process, and ``identify``/``status`` are exactly the answers
    the updater and the fleet-version matrix act on.

    The server already binds under ``umask(0o177)`` and chmods 0600, so a
    genuine socket always passes. This is the client half of the same
    property: refuse anything we do not own, or that anyone else can reach.

    Windows named pipes live in a per-user namespace and carry no st_uid;
    this check is POSIX-only by design.
    """
    if _IS_WINDOWS:
        return True
    try:
        info = os.stat(candidate)
    except OSError:
        return False
    if not stat.S_ISSOCK(info.st_mode):
        logger.debug("Control socket %s is not a socket; ignoring", candidate)
        return False
    euid = _current_euid()
    if euid is not None and info.st_uid != euid:
        logger.warning(
            "Ignoring control socket %s: owned by uid %d, not %d",
            candidate, info.st_uid, euid,
        )
        return False
    if stat.S_IMODE(info.st_mode) & 0o077:
        logger.warning(
            "Ignoring control socket %s: mode %o allows group/other access",
            candidate, stat.S_IMODE(info.st_mode),
        )
        return False
    return True


def resolve_client_socket_path(home: Path) -> Optional[Path]:
    """Where a client should connect for ``home``, or None when nothing exists.

    A candidate that fails :func:`_is_trustworthy_socket` is treated as
    absent, which drops the caller onto the existing state-file/scan
    fallback — the same path taken when no gateway is running.
    """
    direct = _default_socket_path(home)
    if _is_trustworthy_socket(direct):
        return direct
    pointer = _pointer_path(home)
    try:
        if pointer.is_file():
            target = pointer.read_text(encoding="utf-8").strip()
            if target:
                candidate = Path(target)
                if _is_trustworthy_socket(candidate):
                    return candidate
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Default payload builders (import-light; overridable at wiring time)
# ---------------------------------------------------------------------------

def _detect_supervisor() -> str:
    """Best-effort supervisor kind for THIS process, from its own environment.

    Unlike the outside-in `_detect_supervisor_for_pid` scan, this answers from
    the process's own launch context — which is exactly the provenance the
    #92091 design wants declared rather than inferred.
    """
    env = os.environ
    if env.get("INVOCATION_ID"):
        return "systemd"
    if sys.platform == "darwin" and (
        env.get("XPC_SERVICE_NAME", "").startswith("ai.hermes")
        or env.get("LAUNCHD_SOCKET")
    ):
        return "launchd"
    if env.get("HERMES_DESKTOP_MANAGED"):
        return "desktop"
    if "--external-supervisor" in sys.argv:
        return "external"
    return "manual"


def build_identify_payload() -> dict[str, Any]:
    """Default ``identify`` answer, built from gateway.status primitives."""
    from gateway.status import (
        _build_pid_record,
        _get_code_identity_fields,
        _profile_label_for_home,
        read_runtime_status,
    )

    record = _build_pid_record()
    payload: dict[str, Any] = {
        "protocol": CONTROL_PROTOCOL_VERSION,
        "kind": record.get("kind"),
        "pid": record.get("pid"),
        "start_time": record.get("start_time"),
        "hermes_home": record.get("hermes_home"),
        "profile": _profile_label_for_home(record.get("hermes_home") or ""),
        "supervisor": _detect_supervisor(),
    }
    payload.update(_get_code_identity_fields())
    # served_profiles (multiplex mode) is stamped into the runtime status by
    # the runner; surface it when present so fleet consumers see coverage.
    try:
        runtime = read_runtime_status() or {}
        served = runtime.get("served_profiles")
        if isinstance(served, list) and served:
            payload["served_profiles"] = served
    except Exception:
        pass
    return payload


def build_status_payload() -> dict[str, Any]:
    """Default ``status`` answer — current runtime status, answered live."""
    from gateway.status import read_runtime_status

    payload = read_runtime_status() or {}
    payload = dict(payload)
    payload["protocol"] = CONTROL_PROTOCOL_VERSION
    payload["answered_at"] = time.time()
    payload["answering_pid"] = os.getpid()
    return payload


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class GatewayControlServer:
    """Gateway-owned control socket server (identify/status, v1).

    Lifecycle is owned by the gateway process: ``start()`` after the PID-file
    claim (the point where this process becomes the authoritative gateway for
    its HERMES_HOME), ``stop()`` on shutdown. All failures are non-fatal —
    the gateway never refuses to serve messaging because its control socket
    couldn't bind; consumers simply fall back to the scan layer.
    """

    def __init__(
        self,
        home: Optional[Path] = None,
        *,
        verb_handlers: Optional[dict[str, Callable[[], dict[str, Any]]]] = None,
    ) -> None:
        if home is None:
            from gateway.status import _get_process_hermes_home

            home = _get_process_hermes_home()
        self._home = Path(home)
        self._server: Optional[asyncio.AbstractServer] = None
        self._pipe_server: Any = None  # Windows proactor pipe server
        self._bind_path: Optional[Path] = None
        self._pointer_file: Optional[Path] = None
        self._handlers: dict[str, Callable[[], dict[str, Any]]] = {
            "identify": build_identify_payload,
            "status": build_status_payload,
        }
        if verb_handlers:
            self._handlers.update(verb_handlers)

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> bool:
        """Bind and start serving. Returns True on success, False otherwise."""
        try:
            if _IS_WINDOWS:
                return await self._start_windows()
            return await self._start_posix()
        except Exception as exc:
            logger.warning("Gateway control socket failed to start (non-fatal): %s", exc)
            return False

    async def _start_posix(self) -> bool:
        bind_path, pointer_file = resolve_server_socket_path(self._home)
        # Clear a stale socket left by a crashed predecessor. We only get
        # here after winning the PID-file O_EXCL race, so any existing file
        # is either stale or a plain collision — never a live sibling for
        # this HERMES_HOME.
        with contextlib.suppress(OSError):
            if bind_path.exists():
                bind_path.unlink()
        # Bind under a restrictive umask so the socket is never
        # world-connectable, even for the instant before an explicit chmod
        # could run. Restore the process umask immediately after.
        old_umask = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_connection, path=str(bind_path)
            )
        finally:
            os.umask(old_umask)
        with contextlib.suppress(OSError):
            os.chmod(bind_path, 0o600)
        self._bind_path = bind_path
        if pointer_file is not None:
            pointer_file.write_text(str(bind_path), encoding="utf-8")
            self._pointer_file = pointer_file
        logger.info("Gateway control socket listening at %s", bind_path)
        return True

    async def _start_windows(self) -> bool:
        loop = asyncio.get_running_loop()
        start_serving_pipe = getattr(loop, "start_serving_pipe", None)
        if start_serving_pipe is None:
            logger.debug(
                "Event loop %s has no start_serving_pipe — control socket "
                "disabled (selector loop on Windows).",
                type(loop).__name__,
            )
            return False
        pipe_name = windows_pipe_name(self._home)

        def _factory():
            return _PipeControlProtocol(self)

        servers = await start_serving_pipe(_factory, pipe_name)
        self._pipe_server = servers[0] if servers else None
        logger.info("Gateway control pipe listening at %s", pipe_name)
        return self._pipe_server is not None

    async def stop(self) -> None:
        """Stop serving and remove the socket/pointer files."""
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        if self._pipe_server is not None:
            with contextlib.suppress(Exception):
                self._pipe_server.close()
            self._pipe_server = None
        self.cleanup_files()

    def cleanup_files(self) -> None:
        """Best-effort removal of socket + pointer files (atexit-safe)."""
        if self._bind_path is not None:
            with contextlib.suppress(OSError):
                self._bind_path.unlink(missing_ok=True)
        if self._pointer_file is not None:
            with contextlib.suppress(OSError):
                self._pointer_file.unlink(missing_ok=True)

    # -- request handling ----------------------------------------------------

    def handle_request_line(self, raw: bytes) -> bytes:
        """Process one JSON request line, return one JSON response line.

        Shared by the POSIX stream handler and the Windows pipe protocol.
        Never raises.
        """
        request_id: Any = None
        try:
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            request_id = request.get("id")
            verb = request.get("verb")
            handler = self._handlers.get(verb) if isinstance(verb, str) else None
            if handler is None:
                response: dict[str, Any] = {
                    "ok": False,
                    "error": f"unknown verb: {verb!r}",
                    "protocol": CONTROL_PROTOCOL_VERSION,
                    "supported_verbs": sorted(self._handlers),
                }
            else:
                response = {
                    "ok": True,
                    "protocol": CONTROL_PROTOCOL_VERSION,
                    "result": handler(),
                }
        except Exception as exc:
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "protocol": CONTROL_PROTOCOL_VERSION,
            }
        if request_id is not None:
            response["id"] = request_id
        try:
            encoded = json.dumps(response, default=str).encode("utf-8")
        except Exception:
            encoded = b'{"ok": false, "error": "response serialization failed"}'
        if len(encoded) > _MAX_RESPONSE_BYTES:
            encoded = b'{"ok": false, "error": "response too large"}'
        return encoded + b"\n"

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            raw = await asyncio.wait_for(
                reader.readline(), timeout=_DEFAULT_CLIENT_TIMEOUT
            )
            if not raw or len(raw) > _MAX_REQUEST_BYTES:
                return
            # Handlers read state files from disk; keep that off the
            # gateway's event loop (the same loop drives every platform
            # adapter), so a fast-polling consumer can't stall heartbeats.
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, self.handle_request_line, raw.rstrip(b"\n")
            )
            writer.write(response)
            await writer.drain()
        except (asyncio.TimeoutError, ConnectionError, OSError):
            pass
        except Exception:
            logger.debug("Control socket connection handler error", exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                writer.close()


class _PipeControlProtocol(asyncio.Protocol):
    """One-shot request/response protocol for the Windows named pipe."""

    def __init__(self, server: GatewayControlServer) -> None:
        self._server = server
        self._transport: Any = None
        self._buffer = bytearray()

    def connection_made(self, transport) -> None:  # pragma: no cover - windows
        self._transport = transport

    def data_received(self, data: bytes) -> None:  # pragma: no cover - windows
        self._buffer.extend(data)
        if len(self._buffer) > _MAX_REQUEST_BYTES:
            self._transport.close()
            return
        if b"\n" in self._buffer:
            line, _, _ = bytes(self._buffer).partition(b"\n")
            try:
                self._transport.write(self._server.handle_request_line(line))
            finally:
                self._transport.close()


# ---------------------------------------------------------------------------
# Client (synchronous — used by CLI/updater consumers)
# ---------------------------------------------------------------------------

def query_gateway_control(
    home: Path,
    verb: str,
    *,
    timeout: float = _DEFAULT_CLIENT_TIMEOUT,
) -> Optional[dict[str, Any]]:
    """Ask the gateway serving ``home`` a control verb; None when unanswered.

    Returns the verb's ``result`` payload on success. Any failure — no
    socket, stale socket nobody accepts on, timeout, malformed answer,
    ``ok: false`` — returns None so callers fall back to the scan layer.
    Never raises.
    """
    request = (
        json.dumps({"verb": verb, "id": 1, "protocol": CONTROL_PROTOCOL_VERSION})
        .encode("utf-8")
        + b"\n"
    )
    try:
        if _IS_WINDOWS:
            raw = _query_windows_pipe(Path(home), request, timeout)
        else:
            raw = _query_unix_socket(Path(home), request, timeout)
    except Exception:
        return None
    if not raw:
        return None
    try:
        response = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(response, dict) or response.get("ok") is not True:
        return None
    result = response.get("result")
    return result if isinstance(result, dict) else None


def _query_unix_socket(home: Path, request: bytes, timeout: float) -> Optional[bytes]:
    path = resolve_client_socket_path(home)
    if path is None:
        return None
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect(str(path))
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return None
        peer, euid = _peer_euid(sock), _current_euid()
        if peer is not None and euid is not None and peer != euid:
            logger.warning(
                "Refusing control socket %s: peer runs as uid %d, not %d",
                path, peer, euid,
            )
            return None
        sock.sendall(request)
        chunks: list[bytes] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                return None
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
            if sum(len(c) for c in chunks) > _MAX_RESPONSE_BYTES:
                return None
        data = b"".join(chunks)
        line, _, _ = data.partition(b"\n")
        return line or None


def _query_windows_pipe(
    home: Path, request: bytes, timeout: float
) -> Optional[bytes]:  # pragma: no cover - exercised on the wine2e lane
    pipe_name = windows_pipe_name(home)
    deadline = time.monotonic() + timeout
    handle = None
    while handle is None:
        try:
            handle = open(pipe_name, "r+b", buffering=0)
        except FileNotFoundError:
            return None
        except OSError:
            # Pipe busy (another client mid-handshake) — brief retry window.
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)
    try:
        handle.write(request)
        chunks: list[bytes] = []
        while time.monotonic() < deadline:
            chunk = handle.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
            if sum(len(c) for c in chunks) > _MAX_RESPONSE_BYTES:
                return None
        data = b"".join(chunks)
        line, _, _ = data.partition(b"\n")
        return line or None
    finally:
        with contextlib.suppress(Exception):
            handle.close()


def identify_gateway(home: Path, *, timeout: float = _DEFAULT_CLIENT_TIMEOUT) -> Optional[dict[str, Any]]:
    """Convenience wrapper: ``identify`` the gateway serving ``home``."""
    return query_gateway_control(home, "identify", timeout=timeout)


def pause_gateway_for_update(
    home: Path, *, timeout: float = _DEFAULT_CLIENT_TIMEOUT
) -> Optional[dict[str, Any]]:
    """Ask the gateway serving ``home`` to drain and exit for an update.

    Step 2 of the socket migration (#92091). Returns the gateway's ACK —
    ``{"pausing": bool, "already_stopping": bool, "pid": int,
    "drain_timeout": float}`` — or None when no gateway answers (older
    gateway without the verb, no socket, dead socket). None means the
    caller falls back to the legacy pause path (signals / tree-kill),
    exactly as before this verb existed.
    """
    return query_gateway_control(home, "pause-for-update", timeout=timeout)
