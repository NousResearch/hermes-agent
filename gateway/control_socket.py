"""Gateway control socket — the gateway-owned local coordination surface: a local-only socket answering
versioned JSON verbs (``identify``, ``status``). A connectable socket with a well-formed ``identify``
answer IS liveness — no PID-reuse heuristics. Never a TCP port: filesystem/pipe ACLs are the auth
boundary. POSIX: ``$HERMES_HOME/gateway.sock`` (or a temp-dir socket + ``gateway.sock.path`` pointer
file when the home path exceeds ``sun_path``); Windows: named pipe ``\\\\.\\pipe\\hermes-gateway-<hash>``.
Wire contract: ONE request per connection — one JSON line in, one out, then the server closes.
Consumers PREFER the socket and fall back to the state-file/scan layer when it doesn't answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import socket
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
_MAX_UNIX_PATH = 100  # sun_path limit is 104 on macOS/BSD, 108 on Linux; margin for the NUL
# Single-line JSON in/out; bounded so a misbehaving peer can't balloon memory.
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_RESPONSE_BYTES = 512 * 1024
_DEFAULT_CLIENT_TIMEOUT = 2.0


def _home_hash(home: Path) -> str:
    return hashlib.sha256(os.path.normcase(str(Path(home).expanduser().resolve(strict=False))).encode("utf-8")).hexdigest()[:16]


def windows_pipe_name(home: Path) -> str:
    """Per-HERMES_HOME named pipe path (Windows transport)."""
    return rf"\\.\pipe\hermes-gateway-{_home_hash(home)}"


def _fits_sun_path(path: Path) -> bool:
    return len(str(path).encode("utf-8")) <= _MAX_UNIX_PATH


def _fallback_socket_path(home: Path) -> Path:
    """Short temp-dir path for homes whose direct socket path exceeds sun_path: ``tempfile.gettempdir()``
    then ``/tmp`` (POSIX); if nothing fits the tempdir candidate is returned anyway — bind fails
    non-fatally and consumers use the scan layer."""
    name = f"hermes-gw-{_home_hash(home)}.sock"
    candidates = [Path(tempfile.gettempdir()) / name] + ([] if _IS_WINDOWS else [Path("/tmp") / name])
    return next((c for c in candidates if _fits_sun_path(c)), candidates[0])


def resolve_server_socket_path(home: Path) -> tuple[Path, Optional[Path]]:
    """Return ``(bind_path, pointer_file)``; pointer_file is set only for the temp-dir fallback."""
    direct = Path(home) / _SOCKET_FILENAME
    return (direct, None) if _fits_sun_path(direct) else (_fallback_socket_path(home), Path(home) / _POINTER_FILENAME)


def resolve_client_socket_path(home: Path) -> Optional[Path]:
    """Where a client should connect for ``home``, or None when nothing exists."""
    direct = Path(home) / _SOCKET_FILENAME
    if direct.exists():
        return direct
    with contextlib.suppress(OSError):
        pointer = Path(home) / _POINTER_FILENAME
        target = pointer.read_text(encoding="utf-8").strip() if pointer.is_file() else ""
        if target and Path(target).exists():
            return Path(target)
    return None


def _detect_supervisor() -> str:
    """Supervisor kind for THIS process from its own launch env (not inferred outside-in).

    Unlike the outside-in `_detect_supervisor_for_pid` scan, this answers from the process's own launch
    context — which is exactly the provenance the 92091 design wants declared rather than inferred. See
    #92091.
    """
    env = os.environ
    if env.get("INVOCATION_ID"):
        return "systemd"
    if sys.platform == "darwin" and (env.get("XPC_SERVICE_NAME", "").startswith("ai.hermes")
                                     or env.get("LAUNCHD_SOCKET")):
        return "launchd"
    if env.get("HERMES_DESKTOP_MANAGED"):
        return "desktop"
    return "external" if "--external-supervisor" in sys.argv else "manual"


def build_identify_payload() -> dict[str, Any]:
    """Default ``identify`` answer, built from gateway.status primitives."""
    from gateway.status import _build_pid_record, _get_code_identity_fields, _profile_label_for_home, read_runtime_status
    record = _build_pid_record()
    payload: dict[str, Any] = {
        "protocol": CONTROL_PROTOCOL_VERSION,
        **{k: record.get(k) for k in ("kind", "pid", "start_time", "hermes_home")},
        "profile": _profile_label_for_home(record.get("hermes_home") or ""),
        "supervisor": _detect_supervisor(), **_get_code_identity_fields()}
    with contextlib.suppress(Exception):
        # served_profiles (multiplex mode) is stamped into runtime status by the runner.
        served = (read_runtime_status() or {}).get("served_profiles")
        if isinstance(served, list) and served:
            payload["served_profiles"] = served
    return payload


def build_status_payload() -> dict[str, Any]:
    """Default ``status`` answer — current runtime status, answered live."""
    from gateway.status import read_runtime_status
    return {**(read_runtime_status() or {}), "protocol": CONTROL_PROTOCOL_VERSION,
            "answered_at": time.time(), "answering_pid": os.getpid()}


class GatewayControlServer:
    """Gateway-owned control socket server (identify/status, v1): ``start()`` after the PID-file claim,
    ``stop()`` on shutdown. All failures are non-fatal — the gateway never refuses to serve messaging
    because its control socket couldn't bind; consumers fall back to the scan layer."""

    def __init__(self, home: Optional[Path] = None, *,
                 verb_handlers: Optional[dict[str, Callable[[], dict[str, Any]]]] = None) -> None:
        if home is None:
            from gateway.status import _get_process_hermes_home
            home = _get_process_hermes_home()
        self._home = Path(home)
        self._server: Optional[asyncio.AbstractServer] = None
        self._pipe_server: Any = None  # Windows proactor pipe server
        self._bind_path: Optional[Path] = None
        self._pointer_file: Optional[Path] = None
        self._handlers: dict[str, Callable[[], dict[str, Any]]] = {
            "identify": build_identify_payload, "status": build_status_payload, **(verb_handlers or {})}

    async def start(self) -> bool:
        """Bind and start serving. Returns True on success, False otherwise."""
        try:
            return await (self._start_windows() if _IS_WINDOWS else self._start_posix())
        except Exception as exc:
            logger.warning("Gateway control socket failed to start (non-fatal): %s", exc)
            return False

    async def _start_posix(self) -> bool:
        bind_path, pointer_file = resolve_server_socket_path(self._home)
        # We only get here after winning the PID-file O_EXCL race, so any existing
        # file is stale or a collision — never a live sibling.
        with contextlib.suppress(OSError):
            if bind_path.exists():
                bind_path.unlink()
        # Restrictive umask so the socket is never world-connectable, even for the instant before chmod.
        old_umask = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(self._handle_connection, path=str(bind_path))
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
            logger.debug("Event loop %s has no start_serving_pipe — control socket "
                         "disabled (selector loop on Windows).", type(loop).__name__)
            return False
        pipe_name = windows_pipe_name(self._home)
        servers = await start_serving_pipe(lambda: _PipeControlProtocol(self), pipe_name)
        self._pipe_server = servers[0] if servers else None
        logger.info("Gateway control pipe listening at %s", pipe_name)
        return self._pipe_server is not None

    async def stop(self) -> None:
        """Stop serving and remove the socket/pointer files."""
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
        if self._pipe_server is not None:
            with contextlib.suppress(Exception):
                self._pipe_server.close()
        self._server = self._pipe_server = None
        self.cleanup_files()

    def cleanup_files(self) -> None:
        """Best-effort removal of socket + pointer files (atexit-safe)."""
        for path in filter(None, (self._bind_path, self._pointer_file)):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)

    def handle_request_line(self, raw: bytes) -> bytes:
        """One JSON request line -> one JSON response line. Never raises (shared by POSIX + pipe)."""
        request_id: Any = None
        try:
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            request_id, verb = request.get("id"), request.get("verb")
            handler = self._handlers.get(verb) if isinstance(verb, str) else None
            if handler is None:
                response: dict[str, Any] = {"ok": False, "error": f"unknown verb: {verb!r}",
                                            "protocol": CONTROL_PROTOCOL_VERSION, "supported_verbs": sorted(self._handlers)}
            else:
                response = {"ok": True, "protocol": CONTROL_PROTOCOL_VERSION, "result": handler()}
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "protocol": CONTROL_PROTOCOL_VERSION}
        if request_id is not None:
            response["id"] = request_id
        try:
            encoded = json.dumps(response, default=str).encode("utf-8")
        except Exception:
            encoded = b'{"ok": false, "error": "response serialization failed"}'
        if len(encoded) > _MAX_RESPONSE_BYTES:
            encoded = b'{"ok": false, "error": "response too large"}'
        return encoded + b"\n"

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=_DEFAULT_CLIENT_TIMEOUT)
            if not raw or len(raw) > _MAX_REQUEST_BYTES:
                return
            # Handlers read disk; keep that off the loop that drives every platform
            # adapter so a fast-polling consumer can't stall heartbeats.
            response = await asyncio.get_running_loop().run_in_executor(
                None, self.handle_request_line, raw.rstrip(b"\n"))
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
        elif b"\n" in self._buffer:
            try:
                self._transport.write(self._server.handle_request_line(bytes(self._buffer).partition(b"\n")[0]))
            finally:
                self._transport.close()


def query_gateway_control(home: Path, verb: str, *, timeout: float = _DEFAULT_CLIENT_TIMEOUT) -> Optional[dict[str, Any]]:
    """Ask the gateway serving ``home`` a control verb; returns its ``result`` payload. Any failure (no/stale
    socket, timeout, malformed answer, ``ok: false``) returns None so callers fall back to the scan layer.
    Never raises."""
    request = json.dumps({"verb": verb, "id": 1, "protocol": CONTROL_PROTOCOL_VERSION}).encode("utf-8") + b"\n"
    query = _query_windows_pipe if _IS_WINDOWS else _query_unix_socket
    try:
        raw = query(Path(home), request, timeout)
        response = json.loads(raw.decode("utf-8")) if raw else None
    except Exception:
        return None
    result = response.get("result") if isinstance(response, dict) and response.get("ok") is True else None
    return result if isinstance(result, dict) else None


def _read_response_line(read: Callable[[], bytes], deadline: float) -> Optional[bytes]:
    """Read chunks until a newline, EOF, deadline, or the size cap (-> None)."""
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        chunk = read()
        chunks.append(chunk)
        if not chunk or b"\n" in chunk:
            break
        if sum(len(c) for c in chunks) > _MAX_RESPONSE_BYTES:
            return None
    return b"".join(chunks).partition(b"\n")[0] or None


def _query_unix_socket(home: Path, request: bytes, timeout: float) -> Optional[bytes]:
    path = resolve_client_socket_path(home)
    if path is None:
        return None
    # OSError covers ConnectionRefusedError / FileNotFoundError on connect and socket.timeout on read.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock, contextlib.suppress(OSError):
        sock.settimeout(timeout)
        sock.connect(str(path))
        sock.sendall(request)
        return _read_response_line(lambda: sock.recv(65536), time.monotonic() + timeout)
    return None


def _query_windows_pipe(home: Path, request: bytes, timeout: float) -> Optional[bytes]:  # pragma: no cover - wine2e lane
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
        return _read_response_line(lambda: handle.read(65536), deadline)
    finally:
        with contextlib.suppress(Exception):
            handle.close()


def identify_gateway(home: Path, *, timeout: float = _DEFAULT_CLIENT_TIMEOUT) -> Optional[dict[str, Any]]:
    """Convenience wrapper: ``identify`` the gateway serving ``home``."""
    return query_gateway_control(home, "identify", timeout=timeout)


def pause_gateway_for_update(home: Path, *, timeout: float = _DEFAULT_CLIENT_TIMEOUT) -> Optional[dict[str, Any]]:
    """Ask the gateway serving ``home`` to drain and exit for an update. Returns the ACK ``{"pausing",
    "already_stopping", "pid", "drain_timeout"}`` or None when no gateway answers (old gateway without
    the verb, no/dead socket) — the caller then uses the legacy signal/tree-kill pause path.

    Step 2 of the socket migration (#92091).
    """
    return query_gateway_control(home, "pause-for-update", timeout=timeout)
