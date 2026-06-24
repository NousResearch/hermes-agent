"""Local IPC for sending messages to a running Hermes CLI session.

The persisted session store lets a new process resume historical context, but it
cannot safely inject a message into an already running prompt_toolkit session.
This module provides a tiny localhost-only/same-user IPC surface for that live
path.  POSIX uses Unix domain sockets; Python exposes AF_UNIX on modern Windows
too, but unsupported platforms fail closed with a clear error.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import socketserver
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_MAX_PAYLOAD_BYTES = 1024 * 1024
_DEFAULT_TIMEOUT = 5.0


def _runtime_dir() -> Path:
    return get_hermes_home() / "runtime" / "session_ipc"


def _registry_path() -> Path:
    return _runtime_dir() / "live_sessions.json"


def _lock_path() -> Path:
    return _runtime_dir() / "live_sessions.lock"


class AmbiguousSessionError(ValueError):
    """Raised when a session selector matches more than one live session."""

    def __init__(self, selector: str, matches: list[str]):
        self.selector = selector
        self.matches = matches
        preview = ", ".join(matches[:5])
        more = f" (+{len(matches) - 5} more)" if len(matches) > 5 else ""
        super().__init__(
            f"Multiple running Hermes sessions match {selector!r}: {preview}{more}. "
            "Use a longer session id prefix."
        )


def socket_path_for_session(session_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(session_id))
    # Keep Unix socket paths comfortably below the traditional 108-byte limit.
    # Pytest/macOS temp homes and deeply nested profile paths can make
    # ``$HERMES_HOME/runtime/...`` too long, so fall back to /tmp while keeping
    # the authoritative registry under HERMES_HOME.
    candidate = _runtime_dir() / f"{safe}.sock"
    if len(str(candidate).encode("utf-8")) < 100:
        return candidate
    import hashlib
    import tempfile

    home_digest = hashlib.sha256(str(get_hermes_home()).encode("utf-8")).hexdigest()[:12]
    session_digest = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:16]
    uid = getattr(os, "getuid", lambda: os.getpid())()
    base = Path(tempfile.gettempdir()) / f"hermes-ipc-{uid}-{home_digest}"
    return base / f"{session_digest}.sock"


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")
        if os.name == "nt":
            try:
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            except Exception as exc:
                self._fh.close()
                self._fh = None
                raise RuntimeError("session IPC file lock unavailable") from exc
        else:
            try:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            except Exception as exc:
                self._fh.close()
                self._fh = None
                raise RuntimeError("session IPC file lock unavailable") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh is None:
            return
        if os.name == "nt":
            try:
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        else:
            try:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        try:
            self._fh.close()
        finally:
            self._fh = None


def _read_registry(path: Path) -> list[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except Exception:
        logger.warning("Ignoring corrupt live session IPC registry at %s", path)
        return []
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _write_registry(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"entries": entries}, fh, sort_keys=True)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def _pid_alive(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        return bool(_pid_exists(pid_int))
    except Exception:
        try:
            os.kill(pid_int, 0)
            return True
        except PermissionError:
            return True
        except OSError:
            return False


def _socket_usable(path: Any) -> bool:
    if not path:
        return False
    sock_path = Path(str(path))
    if not sock_path.exists():
        return False
    if os.name != "nt":
        try:
            return stat.S_ISSOCK(sock_path.stat().st_mode)
        except OSError:
            return False
    return True


def _prune_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in entries if _pid_alive(entry.get("pid")) and _socket_usable(entry.get("socket_path"))]


def register_live_session(session_id: str, socket_path: Path, *, surface: str = "cli") -> None:
    now = time.time()
    entry = {
        "session_id": str(session_id),
        "surface": str(surface),
        "pid": os.getpid(),
        "socket_path": str(socket_path),
        "started_at": now,
        "updated_at": now,
    }
    path = _registry_path()
    with _FileLock(_lock_path()):
        entries = [e for e in _prune_entries(_read_registry(path)) if str(e.get("session_id")) != str(session_id)]
        entries.append(entry)
        _write_registry(path, entries)


def unregister_live_session(session_id: str, socket_path: Path | None = None) -> None:
    path = _registry_path()
    try:
        with _FileLock(_lock_path()):
            entries = _prune_entries(_read_registry(path))
            kept = []
            for entry in entries:
                same_id = str(entry.get("session_id")) == str(session_id)
                same_socket = socket_path is not None and str(entry.get("socket_path")) == str(socket_path)
                if same_id and (socket_path is None or same_socket):
                    continue
                kept.append(entry)
            _write_registry(path, kept)
    except Exception:
        logger.debug("Failed to unregister live session IPC entry", exc_info=True)


def list_live_sessions() -> list[dict[str, Any]]:
    path = _registry_path()
    with _FileLock(_lock_path()):
        entries = _prune_entries(_read_registry(path))
        _write_registry(path, entries)
        return entries


def resolve_live_session(selector: str | None = None, *, current: bool = False) -> Optional[dict[str, Any]]:
    entries = list_live_sessions()
    if not entries:
        return None
    if current:
        return max(entries, key=lambda e: float(e.get("updated_at") or e.get("started_at") or 0.0))
    wanted = (selector or "").strip()
    if not wanted:
        return None
    exact = [e for e in entries if str(e.get("session_id")) == wanted]
    if len(exact) == 1:
        return exact[0]
    matches = [
        e for e in entries
        if str(e.get("session_id", "")).startswith(wanted)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AmbiguousSessionError(wanted, [str(e.get("session_id")) for e in matches])
    return None


def send_message_to_session(
    *,
    session: str | None,
    message: str,
    mode: str = "auto",
    current: bool = False,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    try:
        target = resolve_live_session(session, current=current)
    except AmbiguousSessionError as exc:
        return {"sent": False, "error": str(exc)}
    if target is None:
        label = "current" if current else session
        return {"sent": False, "error": f"No running Hermes session found for {label!r}"}
    socket_path = target.get("socket_path")
    if not socket_path:
        return {"sent": False, "error": "Running session has no IPC socket registered"}
    payload = {
        "session_id": target.get("session_id"),
        "role": "user",
        "content": message,
        "source": "cli-send",
        "mode": mode,
        "created_at": time.time(),
    }
    raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    if len(raw) > _MAX_PAYLOAD_BYTES:
        return {"sent": False, "error": f"Message too large ({len(raw)} bytes; max {_MAX_PAYLOAD_BYTES})"}
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(socket_path))
            client.sendall(raw)
            client.shutdown(socket.SHUT_WR)
            chunks: list[bytes] = []
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
    except AttributeError:
        return {"sent": False, "error": "Unix-domain sockets are not supported by this Python build"}
    except OSError as exc:
        return {"sent": False, "error": f"Could not reach running session: {exc}"}
    try:
        reply = json.loads(b"".join(chunks).decode("utf-8") or "{}")
    except Exception:
        reply = {"sent": False, "error": "Running session returned an invalid IPC response"}
    if isinstance(reply, dict):
        reply.setdefault("session", target.get("session_id"))
        return reply
    return {"sent": False, "error": "Running session returned a non-object IPC response"}


@dataclass
class SessionIPCServer:
    session_id: str
    callback: Callable[[dict[str, Any]], dict[str, Any]]
    surface: str = "cli"
    socket_path: Path | None = None

    def __post_init__(self) -> None:
        self.socket_path = self.socket_path or socket_path_for_session(self.session_id)
        self._server: socketserver.UnixStreamServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not hasattr(socketserver, "UnixStreamServer") or not hasattr(socket, "AF_UNIX"):
            raise RuntimeError("Unix-domain sockets are not supported by this Python build")
        assert self.socket_path is not None
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            try:
                os.chmod(self.socket_path.parent, 0o700)
            except OSError:
                pass
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("Could not remove stale session IPC socket %s", self.socket_path, exc_info=True)

        outer = self

        class _Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:  # type: ignore[override]
                try:
                    self.request.settimeout(10.0)
                except Exception:
                    pass
                data = bytearray()
                response: dict[str, Any]
                try:
                    while len(data) <= _MAX_PAYLOAD_BYTES:
                        chunk = self.request.recv(65536)
                        if not chunk:
                            break
                        data.extend(chunk)
                except socket.timeout:
                    response = {"sent": False, "error": "Timed out while reading IPC payload"}
                except OSError as exc:
                    response = {"sent": False, "error": f"Failed to read IPC payload: {exc}"}
                else:
                    if len(data) > _MAX_PAYLOAD_BYTES:
                        response = {"sent": False, "error": f"Payload too large (max {_MAX_PAYLOAD_BYTES} bytes)"}
                    else:
                        try:
                            payload = json.loads(data.decode("utf-8"))
                            if not isinstance(payload, dict):
                                raise ValueError("payload must be a JSON object")
                            response = outer.callback(payload)
                        except Exception as exc:
                            logger.debug("Session IPC request failed", exc_info=True)
                            response = {"sent": False, "error": str(exc)}
                try:
                    self.request.sendall(json.dumps(response, ensure_ascii=False).encode("utf-8"))
                except OSError:
                    pass

        class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True
            allow_reuse_address = True

        self._server = _Server(str(self.socket_path), _Handler)
        if os.name != "nt":
            os.chmod(self.socket_path, 0o600)
        register_live_session(self.session_id, self.socket_path, surface=self.surface)
        self._thread = threading.Thread(target=self._server.serve_forever, name=f"hermes-session-ipc-{self.session_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                logger.debug("Failed to stop session IPC server", exc_info=True)
        if self.socket_path is not None:
            unregister_live_session(self.session_id, self.socket_path)
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                logger.debug("Failed to remove session IPC socket", exc_info=True)
