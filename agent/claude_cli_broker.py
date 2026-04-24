"""Shared Claude CLI persistent worker broker for the Claude CLI provider.

The foreground Hermes process is intentionally thin: it sends turn payloads over
a Unix socket, while this broker owns the long-lived ``claude -p`` workers.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.claude_cli_client import (
    ClaudeCLIClient,
    _debug_log,
    _stream_chunk_to_broker_payload,
)


@dataclass
class _ClientEntry:
    client: ClaudeCLIClient
    last_used: float


def _idle_seconds() -> float:
    raw = os.getenv("HERMES_CLAUDE_CLI_BROKER_IDLE_SECONDS", "").strip()
    try:
        return max(30.0, float(raw)) if raw else 1800.0
    except Exception:
        return 1800.0


def _server_idle_seconds() -> float:
    raw = os.getenv("HERMES_CLAUDE_CLI_BROKER_SERVER_IDLE_SECONDS", "").strip()
    try:
        return max(30.0, float(raw)) if raw else _idle_seconds()
    except Exception:
        return _idle_seconds()


def _socket_is_alive(path: Path) -> bool:
    if not path.exists():
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(0.1)
    try:
        sock.connect(str(path))
        return True
    except OSError:
        return False
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _acquire_server_lock(path: Path) -> int | None:
    lock_path = Path(str(path) + ".lock")
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        return None
    return lock_fd


def _config_key(config: dict[str, Any]) -> tuple[Any, ...]:
    args = config.get("args")
    if not isinstance(args, list):
        args = []
    return (
        str(config.get("command") or ""),
        tuple(str(part) for part in args),
        str(config.get("cwd") or ""),
        bool(config.get("strip_runtime")),
    )


def _session_key(request: dict[str, Any]) -> tuple[Any, ...]:
    config = request.get("config")
    if not isinstance(config, dict):
        config = {}
    state = request.get("transport_state")
    if not isinstance(state, dict):
        state = {}
    session_id = str(state.get("claude_session_id") or "").strip()
    if not session_id:
        # Never share no-state first-turn workers across unrelated Hermes sessions.
        session_id = f"request:{request.get('request_id') or id(request)}"
    return (*_config_key(config), session_id)


def _request_with_state(request: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    updated = dict(request)
    updated["transport_state"] = state
    return updated


class ClaudeCLIBroker:
    def __init__(self) -> None:
        self._clients: dict[tuple[Any, ...], _ClientEntry] = {}
        self._last_activity = time.monotonic()
        self._lock = threading.RLock()

    def touch(self) -> None:
        with self._lock:
            self._last_activity = time.monotonic()

    def should_exit_idle(self) -> bool:
        self._cleanup_idle()
        with self._lock:
            return (
                not self._clients
                and time.monotonic() - self._last_activity > _server_idle_seconds()
            )

    def close(self) -> None:
        with self._lock:
            entries = list(self._clients.values())
            self._clients.clear()
        for entry in entries:
            try:
                entry.client.close()
            except Exception:
                pass

    def _cleanup_idle(self) -> None:
        cutoff = time.monotonic() - _idle_seconds()
        stale: list[_ClientEntry] = []
        with self._lock:
            for key, entry in list(self._clients.items()):
                if entry.last_used < cutoff:
                    stale.append(entry)
                    self._clients.pop(key, None)
        for entry in stale:
            try:
                entry.client.close()
            except Exception:
                pass

    def _get_client(self, request: dict[str, Any]) -> tuple[tuple[Any, ...], _ClientEntry]:
        self._cleanup_idle()
        config = request.get("config")
        if not isinstance(config, dict):
            config = {}
        key = _session_key(request)
        with self._lock:
            self._last_activity = time.monotonic()
            entry = self._clients.get(key)
            if entry is None:
                timeout = config.get("timeout")
                entry = _ClientEntry(
                    client=ClaudeCLIClient(
                        command=str(config.get("command") or "") or None,
                        args=config.get("args") if isinstance(config.get("args"), list) else [],
                        claude_cwd=str(config.get("cwd") or "") or None,
                        strip_runtime=bool(config.get("strip_runtime")),
                        timeout=timeout if isinstance(timeout, (int, float)) else None,
                    ),
                    last_used=time.monotonic(),
                )
                self._clients[key] = entry
            else:
                entry.last_used = time.monotonic()

        state = request.get("transport_state")
        if isinstance(state, dict):
            entry.client.import_transport_state(state)
        return key, entry

    def _maybe_rekey(
        self,
        old_key: tuple[Any, ...],
        request: dict[str, Any],
        entry: _ClientEntry,
        state: dict[str, Any] | None,
    ) -> None:
        if not state:
            return
        new_key = _session_key(_request_with_state(request, state))
        if new_key == old_key:
            return
        with self._lock:
            existing = self._clients.get(new_key)
            if existing is None or existing is entry:
                self._clients[new_key] = entry
                self._clients.pop(old_key, None)
            else:
                self._clients.pop(old_key, None)
                try:
                    entry.client.close()
                except Exception:
                    pass

    def handle_stream(
        self,
        request: dict[str, Any],
        send: Callable[[dict[str, Any]], None],
    ) -> None:
        stream = request.get("stream")
        if not isinstance(stream, dict):
            raise RuntimeError("Broker stream request missing stream payload")

        key, entry = self._get_client(request)
        client = entry.client
        last_state: dict[str, Any] | None = None

        def send_state_if_changed() -> dict[str, Any] | None:
            nonlocal last_state
            state = client.export_transport_state()
            if state and state != last_state:
                send({"type": "state", "state": state})
                last_state = dict(state)
            return state

        _debug_log(
            "broker:stream_start "
            f"session_id={client.export_transport_state() or ''} "
            f"structured={bool(stream.get('structured_input'))}"
        )
        stream_iter = client._stream_completion_persistent(
            model=str(stream.get("model") or "claude-cli"),
            prompt_text=str(stream.get("prompt_text") or ""),
            system_prompt=str(stream.get("system_prompt") or ""),
            structured_input=(
                stream.get("structured_input")
                if isinstance(stream.get("structured_input"), str)
                else None
            ),
            structured_system_prompt=str(stream.get("structured_system_prompt") or ""),
            effort=stream.get("effort") if isinstance(stream.get("effort"), str) else None,
            timeout_seconds=float(stream.get("timeout_seconds") or 900.0),
        )
        try:
            for chunk in stream_iter:
                send_state_if_changed()
                send({"type": "chunk", "chunk": _stream_chunk_to_broker_payload(chunk)})
                now = time.monotonic()
                entry.last_used = now
                with self._lock:
                    self._last_activity = now
        except Exception:
            close = getattr(stream_iter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            try:
                client._close_persistent_worker(reason="broker_stream_error")
            except Exception:
                pass
            raise

        state = send_state_if_changed()
        self._maybe_rekey(key, request, entry, state)
        send({"type": "done", "state": state})


def _write_event(writer: Any, event: dict[str, Any]) -> None:
    writer.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
    writer.flush()


def _handle_connection(broker: ClaudeCLIBroker, conn: socket.socket) -> None:
    with conn:
        reader = conn.makefile("r", encoding="utf-8")
        writer = conn.makefile("w", encoding="utf-8")
        try:
            line = reader.readline()
            if not line:
                return
            request = json.loads(line)
            if not isinstance(request, dict):
                raise RuntimeError("Broker request must be a JSON object")
            action = str(request.get("action") or "").strip().lower()
            if action == "ping":
                _write_event(writer, {"type": "pong"})
            elif action == "stream":
                broker.handle_stream(request, lambda event: _write_event(writer, event))
            else:
                raise RuntimeError(f"Unknown broker action: {action or '<empty>'}")
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            _debug_log(f"broker:error error={type(exc).__name__}:{exc}")
            try:
                _write_event(writer, {"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        finally:
            try:
                reader.close()
            except Exception:
                pass
            try:
                writer.close()
            except Exception:
                pass


def serve(socket_path: str) -> None:
    path = Path(socket_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = _acquire_server_lock(path)
    if lock_fd is None:
        _debug_log(f"broker:lock_busy socket={path}")
        return

    server: socket.socket | None = None
    owns_socket = False
    broker = ClaudeCLIBroker()
    atexit.register(broker.close)

    def _cleanup_socket() -> None:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    def _shutdown(_signum: int, _frame: Any) -> None:
        broker.close()
        try:
            if server is not None:
                server.close()
        finally:
            if owns_socket:
                _cleanup_socket()
            try:
                os.close(lock_fd)
            except Exception:
                pass
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        if _socket_is_alive(path):
            _debug_log(f"broker:socket_already_alive socket={path}")
            return
        _cleanup_socket()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o177)
        try:
            server.bind(str(path))
        finally:
            os.umask(old_umask)
        owns_socket = True
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        server.listen(64)
        server.settimeout(5.0)
        _debug_log(f"broker:ready socket={path}")

        while True:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                if broker.should_exit_idle():
                    _debug_log(f"broker:idle_exit socket={path}")
                    break
                continue
            except OSError:
                break

            broker.touch()
            thread = threading.Thread(target=_handle_connection, args=(broker, conn), daemon=True)
            thread.start()
    finally:
        broker.close()
        if server is not None:
            try:
                server.close()
            except Exception:
                pass
        if owns_socket:
            _cleanup_socket()
        try:
            os.close(lock_fd)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m agent.claude_cli_broker <socket-path>", file=sys.stderr)
        return 2
    serve(args[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
