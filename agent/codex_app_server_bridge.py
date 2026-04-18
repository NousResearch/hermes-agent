"""Process-local bridge for Codex app-server JSON-RPC events."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import os
import subprocess
import threading
import time
from typing import Any, Callable

from hermes_cli import __version__ as HERMES_VERSION


IDLE = "idle"
RUNNING = "running"
WAITING_FOR_INPUT = "waiting_for_input"
DIFF_UPDATED = "diff_updated"
COMPLETED = "completed"
FAILED = "failed"

TERMINAL_STATUSES = {COMPLETED, FAILED}

STARTING = "starting"
READY = "ready"
STOPPED = "stopped"
ERROR = "error"

CODEX_APP_SERVER_COMMAND = ("codex", "app-server", "--listen", "stdio://")
INITIALIZE_METHOD = "initialize"
START_THREAD_METHOD = "thread/start"
START_TURN_METHOD = "turn/start"

DEFAULT_INITIALIZE_TIMEOUT_SECONDS = 5.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
STOP_TIMEOUT_SECONDS = 2.0


class CodexAppServerBridgeError(RuntimeError):
    """Base exception for bridge transport/request failures."""


class CodexAppServerTimeoutError(CodexAppServerBridgeError):
    """Raised when a JSON-RPC request does not receive a response in time."""


@dataclass
class BridgeEvent:
    """A normalized app-server notification while preserving raw details."""

    raw_method: str
    normalized_status: str
    payload: dict[str, Any]
    received_at: float
    raw_message: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BridgeTurnState:
    """Current process-local view of a Codex app-server turn."""

    bridge_status: str = IDLE
    normalized_status: str = IDLE
    thread_id: str | None = None
    turn_id: str | None = None
    last_event_at: float | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CodexAppServerBridge:
    """Small stdio JSON-RPC bridge with notification state tracking."""

    def __init__(
        self,
        *,
        recent_events_limit: int = 200,
        clock: Callable[[], float] = time.time,
        command: tuple[str, ...] | list[str] | None = None,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.state = BridgeTurnState()
        self._recent_events: deque[BridgeEvent] = deque(maxlen=recent_events_limit)
        self._pending_responses: dict[Any, dict[str, Any]] = {}
        self._request_id = 0
        self._clock = clock
        self._command = tuple(command or CODEX_APP_SERVER_COMMAND)
        self._popen_factory = popen_factory
        self._process: Any | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_reader = threading.Event()
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._last_completion_event: dict[str, Any] | None = None
        self._enqueued_completion_sessions: set[str] = set()
        self._route_metadata: dict[str, str] = {}

    def start(self, *, timeout: float = DEFAULT_INITIALIZE_TIMEOUT_SECONDS) -> dict[str, Any]:
        """Start ``codex app-server --listen stdio://`` and initialize it."""
        with self._lock:
            if self.state.bridge_status == READY and self._process_is_running_locked():
                return self.get_status()

            self.state.bridge_status = STARTING
            self.state.last_error = None
            self._stop_reader.clear()

        try:
            process = self._popen_factory(
                list(self._command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            self._set_transport_error(f"Failed to start Codex app-server: {exc}")
            raise CodexAppServerBridgeError(str(exc)) from exc

        if process.stdin is None or process.stdout is None:
            self._process = process
            self.stop()
            self._set_transport_error("Codex app-server did not expose stdio pipes")
            raise CodexAppServerBridgeError("Codex app-server did not expose stdio pipes")

        with self._lock:
            self._process = process
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                name="codex-app-server-reader",
                daemon=True,
            )
            self._reader_thread.start()

        try:
            initialize_result = self.initialize(timeout=timeout)
        except Exception as exc:
            self.stop()
            self._set_transport_error(f"Codex app-server initialize failed: {exc}")
            raise

        with self._lock:
            self.state.bridge_status = READY
            self.state.last_error = None

        status = self.get_status()
        status["initialize_result"] = initialize_result
        return status

    def initialize(self, *, timeout: float = DEFAULT_INITIALIZE_TIMEOUT_SECONDS) -> Any:
        """Send the JSON-RPC initialize request and wait for its response."""
        return self.send_request(
            INITIALIZE_METHOD,
            {"clientInfo": {"name": "hermes-agent", "version": HERMES_VERSION}},
            timeout=timeout,
        )

    def set_route_metadata(self, **metadata: str | None) -> None:
        """Remember the originating session so completion events can be routed safely."""
        filtered = {
            key: str(value)
            for key, value in metadata.items()
            if value not in (None, "")
        }
        with self._lock:
            self._route_metadata = filtered

    def stop(self) -> dict[str, Any]:
        """Stop the reader and terminate the owned subprocess."""
        with self._lock:
            process = self._process
            reader_thread = self._reader_thread
            self._stop_reader.set()

        if process is not None:
            self._close_pipe(getattr(process, "stdin", None))
            self._close_pipe(getattr(process, "stdout", None))
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=STOP_TIMEOUT_SECONDS)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=STOP_TIMEOUT_SECONDS)
            except Exception:
                pass

        if reader_thread is not None and reader_thread.is_alive():
            reader_thread.join(timeout=STOP_TIMEOUT_SECONDS)

        with self._lock:
            self._process = None
            self._reader_thread = None
            for pending in self._pending_responses.values():
                pending["error"] = {"message": "bridge stopped"}
                pending["event"].set()
            self._pending_responses.clear()
            self.state.bridge_status = STOPPED
        return self.get_status()

    def send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> Any:
        """Send a JSON-RPC request and wait for the correlated response."""
        if not method:
            raise ValueError("method is required")

        with self._lock:
            process = self._process
            if process is None or getattr(process, "stdin", None) is None:
                raise CodexAppServerBridgeError("Codex app-server bridge is not started")
            if process.poll() is not None:
                raise CodexAppServerBridgeError("Codex app-server process is not running")

        request_id = self._next_request_id()
        pending = self._register_pending_request(request_id)
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        try:
            self._write_json(message)
        except Exception as exc:
            with self._lock:
                self._pending_responses.pop(request_id, None)
            self._set_transport_error(f"Failed to write JSON-RPC request: {exc}")
            raise CodexAppServerBridgeError(str(exc)) from exc

        if not pending["event"].wait(timeout=max(0.0, float(timeout))):
            with self._lock:
                self._pending_responses.pop(request_id, None)
            raise CodexAppServerTimeoutError(
                f"Timed out waiting for Codex app-server response to {method}"
            )

        with self._lock:
            self._pending_responses.pop(request_id, None)
            error = pending.get("error")
            result = pending.get("result")

        if error:
            message = self._format_response_error(error)
            self._set_transport_error(message)
            raise CodexAppServerBridgeError(message)

        return result

    def start_turn(
        self,
        *,
        repo_path: str,
        prompt: str,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Start a repo-rooted app-server thread, then start a prompt turn."""
        repo_path = os.fspath(repo_path)
        if not repo_path:
            raise ValueError("repo_path is required")
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt is required")

        thread_result = self.send_request(
            START_THREAD_METHOD,
            {"cwd": repo_path},
            timeout=timeout,
        )
        thread_id = self._extract_id(
            thread_result,
            "thread_id",
            "threadId",
            "id",
            "thread.id",
        )
        if not thread_id:
            raise CodexAppServerBridgeError("thread/start response did not include a thread id")

        turn_result = self.send_request(
            START_TURN_METHOD,
            {
                "threadId": thread_id,
                "input": [
                    {
                        "type": "text",
                        "text": str(prompt),
                    }
                ],
            },
            timeout=timeout,
        )
        turn_id = self._extract_id(turn_result, "turn_id", "turnId", "id", "turn.id")

        with self._lock:
            if thread_id:
                self.state.thread_id = thread_id
            if turn_id:
                self.state.turn_id = turn_id
            self.state.normalized_status = RUNNING
            self.state.last_error = None
            identifier = self.state.turn_id or self.state.thread_id
            if identifier:
                self._enqueued_completion_sessions.discard(f"codex_turn_{identifier}")

        return {
            "thread_result": thread_result,
            "turn_result": turn_result,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "methods": {
                "start_thread": START_THREAD_METHOD,
                "start_turn": START_TURN_METHOD,
            },
            "status": self.get_status(),
        }

    def _next_request_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def _register_pending_request(self, request_id: Any | None = None) -> dict[str, Any]:
        """Create a tiny in-memory response slot for a JSON-RPC request."""
        if request_id is None:
            request_id = self._next_request_id()
        pending = {
            "id": request_id,
            "event": threading.Event(),
            "response": None,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._pending_responses[request_id] = pending
        return pending

    def _handle_incoming_line(self, line: str) -> dict[str, Any] | BridgeEvent | None:
        """Parse one newline-delimited JSON-RPC message.

        Malformed JSON and unsupported message shapes are converted into safe
        failed/unknown events instead of escaping exceptions into reader loops.
        """
        raw = line.strip()
        if not raw:
            return None

        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            return self._record_event(
                raw_method="jsonrpc.parse_error",
                normalized_status=FAILED,
                payload={"line": raw, "error": str(exc)},
                raw_message=None,
            )

        if not isinstance(message, dict):
            return self._record_event(
                raw_method="jsonrpc.invalid_message",
                normalized_status=FAILED,
                payload={"message": message},
                raw_message={"message": message},
            )

        if "id" in message and ("result" in message or "error" in message):
            return self._handle_response(message)

        if "id" in message and "method" in message:
            return self._handle_server_request(message)

        if "method" in message:
            return self._handle_notification(message)

        return self._record_event(
            raw_method="jsonrpc.unknown_message",
            normalized_status=self.state.normalized_status,
            payload=message,
            raw_message=message,
        )

    def _handle_response(self, message: dict[str, Any]) -> dict[str, Any]:
        """Correlate a JSON-RPC response to an in-memory pending request."""
        request_id = message.get("id")
        with self._lock:
            pending = self._pending_responses.get(request_id)
            if pending is None:
                return {
                    "matched": False,
                    "id": request_id,
                    "response": message,
                }

            pending["response"] = message
            pending["result"] = message.get("result")
            pending["error"] = message.get("error")
            pending["event"].set()
            return {
                "matched": True,
                "id": request_id,
                "result": pending["result"],
                "error": pending["error"],
            }

    def _handle_notification(self, message: dict[str, Any]) -> BridgeEvent:
        """Normalize and store a JSON-RPC notification."""
        method = str(message.get("method", ""))
        params = message.get("params", {})
        payload = params if isinstance(params, dict) else {"params": params}
        normalized_status = self._normalize_notification(method, payload)
        return self._record_event(
            raw_method=method,
            normalized_status=normalized_status,
            payload=payload,
            raw_message=message,
        )

    def _handle_server_request(self, message: dict[str, Any]) -> BridgeEvent:
        """Fail fast on server->client requests the bridge does not implement."""
        request_id = message.get("id")
        method = str(message.get("method", ""))
        params = message.get("params", {})
        payload = params if isinstance(params, dict) else {"params": params}
        self._update_ids(payload)
        error_message = (
            f"Unsupported Codex app-server request: {method}. "
            "Hermes bridge does not handle approval/input requests yet."
        )
        self._send_error_response(request_id, error_message)
        event = self._record_event(
            raw_method=method,
            normalized_status=FAILED,
            payload={
                "error": {"message": error_message},
                "server_request": payload,
                "request_id": request_id,
            },
            raw_message=message,
        )
        with self._lock:
            self.state.bridge_status = ERROR
            for pending in self._pending_responses.values():
                pending["error"] = {"message": error_message}
                pending["event"].set()
        return event

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            status = self.state.to_dict()
            status["recent_event_count"] = len(self._recent_events)
            status["process_running"] = self._process_is_running_locked()
            status["reader_alive"] = (
                self._reader_thread is not None and self._reader_thread.is_alive()
            )
            return status

    def get_recent_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._recent_events)
        if limit is not None:
            events = events[-max(0, int(limit)) :]
        return [event.to_dict() for event in events]

    def build_completion_event(self, event: BridgeEvent | None = None) -> dict[str, Any]:
        """Build a Hermes completion-queue-compatible payload."""
        with self._lock:
            normalized_status = (
                event.normalized_status if event is not None else self.state.normalized_status
            )
            identifier = self.state.turn_id or self.state.thread_id or "unknown"
            last_error = self.state.last_error
            route_metadata = dict(self._route_metadata)

        failed = normalized_status == FAILED
        exit_code = 1 if failed else 0
        output = (
            f"Codex turn failed via app-server: {last_error or 'unknown error'}"
            if failed
            else "Codex turn completed via app-server"
        )
        payload = {
            "type": "completion",
            "session_id": f"codex_turn_{identifier}",
            "command": "codex app-server turn",
            "exit_code": exit_code,
            "output": output,
        }
        payload.update(route_metadata)
        return payload

    def get_last_completion_event(self) -> dict[str, Any] | None:
        with self._lock:
            if self._last_completion_event is None:
                return None
            return dict(self._last_completion_event)

    def _record_event(
        self,
        *,
        raw_method: str,
        normalized_status: str,
        payload: dict[str, Any],
        raw_message: dict[str, Any] | None,
    ) -> BridgeEvent:
        now = float(self._clock())
        event = BridgeEvent(
            raw_method=raw_method,
            normalized_status=normalized_status,
            payload=dict(payload),
            received_at=now,
            raw_message=raw_message,
        )

        with self._lock:
            previous_status = self.state.normalized_status
            self._recent_events.append(event)
            self.state.last_event_at = now
            self._update_ids(payload)

            if normalized_status != previous_status:
                self.state.normalized_status = normalized_status
            if normalized_status == FAILED:
                self.state.last_error = self._extract_error(payload)
            elif normalized_status == COMPLETED:
                self.state.last_error = None

            completion_event = None
            if normalized_status in TERMINAL_STATUSES:
                completion_event = self.build_completion_event(event)
                if completion_event["session_id"] in self._enqueued_completion_sessions:
                    completion_event = None
                else:
                    self._enqueued_completion_sessions.add(completion_event["session_id"])
                    self._last_completion_event = dict(completion_event)

        if completion_event is not None:
            self._enqueue_completion_event(completion_event)

        return event

    @staticmethod
    def _enqueue_completion_event(event: dict[str, Any]) -> None:
        from tools.process_registry import process_registry

        process_registry.completion_queue.put(dict(event))

    def _reader_loop(self) -> None:
        while not self._stop_reader.is_set():
            with self._lock:
                process = self._process
                stdout = getattr(process, "stdout", None) if process is not None else None
            if stdout is None:
                return
            try:
                line = stdout.readline()
            except ValueError:
                return
            except Exception as exc:
                if not self._stop_reader.is_set():
                    self._set_transport_error(f"Codex app-server reader failed: {exc}")
                return
            if line == "":
                if not self._stop_reader.is_set():
                    self._set_transport_error("Codex app-server stdout closed")
                return
            self._handle_incoming_line(line)

    def _write_json(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, ensure_ascii=False) + "\n"
        with self._lock:
            process = self._process
            stdin = getattr(process, "stdin", None) if process is not None else None
        if stdin is None:
            raise CodexAppServerBridgeError("Codex app-server stdin is closed")
        with self._write_lock:
            stdin.write(line)
            stdin.flush()

    def _send_error_response(self, request_id: Any, message: str) -> None:
        if request_id is None:
            return
        try:
            self._write_json(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": message,
                    },
                }
            )
        except Exception:
            # Best-effort only — the bridge should still surface the failure locally.
            return

    def _process_is_running_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _set_transport_error(self, message: str) -> None:
        with self._lock:
            self.state.bridge_status = ERROR
            self.state.normalized_status = FAILED
            self.state.last_error = message

    @staticmethod
    def _close_pipe(pipe: Any) -> None:
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass

    @staticmethod
    def _format_response_error(error: Any) -> str:
        if isinstance(error, dict):
            message = error.get("message") or error.get("detail") or error.get("code")
            if message is not None:
                return str(message)
            return json.dumps(error, ensure_ascii=False)
        return str(error)

    @staticmethod
    def _extract_id(result: Any, *keys: str) -> str | None:
        if not isinstance(result, dict):
            return None
        for key in keys:
            if "." in key:
                parts = key.split(".")
                value: Any = result
                for part in parts:
                    if not isinstance(value, dict):
                        value = None
                        break
                    value = value.get(part)
            else:
                value = result.get(key)
            if value is not None and value != "":
                return str(value)
        return None

    def _normalize_notification(self, method: str, payload: dict[str, Any]) -> str:
        method_l = method.lower()
        status = _lower_string(payload.get("status"))
        event_type = _lower_string(payload.get("type"))
        combined = " ".join(part for part in (method_l, status, event_type) if part)

        if _contains_any(combined, ("error", "failed", "failure", "exception", "crash")):
            return FAILED
        if _contains_any(combined, ("complete", "completed", "success", "succeeded", "finished", "done")):
            return COMPLETED
        if _contains_any(
            combined,
            ("waiting", "input", "approval", "permission", "confirm", "requires_action"),
        ):
            return WAITING_FOR_INPUT
        if _contains_any(combined, ("diff", "patch", "file", "fs", "workspace")):
            return DIFF_UPDATED
        if _contains_any(
            combined,
            ("delta", "message", "agent", "progress", "started", "start", "running", "turn"),
        ):
            return RUNNING
        if any(key in payload for key in ("diff", "patch", "files", "file_changes")):
            return DIFF_UPDATED
        if "error" in payload:
            return FAILED
        return self.state.normalized_status

    def _update_ids(self, payload: dict[str, Any]) -> None:
        thread_id = _first_string(
            payload.get("thread_id"),
            payload.get("threadId"),
            _nested_string(payload, "thread", "id"),
            _nested_string(payload, "conversation", "id"),
        )
        turn_id = _first_string(
            payload.get("turn_id"),
            payload.get("turnId"),
            _nested_string(payload, "turn", "id"),
        )
        if thread_id:
            self.state.thread_id = thread_id
        if turn_id:
            self.state.turn_id = turn_id

    def _extract_error(self, payload: dict[str, Any]) -> str | None:
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("detail") or error.get("code")
            return str(message) if message is not None else json.dumps(error, ensure_ascii=False)
        if error is not None:
            return str(error)
        for key in ("message", "detail", "reason"):
            value = payload.get(key)
            if value is not None:
                return str(value)
        return None


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)


def _lower_string(value: Any) -> str:
    return value.lower() if isinstance(value, str) else ""


def _nested_string(payload: dict[str, Any], outer: str, inner: str) -> str | None:
    value = payload.get(outer)
    if isinstance(value, dict):
        nested = value.get(inner)
        if nested is not None:
            return str(nested)
    return None


def _first_string(*values: Any) -> str | None:
    for value in values:
        if value is not None and value != "":
            return str(value)
    return None
