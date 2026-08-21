"""Versioned persistent JSONL RPC transport for the supported Pi profiles."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
import signal
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any

from . import protocol, security

if TYPE_CHECKING:
    from .launchers import LaunchOutcome, LaunchRequest, PiRpcLauncherSpec

_MAX_FRAME_BYTES = 1024 * 1024
_MAX_STDERR_BYTES = 4 * 1024 * 1024
_MAX_RETAINED_FRAMES = 1024
_FIRE_AND_FORGET_UI = frozenset({"notify", "setWidget", "set_widget", "updateWidget"})


@dataclass(frozen=True)
class _Profile:
    terminal: str
    needs_ready: bool


# Versioned lifecycle profiles for Pi-family RPC tools.
# Each profile defines the terminal event and whether a "ready" handshake
# is required before the first prompt. Only profiles with proven versioned
# contracts are registered; bare Pi remains unsupported until a successful
# model turn proves its lifecycle contract.
_PROFILES = {
    "omp": _Profile("agent_end", True),
    "feynman": _Profile("agent_settled", False),
}


def _outcome(state: str, reply: str):
    from .launchers import LaunchOutcome

    return LaunchOutcome(state, security.redact_outbound(reply[-2000:]))


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _assistant_text(value: object) -> str | None:
    if not isinstance(value, dict) or value.get("role") != "assistant":
        return None
    content = value.get("content")
    if not isinstance(content, list):
        return None
    parts = [
        item["text"]
        for item in content
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    ]
    return "".join(parts) if parts else None


class PiRpcWorker:
    """A context-owned worker with continuous readers and a serialized turn lock."""

    def __init__(self, spec: PiRpcLauncherSpec):
        self.spec = spec
        self.profile = _PROFILES[spec.protocol_profile]
        self.process: subprocess.Popen[bytes] | None = None
        self.write_lock = threading.Lock()
        self.turn_lock = threading.Lock()
        self.condition = threading.Condition()
        self.frames: deque[tuple[int, dict[str, Any]]] = deque(
            maxlen=_MAX_RETAINED_FRAMES
        )
        self.sequence = 0
        self.request_sequence = 0
        self.stderr = bytearray()
        self.failed_reason: str | None = None
        self.ready = False
        self.frame_limit = _MAX_FRAME_BYTES
        self.current_task: str | None = None
        self.cancelled = threading.Event()
        self.cancelled_tasks: set[str] = set()
        self.abort_safe = False
        self.has_prompted = False
        self.last_used = time.monotonic()

    def _environment(self) -> dict[str, str]:
        names = (
            "PATH",
            "HOME",
            "USERPROFILE",
            "SYSTEMROOT",
            "PATHEXT",
            "TEMP",
            "TMP",
            "TMPDIR",
        )
        return {key: os.environ[key] for key in names if key in os.environ}

    def _fail(self, reason: str) -> None:
        with self.condition:
            if self.failed_reason is None:
                self.failed_reason = reason
            self.condition.notify_all()

    def _start(self) -> None:
        if self.process is not None:
            return
        kwargs: dict[str, object] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "cwd": self.spec.cwd or os.getcwd(),
            "env": self._environment(),
            "shell": False,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.process = subprocess.Popen(self.spec.command, **kwargs)
        assert self.process.stdout is not None and self.process.stderr is not None
        threading.Thread(
            target=self._read_stdout, args=(self.process.stdout,), daemon=True
        ).start()
        threading.Thread(
            target=self._read_stderr, args=(self.process.stderr,), daemon=True
        ).start()
        if self.profile.needs_ready:
            cursor, deadline = 0, time.monotonic() + self.spec.startup_timeout
            while not self.ready:
                cursor, _ = self._wait(cursor, deadline)
                if self.failed_reason or self.process.poll() is not None:
                    raise RuntimeError(
                        self.failed_reason or "worker exited before ready"
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError("RPC ready handshake timed out")

    def _read_stdout(self, stream) -> None:
        while True:
            # Buffered read(size) waits for size bytes on a live pipe. readline
            # wakes for each LF record while the bounded limit detects frames
            # that never terminate.
            raw = stream.readline(self.frame_limit + 2)
            if not raw:
                return
            if not raw.endswith(b"\n") or len(raw) > self.frame_limit + 1:
                self._fail("RPC frame exceeded limit or lacked LF framing")
                return
            raw = raw[:-1]
            try:
                frame = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._fail("RPC stdout was not strict UTF-8 JSONL")
                return
            if not isinstance(frame, dict):
                self._fail("RPC frame was not a JSON object")
                return
            if frame.get("type") == "ready":
                advertised = frame.get("maxFrameBytes")
                if isinstance(advertised, int) and 0 < advertised < self.frame_limit:
                    self.frame_limit = advertised
                self.ready = True
            with self.condition:
                self.sequence += 1
                self.frames.append((self.sequence, frame))
                self.condition.notify_all()

    def _read_stderr(self, stream) -> None:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            remaining = _MAX_STDERR_BYTES - len(self.stderr)
            if remaining > 0:
                self.stderr.extend(chunk[:remaining])
            if len(chunk) > remaining:
                self._fail("RPC stderr exceeded capture limit")
                return

    def _wait(self, cursor: int, deadline: float) -> tuple[int, list[dict[str, Any]]]:
        with self.condition:
            while (
                self.sequence <= cursor
                and self.failed_reason is None
                and (self.process is None or self.process.poll() is None)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return cursor, []
                self.condition.wait(remaining)
            return self.sequence, [
                frame for number, frame in self.frames if number > cursor
            ]

    def _write(self, frame: dict[str, Any]) -> None:
        encoded = json.dumps(frame, separators=(",", ":")).encode("utf-8") + b"\n"
        with self.write_lock:
            if (
                self.process is None
                or self.process.stdin is None
                or self.process.poll() is not None
            ):
                raise RuntimeError("RPC worker is not running")
            self.process.stdin.write(encoded)
            self.process.stdin.flush()

    @staticmethod
    def _response(frame: dict[str, Any], request_id: str) -> bool | None:
        if frame.get("type") != "response" or frame.get("id") != request_id:
            return None
        return frame["success"] if isinstance(frame.get("success"), bool) else False

    def _handle_ui(self, frame: dict[str, Any]) -> None:
        if frame.get("method") in _FIRE_AND_FORGET_UI:
            return
        request_id = frame.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError("unclassified RPC UI request")
        self._write({
            "type": "extension_ui_response",
            "id": request_id,
            "cancelled": True,
        })

    def send(self, request: LaunchRequest):
        with self.turn_lock:
            if request.task_id in self.cancelled_tasks:
                self.cancelled_tasks.discard(request.task_id)
                return _outcome(protocol.STATE_FAILED, "[RPC prompt was cancelled]")
            try:
                self._start()
                self.current_task, self.abort_safe = request.task_id, False
                self.cancelled.clear()
                self.request_sequence += 1
                request_id = f"a2a-prompt-{self.request_sequence}"
                cursor = self.sequence
                deadline = time.monotonic() + self.spec.timeout
                acceptance_deadline = time.monotonic() + (
                    self.spec.startup_timeout
                    if not self.has_prompted
                    else self.spec.timeout
                )
                self._write({
                    "type": "prompt",
                    "id": request_id,
                    "message": request.prompt,
                })
                self.has_prompted = True
                accepted, answers = False, []
                while True:
                    if not accepted and time.monotonic() >= acceptance_deadline:
                        raise RuntimeError("RPC prompt acceptance timed out")
                    cursor, frames = self._wait(cursor, deadline)
                    if self.failed_reason:
                        raise RuntimeError(self.failed_reason)
                    if self.process is None or self.process.poll() is not None:
                        raise RuntimeError("RPC worker exited")
                    if not frames:
                        raise RuntimeError("RPC prompt timed out")
                    for frame in frames:
                        if frame.get("type") == "extension_ui_request":
                            self._handle_ui(frame)
                        response = self._response(frame, request_id)
                        if response is False:
                            raise RuntimeError("RPC prompt was rejected")
                        accepted = accepted or response is True
                        if frame.get("type") == "message_end":
                            answer = _assistant_text(frame.get("message"))
                            if answer is not None:
                                answers.append(answer)
                        if frame.get("type") == self.profile.terminal:
                            if self.cancelled.is_set():
                                return _outcome(
                                    protocol.STATE_FAILED, "[RPC prompt was cancelled]"
                                )
                            if not accepted:
                                raise RuntimeError(
                                    "RPC terminal arrived before acceptance"
                                )
                            reply = answers[-1].strip() if answers else ""
                            if (
                                not reply
                                or len(reply.encode("utf-8")) > _MAX_FRAME_BYTES
                            ):
                                raise RuntimeError(
                                    "RPC terminal had no assistant reply"
                                )
                            return _outcome(protocol.STATE_COMPLETED, reply)
            except (OSError, RuntimeError, ValueError) as exc:
                self.close()
                return _outcome(protocol.STATE_FAILED, f"[RPC launcher failed: {exc}]")
            finally:
                self.current_task = None
                self.last_used = time.monotonic()

    def cancel(self, task_id: str) -> bool:
        with self.condition:
            self.cancelled_tasks.add(task_id)
        if self.current_task != task_id:
            return True
        try:
            self.request_sequence += 1
            abort_id = f"a2a-abort-{self.request_sequence}"
            cursor, deadline = (
                self.sequence,
                time.monotonic() + self.spec.startup_timeout,
            )
            self.cancelled.set()
            self._write({"type": "abort", "id": abort_id})
            accepted, settled = False, False
            while not settled:
                cursor, frames = self._wait(cursor, deadline)
                if (
                    self.failed_reason
                    or self.process is None
                    or self.process.poll() is not None
                ):
                    raise RuntimeError(
                        self.failed_reason or "RPC worker exited during abort"
                    )
                if not frames:
                    raise RuntimeError("RPC abort did not settle")
                for frame in frames:
                    if frame.get("type") == "extension_ui_request":
                        self._handle_ui(frame)
                    response = self._response(frame, abort_id)
                    if response is False:
                        raise RuntimeError("RPC abort was rejected")
                    accepted = accepted or response is True
                    settled = settled or frame.get("type") == self.profile.terminal
            if not accepted:
                raise RuntimeError("RPC abort settled without acknowledgement")
            self.abort_safe = True
            return True
        except (OSError, RuntimeError):
            self.close()
            return False

    def close(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            _terminate(process)
        self._fail(self.failed_reason or "RPC worker closed")


class PiRpcLauncher:
    """Routes requests to one persistent worker per exact agent/context key."""

    def __init__(self, spec: PiRpcLauncherSpec):
        self.spec = spec
        self._workers: dict[tuple[str, str], PiRpcWorker] = {}
        self._lock = threading.Lock()

    def send(self, request: LaunchRequest):
        key = (request.agent_slug, request.context_id)
        with self._lock:
            worker = self._workers.get(key)
            if worker is None or worker.failed_reason is not None:
                worker = PiRpcWorker(self.spec)
                self._workers[key] = worker
        outcome = worker.send(request)
        if outcome.state != protocol.STATE_COMPLETED and not worker.abort_safe:
            with self._lock:
                if self._workers.get(key) is worker:
                    self._workers.pop(key, None)
            worker.close()
        return outcome

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            workers = tuple(self._workers.values())
        return any(worker.cancel(task_id) for worker in workers)

    def reap_idle(self, now: float) -> None:
        with self._lock:
            stale = [
                (key, worker)
                for key, worker in self._workers.items()
                if worker.current_task is None
                and not worker.turn_lock.locked()
                and worker.last_used + self.spec.idle_timeout <= now
            ]
            for key, _ in stale:
                self._workers.pop(key, None)
        for _, worker in stale:
            worker.close()

    def close(self) -> None:
        with self._lock:
            workers = tuple(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.close()
