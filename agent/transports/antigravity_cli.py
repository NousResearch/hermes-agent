"""Low-level ``agy`` 1.2.7 stream-json subprocess transport.

The CLI accepts newline-delimited ``user`` messages on stdin and emits NDJSON
``init`` / ``step_update`` / ``result`` records.  This module deliberately does
not make approval decisions: every event, including unknown or denied-tool
records, is preserved for the caller to handle under its own policy.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from agent.deadline import kill_process_tree
from hermes_cli._subprocess_compat import windows_hide_flags
from tools.environments.local import hermes_subprocess_env

AGY_MIN_VERSION = (1, 2, 7)
_MAX_QUEUED_EVENTS = 1024
_MAX_RETAINED_EVENTS = 10000
_DEFAULT_KNOWN_LOCATIONS = (
    "~/.local/bin/agy",
    "/usr/local/bin/agy",
    "/opt/homebrew/bin/agy",
)


class AntigravityError(RuntimeError):
    """Base error for agy CLI discovery, process, and protocol failures."""


class AntigravityNotFoundError(AntigravityError):
    """No usable agy executable could be found."""


class AntigravityStartupTimeout(AntigravityError, TimeoutError):
    """agy did not emit its initial stream event within the startup deadline."""


class AntigravityRequestTimeout(AntigravityError, TimeoutError):
    """agy did not complete a turn within the request deadline."""


class AntigravityCancelled(AntigravityError):
    """The caller cancelled a turn and the agy process was interrupted."""


class AntigravityProtocolError(AntigravityError):
    """agy emitted invalid stream-json or ended without a terminal result."""


class AntigravityTurnError(AntigravityError):
    """agy ended a turn without completing the requested work."""


class AntigravityProcessError(AntigravityError):
    """agy exited unsuccessfully; ``stderr_tail`` is bounded diagnostic context."""

    def __init__(self, message: str, *, returncode: int | None = None, stderr_tail: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = tuple(stderr_tail)


@dataclass(frozen=True)
class AntigravityCapabilities:
    available: bool
    executable: str | None
    version: tuple[int, int, int] | None
    stream_json: bool
    sandbox: bool
    resume: bool
    message: str = ""
    authenticated: bool | None = None
    models: tuple[str, ...] = ()


@dataclass(frozen=True)
class AntigravityTurnResult:
    text: str
    conversation_id: str | None
    events: list[dict[str, Any]]
    stderr_tail: list[str]
    argv: list[str]
    returncode: int
    usage: dict[str, Any] | None = None
    interrupted: bool = False
    error: str | None = None

    @property
    def final_text(self) -> str:
        return self.text


def parse_agy_version(output: str) -> tuple[int, int, int] | None:
    """Parse a semantic version from ``agy --version`` output."""
    match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", output or "")
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def parse_agy_models(output: str) -> tuple[str, ...]:
    """Parse stable model ids from ``agy models`` tabular output."""
    models: list[str] = []
    for raw_line in (output or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        model_id = line.split("\t", 1)[0].strip()
        if " " in model_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]*", model_id):
            continue
        if model_id not in models:
            models.append(model_id)
    return tuple(models)


def discover_agy(
    config_path: str | os.PathLike[str] | None = None,
    *,
    known_locations: Iterable[str | os.PathLike[str]] = _DEFAULT_KNOWN_LOCATIONS,
) -> str:
    """Resolve agy from explicit config, PATH, then known absolute locations."""
    candidates: list[str] = []
    if config_path:
        candidates.append(os.fspath(config_path))
    if found := shutil.which("agy"):
        candidates.append(found)
    candidates.extend(os.path.expanduser(os.fspath(item)) for item in known_locations)
    for candidate in dict.fromkeys(candidates):
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise AntigravityNotFoundError("agy CLI not found (checked configured path, PATH, and known locations)")


class AntigravityClient:
    """One-shot streaming client for an ``agy --print=`` process.

    A client is intentionally stateless except for executable discovery.  The caller carries
    ``conversation_id`` between turns and supplies it back through ``--conversation``.
    """

    def __init__(
        self,
        config_path: str | os.PathLike[str] | None = None,
        *,
        known_locations: Iterable[str | os.PathLike[str]] = _DEFAULT_KNOWN_LOCATIONS,
        startup_timeout: float = 10.0,
        request_timeout: float = 300.0,
        shutdown_timeout: float = 5.0,
        stderr_line_limit: int = 100,
        env: Mapping[str, str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        sandbox: bool = True,
        dangerously_skip_permissions: bool = False,
        debug_log: str | os.PathLike[str] | None = None,
    ) -> None:
        self._config_path = os.fspath(config_path) if config_path else None
        self._known_locations = tuple(os.fspath(path) for path in known_locations)
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.shutdown_timeout = shutdown_timeout
        self.stderr_line_limit = max(1, int(stderr_line_limit))
        self._env = dict(env or {})
        self._cwd = os.fspath(cwd) if cwd is not None else None
        self._sandbox = bool(sandbox)
        self._dangerously_skip_permissions = bool(dangerously_skip_permissions)
        self._debug_log = Path(debug_log) if debug_log is not None else None
        self._executable: str | None = None
        self._active_process: subprocess.Popen[Any] | None = None
        self._active_lock = threading.Lock()

    @property
    def executable(self) -> str:
        if self._executable is None:
            self._executable = discover_agy(self._config_path, known_locations=self._known_locations)
        return self._executable

    def probe(self, *, timeout: float = 5.0) -> AntigravityCapabilities:
        """Check executable/version without sending a model request."""
        try:
            executable = self.executable
        except AntigravityNotFoundError as exc:
            return AntigravityCapabilities(False, None, None, False, False, False, str(exc))
        env = hermes_subprocess_env(inherit_credentials=False)
        env.update(self._env)
        try:
            completed = subprocess.run(
                [executable, "--version"], stdin=subprocess.DEVNULL, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env,
                creationflags=windows_hide_flags(), check=False,
            )
        except subprocess.TimeoutExpired:
            return AntigravityCapabilities(False, executable, None, False, False, False, "agy --version timed out")
        except OSError as exc:
            return AntigravityCapabilities(False, executable, None, False, False, False, str(exc))
        version = parse_agy_version(completed.stdout)
        if completed.returncode != 0 or version is None:
            detail = (completed.stderr or completed.stdout).strip()
            return AntigravityCapabilities(False, executable, version, False, False, False, detail or "unable to parse agy version")
        supported = version >= AGY_MIN_VERSION
        message = "" if supported else f"agy {'.'.join(map(str, version))} is older than required {'.'.join(map(str, AGY_MIN_VERSION))}"
        authenticated: bool | None = None
        models: tuple[str, ...] = ()
        if supported:
            try:
                auth_probe = subprocess.run(
                    [executable, "models"], stdin=subprocess.DEVNULL, capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env,
                    creationflags=windows_hide_flags(), check=False,
                )
                models = parse_agy_models(auth_probe.stdout)
                authenticated = auth_probe.returncode == 0 and bool(models)
                if not authenticated and not message:
                    message = (auth_probe.stderr or "Antigravity authentication is required").strip()
            except (subprocess.TimeoutExpired, OSError) as exc:
                message = f"unable to verify Antigravity authentication: {exc}"
        return AntigravityCapabilities(
            supported, executable, version, supported, supported, supported, message,
            authenticated=authenticated, models=models,
        )

    def cancel(self) -> bool:
        """Interrupt the active turn with SIGINT, then terminate its process tree if needed."""
        with self._active_lock:
            proc = self._active_process
        if proc is None or proc.poll() is not None:
            return False
        with contextlib.suppress(Exception):
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            else:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                kill_process_tree(proc.pid, sig=signal.SIGTERM)
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(Exception):
                    kill_process_tree(proc.pid)
                with contextlib.suppress(Exception):
                    proc.wait(timeout=1.0)
        return True

    @staticmethod
    def _redact_debug(value: Any) -> Any:
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                lowered = str(key).lower()
                if any(marker in lowered for marker in (
                    "token", "secret", "password", "cookie", "authorization", "api_key", "apikey",
                )) or (lowered in {"content", "prompt", "response", "output", "stdout", "stderr"}
                       and isinstance(item, str)):
                    redacted[key] = "[REDACTED]"
                else:
                    redacted[key] = AntigravityClient._redact_debug(item)
            return redacted
        if isinstance(value, list):
            return [AntigravityClient._redact_debug(item) for item in value]
        return value

    def _write_debug(self, direction: str, payload: Any) -> None:
        if self._debug_log is None:
            return
        self._debug_log.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": time.time(), "direction": direction,
                  "payload": self._redact_debug(payload)}
        fd = os.open(self._debug_log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.chmod(self._debug_log, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                fd = -1
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        finally:
            if fd >= 0:
                os.close(fd)

    def run_turn(
        self,
        user_text: str,
        *,
        conversation_id: str | None = None,
        model: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
        startup_timeout: float | None = None,
        request_timeout: float | None = None,
    ) -> AntigravityTurnResult:
        """Run one NDJSON user turn and return all observed events unchanged."""
        effective_request_timeout = self.request_timeout if request_timeout is None else request_timeout
        argv = [
            self.executable, "--input-format", "stream-json", "--output-format", "stream-json",
            "--print-timeout", f"{max(1, int(effective_request_timeout + 5))}s",
        ]
        if self._dangerously_skip_permissions:
            argv.append("--dangerously-skip-permissions")
        elif self._sandbox:
            argv.append("--sandbox")
        selected_model = str(model or "").strip()
        if selected_model and selected_model.lower() != "auto":
            argv.extend(("--model", selected_model))
        argv.append("--print=")
        if conversation_id:
            argv.extend(("--conversation", conversation_id))
        env = hermes_subprocess_env(inherit_credentials=False)
        env.update(self._env)
        popen_kwargs: dict[str, Any] = {"start_new_session": os.name == "posix"}
        if os.name == "nt":
            popen_kwargs["creationflags"] = windows_hide_flags() | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=env, cwd=self._cwd, bufsize=0, **popen_kwargs,
            )
        except FileNotFoundError as exc:
            raise AntigravityNotFoundError(f"agy executable not found: {self.executable}") from exc
        except OSError as exc:
            raise AntigravityProcessError(f"could not start agy: {exc}") from exc
        with self._active_lock:
            self._active_process = proc
        stderr_tail: deque[str] = deque(maxlen=self.stderr_line_limit)
        stderr_lock = threading.Lock()
        event_queue: queue.Queue[dict[str, Any] | BaseException | None] = queue.Queue(maxsize=_MAX_QUEUED_EVENTS)

        def stderr_snapshot() -> list[str]:
            with stderr_lock:
                return list(stderr_tail)

        def read_stdout() -> None:
            assert proc.stdout is not None
            try:
                for raw in iter(proc.stdout.readline, b""):
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        event_queue.put(AntigravityProtocolError(f"invalid JSON on agy stdout: {line[:200]!r}"))
                        return
                    if not isinstance(value, dict):
                        event_queue.put(AntigravityProtocolError("agy stream record must be an object"))
                        return
                    event_queue.put(value)
            finally:
                event_queue.put(None)

        def read_stderr() -> None:
            assert proc.stderr is not None
            for raw in iter(proc.stderr.readline, b""):
                with stderr_lock:
                    stderr_tail.append(raw.decode("utf-8", "replace").rstrip())

        stdout_thread = threading.Thread(target=read_stdout, name="agy-stdout", daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, name="agy-stderr", daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        try:
            assert proc.stdin is not None
            # agy 1.2.7 stream input uses an ``event`` discriminator and a
            # user-role message object rather than OpenAI chat-completions input.
            payload = {"event": "user", "message": {"role": "user", "content": str(user_text)}}
            self._write_debug("stdin", payload)
            proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            proc.stdin.close()
            events: deque[dict[str, Any]] = deque(maxlen=_MAX_RETAINED_EVENTS)
            result_event: dict[str, Any] | None = None
            started = time.monotonic()
            first_deadline = started + (self.startup_timeout if startup_timeout is None else startup_timeout)
            final_deadline = started + effective_request_timeout
            callback = event_callback or on_event
            while result_event is None:
                if cancel_event is not None and cancel_event.is_set():
                    self.cancel()
                    raise AntigravityCancelled("agy turn cancelled")
                now = time.monotonic()
                if not events and now >= first_deadline:
                    self.cancel()
                    raise AntigravityStartupTimeout("agy did not emit an initial stream event before startup timeout")
                if now >= final_deadline:
                    self.cancel()
                    raise AntigravityRequestTimeout("agy turn did not finish before request timeout")
                wait_for = min(final_deadline - now, (first_deadline - now) if not events else final_deadline - now, 0.1)
                try:
                    item = event_queue.get(timeout=max(0.001, wait_for))
                except queue.Empty:
                    continue
                if isinstance(item, BaseException):
                    raise item
                if item is None:
                    break
                events.append(item)
                self._write_debug("stdout", item)
                if callback is not None:
                    callback(item)
                if (item.get("event") or item.get("type")) == "result":
                    result_event = item
            try:
                proc.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                self.cancel()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=self.shutdown_timeout)
            stderr_thread.join(timeout=1.0)
            if result_event is None:
                if proc.returncode:
                    raise AntigravityProcessError(
                        f"agy exited {proc.returncode} before terminal result", returncode=proc.returncode,
                        stderr_tail=stderr_snapshot(),
                    )
                raise AntigravityProtocolError("agy stream ended without terminal result")
            terminal = result_event.get("result")
            if not isinstance(terminal, dict):
                raise AntigravityProtocolError("agy result event is missing its result object")
            if terminal.get("status") != "SUCCESS":
                raise AntigravityProcessError(
                    f"agy reported terminal status {terminal.get('status', 'UNKNOWN')}: {terminal.get('error', '')}",
                    returncode=proc.returncode, stderr_tail=stderr_snapshot(),
                )
            text = terminal.get("response", "")
            denied_actions = terminal.get("denied_actions")
            if isinstance(denied_actions, list) and denied_actions:
                denied_names = [str(item.get("display_name") or item.get("action") or "unknown")
                                for item in denied_actions if isinstance(item, dict)]
                raise AntigravityTurnError(
                    "agy denied required action(s): " + ", ".join(denied_names or ["unknown"])
                )
            if not text and terminal.get("num_turns") == 0:
                raise AntigravityTurnError("agy returned before executing a turn (possible print timeout)")
            usage = terminal.get("usage") if isinstance(terminal.get("usage"), dict) else None
            return AntigravityTurnResult(
                text=text if isinstance(text, str) else json.dumps(text, ensure_ascii=False),
                conversation_id=(terminal.get("conversation_id") or terminal.get("conversationId")
                                 or result_event.get("conversation_id") or conversation_id),
                events=list(events), stderr_tail=stderr_snapshot(), argv=argv, returncode=proc.returncode or 0,
                usage=usage,
            )
        except KeyboardInterrupt as exc:
            self.cancel()
            raise AntigravityCancelled("agy turn cancelled") from exc
        finally:
            with self._active_lock:
                if self._active_process is proc:
                    self._active_process = None
            if proc.poll() is None:
                self.cancel()
            with contextlib.suppress(Exception):
                proc.wait(timeout=0.2)

    def close(self) -> None:
        """Cancel an active child; one-shot clients hold no other resources."""
        self.cancel()
