"""Validated launcher contracts and owned process dispatch for A2A."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time

_PLACEHOLDER_RE = re.compile(
    r"\{(prompt|context_id|session_key|peer|agent_slug|session_id)\}"
)
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Mapping, Protocol

from . import protocol, security


_MAX_CAPTURE_BYTES = 4 * 1024 * 1024
_MAX_REPLY_BYTES = 1024 * 1024
_MINIMAL_ENV_NAMES = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "PATHEXT",
    "TEMP",
    "TMP",
    "TMPDIR",
)
_PLACEHOLDERS = frozenset({
    "prompt",
    "context_id",
    "session_key",
    "peer",
    "agent_slug",
    "session_id",
})


@dataclass(frozen=True)
class LaunchRequest:
    task_id: str
    agent_slug: str
    peer: str
    context_id: str
    prompt: str


@dataclass(frozen=True)
class LaunchOutcome:
    state: str
    reply: str
    session_id: str | None = None


@dataclass(frozen=True)
class ProcessOutputSpec:
    format: str
    reply_from: str
    session_id_from: str | None
    session_id_regex: re.Pattern[str] | None
    strip_session_match: bool
    reply_field: tuple[str, ...] | None
    session_id_field: tuple[str, ...] | None


@dataclass(frozen=True)
class ProcessLauncherSpec:
    transport: str
    start: tuple[str, ...]
    resume: tuple[str, ...] | None
    timeout: float
    cwd: str | None
    inherit_env: bool
    pass_env: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    output: ProcessOutputSpec
    continuity: str


@dataclass(frozen=True)
class PiRpcLauncherSpec:
    transport: str
    protocol_profile: str
    command: tuple[str, ...]
    timeout: float
    startup_timeout: float
    idle_timeout: float
    cwd: str | None


LauncherSpec = ProcessLauncherSpec | PiRpcLauncherSpec


class AgentLauncher(Protocol):
    def send(self, request: LaunchRequest) -> LaunchOutcome: ...
    def cancel(self, task_id: str) -> bool: ...
    def close(self) -> None: ...


def _argv(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"launcher.{name} must be a non-empty argv array")
    compiled = tuple(value)
    for item in compiled:
        _validate_placeholders(item, name)
    return compiled


def _validate_placeholders(value: str, name: str) -> None:
    # Only brace bodies that resemble substitution syntax are placeholders;
    # arbitrary JSON/Python literal braces (quotes, whitespace, operators)
    # remain valid argv text.
    for match in re.finditer(r"\{([^{}]*)\}", value):
        field = match.group(1)
        if field in _PLACEHOLDERS:
            continue
        if not field or any(
            field.startswith(placeholder) for placeholder in _PLACEHOLDERS
        ):
            raise ValueError(f"launcher.{name} contains unsupported placeholder")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.!:-]*", field):
            raise ValueError(f"launcher.{name} contains unsupported placeholder")
    if re.search(
        r"\{(?:prompt|context_id|session_key|peer|agent_slug|session_id)[^{}]*$", value
    ):
        raise ValueError(f"launcher.{name} contains malformed placeholder")


def _positive(value: object, name: str, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"launcher.{name} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"launcher.{name} must be a positive finite number") from exc
    if number <= 0 or not math.isfinite(number):
        raise ValueError(f"launcher.{name} must be a positive finite number")
    return number


def _cwd(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or not os.path.isdir(value):
        raise ValueError("launcher.cwd must be an existing directory")
    return value


def _env_name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "=" in value or "\x00" in value:
        raise ValueError(
            f"launcher.{name} contains an invalid environment variable name"
        )
    return value


def _field(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"launcher.output.{name} must be a non-empty dot field")
    parts = tuple(value.split("."))
    if any(not part or not part.isidentifier() for part in parts):
        raise ValueError(f"launcher.output.{name} must be a simple dot field")
    return parts


def _output(value: object) -> ProcessOutputSpec:
    raw = {} if value is None else value
    if not isinstance(raw, Mapping):
        raise ValueError("launcher.output must be a mapping")
    format_name = raw.get("format", "text")
    if format_name == "text":
        if set(raw).difference({
            "format",
            "reply_from",
            "session_id_from",
            "session_id_regex",
            "strip_session_match",
        }):
            raise ValueError("launcher.output text contains unsupported settings")
        reply_from = raw.get("reply_from", "stdout")
        session_from = raw.get("session_id_from")
        if reply_from not in {"stdout", "stderr"} or session_from not in {
            None,
            "stdout",
            "stderr",
        }:
            raise ValueError("launcher.output stream must be stdout or stderr")
        regex_value = raw.get("session_id_regex")
        if regex_value is not None and (
            not isinstance(regex_value, str) or not regex_value
        ):
            raise ValueError(
                "launcher.output.session_id_regex must be a non-empty string"
            )
        if regex_value is None and session_from is not None:
            raise ValueError(
                "launcher.output.session_id_from requires session_id_regex"
            )
        try:
            regex = re.compile(regex_value) if regex_value is not None else None
        except re.error as exc:
            raise ValueError("launcher.output.session_id_regex is invalid") from exc
        if regex is not None and regex.groups < 1:
            raise ValueError(
                "launcher.output.session_id_regex must capture the session id"
            )
        strip = raw.get("strip_session_match", False)
        if not isinstance(strip, bool):
            raise ValueError("launcher.output.strip_session_match must be boolean")
        return ProcessOutputSpec(
            "text", reply_from, session_from, regex, strip, None, None
        )
    if format_name == "json":
        if set(raw).difference({"format", "reply_field", "session_id_field"}):
            raise ValueError(
                "launcher.output JSON accepts only format and field settings"
            )
        reply = _field(raw.get("reply_field"), "reply_field")
        session = raw.get("session_id_field")
        return ProcessOutputSpec(
            "json",
            "stdout",
            None,
            None,
            False,
            reply,
            None if session is None else _field(session, "session_id_field"),
        )
    raise ValueError("launcher.output.format must be text or json")


def _contains(argv: tuple[str, ...], placeholder: str) -> bool:
    return any("{" + placeholder + "}" in item for item in argv)


def _process_spec(raw: Mapping[str, object]) -> ProcessLauncherSpec:
    unknown = set(raw).difference({
        "transport",
        "start",
        "resume",
        "timeout",
        "cwd",
        "inherit_env",
        "pass_env",
        "env",
        "output",
    })
    if unknown:
        raise ValueError("launcher.process contains unsupported settings")
    start = _argv(raw.get("start"), "start")
    resume_raw = raw.get("resume")
    resume = None if resume_raw is None else _argv(resume_raw, "resume")
    inherit_env = raw.get("inherit_env", False)
    if not isinstance(inherit_env, bool):
        raise ValueError("launcher.inherit_env must be boolean")
    pass_raw = raw.get("pass_env", [])
    if not isinstance(pass_raw, list):
        raise ValueError("launcher.pass_env must be an array")
    pass_env = tuple(_env_name(item, "pass_env") for item in pass_raw)
    if len(set(pass_env)) != len(pass_env):
        raise ValueError("launcher.pass_env must not contain duplicates")
    env_raw = raw.get("env", {})
    if not isinstance(env_raw, Mapping):
        raise ValueError("launcher.env must be a mapping")
    env: list[tuple[str, str]] = []
    for key, item in env_raw.items():
        key = _env_name(key, "env")
        if not isinstance(item, str) or "\x00" in item:
            raise ValueError("launcher.env values must be strings")
        env.append((key, item))
    output = _output(raw.get("output"))
    has_key = _contains(start, "session_key") or (
        resume is not None and _contains(resume, "session_key")
    )
    has_id = _contains(start, "session_id") or (
        resume is not None and _contains(resume, "session_id")
    )
    if resume is not None:
        if not _contains(resume, "session_id") or has_key:
            raise ValueError(
                "launcher.resume requires opaque {session_id} continuity only"
            )
        continuity = "opaque"
    elif has_id:
        raise ValueError("launcher {session_id} requires launcher.resume")
    elif has_key:
        continuity = "deterministic"
    else:
        continuity = "stateless"
    if continuity != "opaque" and output.session_id_regex is not None:
        raise ValueError("launcher session metadata requires opaque continuity")
    if continuity != "opaque" and output.session_id_field is not None:
        raise ValueError("launcher session metadata requires opaque continuity")
    return ProcessLauncherSpec(
        "process",
        start,
        resume,
        _positive(raw.get("timeout"), "timeout", 300.0),
        _cwd(raw.get("cwd")),
        inherit_env,
        pass_env,
        tuple(env),
        output,
        continuity,
    )


def parse_launcher_spec(raw: object) -> LauncherSpec:
    """Validate one route-local launcher configuration into an immutable spec."""
    if not isinstance(raw, Mapping):
        raise ValueError("launcher must be a mapping")
    transport = raw.get("transport")
    if transport == "process":
        return _process_spec(raw)
    if transport == "pi_rpc":
        # Covers the Pi family (bare Pi, OMP, Feynman, and compatible
        # derivatives); only profiles with proven versioned contracts are
        # enabled by default.
        unknown = set(raw).difference({
            "transport",
            "protocol_profile",
            "command",
            "timeout",
            "startup_timeout",
            "idle_timeout",
            "cwd",
        })
        if unknown:
            raise ValueError("launcher.pi_rpc contains unsupported settings")
        profile = raw.get("protocol_profile")
        if profile not in {"omp", "feynman"}:
            raise ValueError("launcher.protocol_profile is unsupported")
        command = _argv(raw.get("command"), "command")
        if (
            "--mode" not in command
            or command.index("--mode") + 1 >= len(command)
            or command[command.index("--mode") + 1] != "rpc"
        ):
            raise ValueError(
                "launcher.command must contain separate '--mode', 'rpc' argv values"
            )
        return PiRpcLauncherSpec(
            "pi_rpc",
            profile,
            command,
            _positive(raw.get("timeout"), "timeout", 300.0),
            _positive(raw.get("startup_timeout"), "startup_timeout", 30.0),
            _positive(raw.get("idle_timeout"), "idle_timeout", 900.0),
            _cwd(raw.get("cwd")),
        )
    raise ValueError("launcher.transport must be 'process' or 'pi_rpc'")


def _session_key(agent_slug: str, context_id: str) -> str:
    return hashlib.sha256((agent_slug + "\0" + context_id).encode("utf-8")).hexdigest()


class ProcessSessionStore:
    """Small, profile-scoped opaque session map with atomic cross-process updates."""

    def __init__(self, hermes_home: str | None):
        self._path = (
            Path(hermes_home or tempfile.gettempdir())
            / "a2a_launchers"
            / "sessions.json"
        )
        self._lock_path = self._path.with_suffix(".lock")
        self._lock = threading.Lock()

    @contextmanager
    def _locked(self):
        with self._lock:
            self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if os.name == "nt":
                    import msvcrt

                    os.write(fd, b" ") if os.fstat(fd).st_size == 0 else None
                    deadline = time.monotonic() + 5.0
                    while True:
                        try:
                            os.lseek(fd, 0, os.SEEK_SET)
                            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                            break
                        except OSError:
                            if time.monotonic() >= deadline:
                                raise
                            time.sleep(0.02)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                try:
                    if os.name == "nt":
                        import msvcrt

                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    def _read(self) -> dict[str, dict[str, str]]:
        try:
            with self._path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError, TypeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        result: dict[str, dict[str, str]] = {}
        for slug, contexts in raw.items():
            if isinstance(slug, str) and isinstance(contexts, dict):
                valid = {
                    context: session
                    for context, session in contexts.items()
                    if isinstance(context, str) and isinstance(session, str) and session
                }
                if valid:
                    result[slug] = valid
        return result

    def get(self, agent_slug: str, context_id: str) -> str | None:
        with self._locked():
            return self._read().get(agent_slug, {}).get(context_id)

    def put(self, agent_slug: str, context_id: str, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id or len(session_id) > 4096:
            return
        with self._locked():
            data = self._read()
            data.setdefault(agent_slug, {})[context_id] = session_id
            fd, temporary = tempfile.mkstemp(dir=self._path.parent, prefix="sessions.")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, separators=(",", ":"), sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, self._path)
                os.chmod(self._path, 0o600)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)


class ProcessLauncher:
    """Runs one shell-free, bounded, tree-owned process per A2A turn."""

    def __init__(self, spec: ProcessLauncherSpec, hermes_home: str | None):
        self.spec = spec
        self._sessions = ProcessSessionStore(hermes_home)
        self._active: dict[
            str, tuple[subprocess.Popen[bytes], int | None, int | None]
        ] = {}
        self._cancelled: set[str] = set()
        self._closed = False
        self._context_locks: dict[tuple[str, str], threading.Lock] = {}
        self._guard = threading.Lock()

    def _environment(self) -> dict[str, str]:
        environment = (
            os.environ.copy()
            if self.spec.inherit_env
            else {
                key: os.environ[key] for key in _MINIMAL_ENV_NAMES if key in os.environ
            }
        )
        for key in self.spec.pass_env:
            if key in os.environ:
                environment[key] = os.environ[key]
        environment.update(self.spec.env)
        return environment

    def _expand(
        self, argv: tuple[str, ...], request: LaunchRequest, session_id: str | None
    ) -> list[str]:
        values = {
            "prompt": request.prompt,
            "context_id": request.context_id,
            "session_key": _session_key(request.agent_slug, request.context_id),
            "peer": request.peer,
            "agent_slug": request.agent_slug,
            "session_id": session_id or "",
        }
        return [
            _PLACEHOLDER_RE.sub(lambda match: values[match.group(1)], item)
            for item in argv
        ]

    @staticmethod
    def _drain(stream, retained: bytearray, overflow: threading.Event) -> None:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            remaining = _MAX_CAPTURE_BYTES - len(retained)
            if remaining > 0:
                retained.extend(chunk[:remaining])
            if len(chunk) > remaining:
                overflow.set()

    @staticmethod
    def _owned_group_live(pgid: int, session_id: int) -> bool:
        """Whether a non-zombie process remains in a captured POSIX session."""
        try:
            for entry in Path("/proc").glob("[0-9]*/stat"):
                fields = entry.read_text(encoding="utf-8").rsplit(") ", 1)[-1].split()
                if (
                    len(fields) >= 4
                    and fields[0] != "Z"
                    and fields[2:4] == [str(pgid), str(session_id)]
                ):
                    return True
        except OSError:
            return False
        return False

    @staticmethod
    def _terminate(
        process: subprocess.Popen,
        pgid: int | None = None,
        session_id: int | None = None,
    ) -> None:
        """Terminate an owned tree and wait until descendants cannot execute."""
        if os.name == "nt":
            if process.poll() is None:
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=5,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
            return
        group = pgid if pgid is not None else process.pid
        if process.poll() is not None:
            if session_id is None or not ProcessLauncher._owned_group_live(
                group, session_id
            ):
                return
        try:
            os.killpg(group, signal.SIGTERM)
        except OSError:
            return
        deadline = time.monotonic() + 1.0
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        # Terminating the leader does not imply its descendants are dead.
        try:
            os.killpg(group, signal.SIGKILL)
        except OSError:
            return
        if session_id is not None:
            deadline = time.monotonic() + 5.0
            while (
                ProcessLauncher._owned_group_live(group, session_id)
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)

    @staticmethod
    def _failure(message: str) -> LaunchOutcome:
        return LaunchOutcome(
            protocol.STATE_FAILED, security.redact_outbound(message[-2000:])
        )

    def _parse(self, stdout: bytes, stderr: bytes) -> tuple[str, str | None]:
        try:
            out = stdout.decode("utf-8")
            err = stderr.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("launcher output was not valid UTF-8") from exc
        if self.spec.output.format == "json":
            try:
                value = json.loads(out)
            except ValueError as exc:
                raise ValueError("launcher output was not valid JSON") from exc

            def field(parts: tuple[str, ...]) -> object:
                current: object = value
                for part in parts:
                    if not isinstance(current, Mapping) or part not in current:
                        raise ValueError(
                            "launcher JSON output is missing a configured field"
                        )
                    current = current[part]
                return current

            reply = field(self.spec.output.reply_field or ())
            if not isinstance(reply, str):
                raise ValueError("launcher JSON reply field must be a string")
            session = None
            if self.spec.output.session_id_field is not None:
                try:
                    candidate = field(self.spec.output.session_id_field)
                except ValueError:
                    candidate = None
                if candidate is not None:
                    if not isinstance(candidate, str):
                        raise ValueError("launcher JSON session field must be a string")
                    session = candidate
            return reply.strip(), session
        streams = {"stdout": out, "stderr": err}
        reply = streams[self.spec.output.reply_from]
        session = None
        regex = self.spec.output.session_id_regex
        if regex is not None:
            source = streams[self.spec.output.session_id_from or "stderr"]
            match = regex.search(source)
            if match is not None:
                session = match.group(1)
                if (
                    self.spec.output.strip_session_match
                    and self.spec.output.session_id_from == self.spec.output.reply_from
                ):
                    reply = reply[: match.start()] + reply[match.end() :]
        return reply.strip(), session

    def send(self, request: LaunchRequest) -> LaunchOutcome:
        with self._guard:
            if self._closed or request.task_id in self._cancelled:
                return self._failure("[launcher was cancelled]")
            key = (request.agent_slug, request.context_id)
            context_lock = self._context_locks.setdefault(key, threading.Lock())
        with context_lock:
            return self._send_locked(request)

    def _send_locked(self, request: LaunchRequest) -> LaunchOutcome:
        opaque_session_id = (
            self._sessions.get(request.agent_slug, request.context_id)
            if self.spec.continuity == "opaque"
            else None
        )
        command = (
            self.spec.resume
            if opaque_session_id and self.spec.resume
            else self.spec.start
        )
        process: subprocess.Popen[bytes] | None = None
        pgid: int | None = None
        owner_session: int | None = None
        stdout, stderr = bytearray(), bytearray()
        overflow = threading.Event()
        try:
            kwargs: dict[str, object] = {
                "stdin": subprocess.DEVNULL,
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
            process = subprocess.Popen(
                self._expand(command, request, opaque_session_id), **kwargs
            )
            pgid = os.getpgid(process.pid) if os.name != "nt" else None
            owner_session = os.getsid(process.pid) if os.name != "nt" else None
            with self._guard:
                if self._closed or request.task_id in self._cancelled:
                    self._terminate(process, pgid, owner_session)
                    return self._failure("[launcher was cancelled]")
                self._active[request.task_id] = (process, pgid, owner_session)
            assert process.stdout is not None and process.stderr is not None
            readers = [
                threading.Thread(
                    target=self._drain, args=(stream, capture, overflow), daemon=True
                )
                for stream, capture in (
                    (process.stdout, stdout),
                    (process.stderr, stderr),
                )
            ]
            for reader in readers:
                reader.start()
            deadline = time.monotonic() + self.spec.timeout
            while process.poll() is None and not overflow.is_set():
                if time.monotonic() >= deadline:
                    self._terminate(process, pgid, owner_session)
                    return self._failure("[launcher did not reply in time]")
                time.sleep(0.01)
            if overflow.is_set():
                self._terminate(process, pgid, owner_session)
                return self._failure("[launcher output exceeded capture limit]")
            process.wait(timeout=1.0)
            for reader in readers:
                reader.join(timeout=2.0)
            if overflow.is_set():
                self._terminate(process, pgid, owner_session)
                return self._failure("[launcher output exceeded capture limit]")
            if process.returncode:
                diagnostic = (
                    bytes(stderr or stdout).decode("utf-8", errors="replace").strip()
                    or f"launcher exited {process.returncode}"
                )
                return self._failure(diagnostic)
            reply, new_session_id = self._parse(bytes(stdout), bytes(stderr))
            if not reply or len(reply.encode("utf-8")) > _MAX_REPLY_BYTES:
                return self._failure("[launcher returned an empty or oversized reply]")
            if self.spec.continuity == "opaque" and new_session_id:
                self._sessions.put(
                    request.agent_slug, request.context_id, new_session_id
                )
            return LaunchOutcome(
                protocol.STATE_COMPLETED,
                security.redact_outbound(reply),
                new_session_id or None,
            )
        except FileNotFoundError:
            return self._failure("[launcher executable was not found]")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return self._failure(f"Launcher dispatch failed: {exc}")
        finally:
            if process is not None:
                self._terminate(process, pgid, owner_session)
                with self._guard:
                    self._active.pop(request.task_id, None)

    def cancel(self, task_id: str) -> bool:
        with self._guard:
            self._cancelled.add(task_id)
            owned = self._active.get(task_id)
        if owned is None:
            return True
        self._terminate(*owned)
        return True

    def close(self) -> None:
        with self._guard:
            self._closed = True
            processes = tuple(self._active.values())
        for process, pgid, session_id in processes:
            self._terminate(process, pgid, session_id)


class HermesProfileLauncher:
    """Compatibility launcher preserving the pre-launcher Hermes CLI contract."""

    def __init__(
        self, profile: str, agent_slug: str, hermes_home: str | None, timeout: float
    ):
        self.profile = profile
        self.agent_slug = agent_slug
        self.hermes_home = hermes_home
        self.timeout = timeout
        self._sessions: dict[tuple[str, str], str] = {}
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        self._active: dict[
            str, tuple[subprocess.Popen[str], int | None, int | None]
        ] = {}
        self._cancelled: set[str] = set()
        self._closed = False
        self._guard = threading.Lock()

    def _lock(self, key: tuple[str, str]) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def _state_db(self) -> str | None:
        return os.path.join(self.hermes_home, "state.db") if self.hermes_home else None

    def _lookup(self, title: str) -> str:
        db = self._state_db()
        if not db or not os.path.exists(db):
            return ""
        try:
            with sqlite3.connect(db, timeout=5) as con:
                row = con.execute(
                    "SELECT id FROM sessions WHERE title = ? ORDER BY started_at DESC LIMIT 1",
                    (title,),
                ).fetchone()
            return str(row[0]) if row else ""
        except Exception:
            return ""

    def _latest(self, started_after: float) -> str:
        db = self._state_db()
        if not db or not os.path.exists(db):
            return ""
        try:
            with sqlite3.connect(db, timeout=5) as con:
                row = con.execute(
                    "SELECT id FROM sessions WHERE source = 'a2a' AND started_at >= ? ORDER BY started_at DESC LIMIT 1",
                    (started_after - 2.0,),
                ).fetchone()
            return str(row[0]) if row else ""
        except Exception:
            return ""

    def _title(self, session_id: str, title: str) -> None:
        db = self._state_db()
        if not db or not os.path.exists(db) or not session_id:
            return
        try:
            with sqlite3.connect(db, timeout=5) as con:
                con.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
                )
        except Exception:
            return

    def send(self, request: LaunchRequest) -> LaunchOutcome:
        with self._guard:
            if self._closed or request.task_id in self._cancelled:
                return LaunchOutcome(protocol.STATE_FAILED, "[profile was cancelled]")
        safe_context = (
            "".join(
                char if char.isalnum() or char in "_.-" else "-"
                for char in request.context_id
            ).strip("-._")[:96]
            or "ctx"
        )
        title, key = (
            f"a2a-{self.agent_slug}-{safe_context}",
            (self.agent_slug, safe_context),
        )
        with self._lock(key):
            persisted_session = self._sessions.get(key) or self._lookup(title)
            command = ["hermes", "chat", "-q", request.prompt, "-Q", "--source", "a2a"]
            if persisted_session:
                command.extend(["--resume", persisted_session])
            environment = os.environ.copy()
            if self.hermes_home:
                environment["HERMES_HOME"] = self.hermes_home
            environment["HERMES_A2A_PEER"] = request.peer
            started = time.time()
            process: subprocess.Popen[str] | None = None
            pgid: int | None = None
            owner_session: int | None = None
            try:
                kwargs: dict[str, object] = {
                    "stdin": subprocess.DEVNULL,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                    "text": True,
                    "env": environment,
                    "shell": False,
                }
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    kwargs["start_new_session"] = True
                process = subprocess.Popen(command, **kwargs)
                pgid = os.getpgid(process.pid) if os.name != "nt" else None
                owner_session = os.getsid(process.pid) if os.name != "nt" else None
                with self._guard:
                    if self._closed or request.task_id in self._cancelled:
                        ProcessLauncher._terminate(process, pgid, owner_session)
                        return LaunchOutcome(
                            protocol.STATE_FAILED, "[profile was cancelled]"
                        )
                    self._active[request.task_id] = (process, pgid, owner_session)
                stdout, stderr = process.communicate(timeout=self.timeout)
                if process.returncode:
                    return LaunchOutcome(
                        protocol.STATE_FAILED,
                        security.redact_outbound(
                            (
                                stderr
                                or stdout
                                or f"profile exited {process.returncode}"
                            ).strip()[-2000:]
                        ),
                    )
                if not persisted_session:
                    persisted_session = self._latest(started)
                    if persisted_session:
                        self._sessions[key] = persisted_session
                        self._title(persisted_session, title)
                return LaunchOutcome(
                    protocol.STATE_COMPLETED,
                    security.redact_outbound((stdout or "").strip()),
                    persisted_session or None,
                )
            except subprocess.TimeoutExpired:
                if process:
                    ProcessLauncher._terminate(process, pgid, owner_session)
                return LaunchOutcome(
                    protocol.STATE_FAILED, "[profile did not reply in time]"
                )
            except Exception as exc:
                return LaunchOutcome(
                    protocol.STATE_FAILED,
                    security.redact_outbound(f"Profile dispatch failed: {exc}"),
                )
            finally:
                if process:
                    ProcessLauncher._terminate(process, pgid, owner_session)
                with self._guard:
                    self._active.pop(request.task_id, None)

    def cancel(self, task_id: str) -> bool:
        with self._guard:
            self._cancelled.add(task_id)
            owned = self._active.get(task_id)
        if owned:
            ProcessLauncher._terminate(*owned)
        return True

    def close(self) -> None:
        with self._guard:
            self._closed = True
            processes = tuple(self._active.values())
        for process, pgid, session_id in processes:
            ProcessLauncher._terminate(process, pgid, session_id)


class LauncherManager:
    """Owns validated route launchers and their explicit captured Hermes homes."""

    def __init__(self, hermes_home: str | None):
        self.hermes_home = hermes_home
        self._launchers: dict[str, AgentLauncher] = {}
        self._active: dict[str, AgentLauncher] = {}
        self._closing = False
        self._lock = threading.Lock()

    def add_profile(
        self, slug: str, profile: str, profile_home: str | None, timeout: float
    ) -> None:
        self._launchers[slug] = HermesProfileLauncher(
            profile, slug, profile_home, timeout
        )

    def add_process(self, slug: str, spec: ProcessLauncherSpec) -> None:
        self._launchers[slug] = ProcessLauncher(spec, self.hermes_home)

    def add_pi_rpc(self, slug: str, spec: PiRpcLauncherSpec) -> None:
        from .pi_rpc import PiRpcLauncher

        self._launchers[slug] = PiRpcLauncher(spec)

    def reap_idle(self, now: float | None = None) -> None:
        moment = time.monotonic() if now is None else now
        for launcher in self._launchers.values():
            reap = getattr(launcher, "reap_idle", None)
            if reap is not None:
                reap(moment)

    def send(self, request: LaunchRequest) -> LaunchOutcome:
        with self._lock:
            if self._closing:
                return LaunchOutcome(
                    protocol.STATE_FAILED, "[configured launcher is closing]"
                )
            launcher = self._launchers.get(request.agent_slug)
            if launcher is None:
                return LaunchOutcome(
                    protocol.STATE_FAILED, "[configured launcher is unavailable]"
                )
            self._active[request.task_id] = launcher
        try:
            return launcher.send(request)
        finally:
            with self._lock:
                self._active.pop(request.task_id, None)

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            launcher = self._active.get(task_id)
        return launcher.cancel(task_id) if launcher is not None else False

    def active_task_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._active)

    def close(self) -> None:
        with self._lock:
            self._closing = True
            active = tuple(self._active.items())
            launchers = tuple(self._launchers.values())
        for task_id, launcher in active:
            launcher.cancel(task_id)
        for launcher in launchers:
            launcher.close()
