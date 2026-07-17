"""Durable runtime discovery and management for `agy agentapi`."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import psutil

from utils import atomic_json_write

_DEFAULT_PROBE_TIMEOUT_SECONDS = 0.2
_DEFAULT_DISCOVERY_WAIT_SECONDS = 0.1
_DEFAULT_RECENT_LOG_COUNT = 10
_MANAGED_RUNTIME_STATE_NAME = "hermes-antigravity-runtime.json"
_MANAGED_RUNTIME_LOCK_NAME = "hermes-antigravity-runtime.lock"
_MANAGED_RUNTIME_LOG_NAME = "hermes-antigravity-managed.log"
_LOG_PORT_RE = re.compile(r"Language server listening on random port at (\d+) for HTTP")
_DAEMON_IDLE_PROMPT = (
    "Remain idle and keep the Antigravity language server alive for Hermes "
    "delegate_task agentapi sessions. Do not exit unless Hermes stops the runtime."
)
_DAEMON_STARTUP_LOCK = threading.Lock()


def normalize_antigravity_address(value: str | None) -> str:
    """Return a normalized HTTP address or an empty string."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    host = parsed.hostname or ""
    if not host or not parsed.port:
        return ""
    scheme = parsed.scheme or "http"
    return f"{scheme}://{host}:{parsed.port}"


def probe_antigravity_address(
    address: str | None, timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS
) -> bool:
    """True when ``address`` accepts a bounded TCP connection."""

    normalized = normalize_antigravity_address(address)
    if not normalized:
        return False
    parsed = urlparse(normalized)
    assert parsed.hostname is not None  # normalized form guarantees this
    assert parsed.port is not None
    try:
        with socket.create_connection(
            (parsed.hostname, parsed.port),
            timeout=max(float(timeout_seconds), 0.01),
        ):
            return True
    except OSError:
        return False


def parse_antigravity_log_address(line: str) -> str | None:
    """Extract the HTTP address from the known Antigravity log line."""

    match = _LOG_PORT_RE.search(str(line or ""))
    if not match:
        return None
    return f"http://localhost:{match.group(1)}"


@dataclass
class AntigravityRuntimeState:
    status: str
    address: str
    project_id: str
    daemon_pid: int
    agy_pid: int
    log_path: str

    @classmethod
    def from_dict(cls, payload: dict | None) -> "AntigravityRuntimeState | None":
        if not isinstance(payload, dict):
            return None
        try:
            return cls(
                status=str(payload.get("status") or "").strip() or "unknown",
                address=str(payload.get("address") or "").strip(),
                project_id=str(payload.get("project_id") or "").strip(),
                daemon_pid=int(payload.get("daemon_pid") or 0),
                agy_pid=int(payload.get("agy_pid") or 0),
                log_path=str(payload.get("log_path") or "").strip(),
            )
        except Exception:
            return None

    def to_dict(self) -> dict[str, str | int]:
        return {
            "status": self.status,
            "address": self.address,
            "project_id": self.project_id,
            "daemon_pid": self.daemon_pid,
            "agy_pid": self.agy_pid,
            "log_path": self.log_path,
        }


def _gemini_root(home: str | Path) -> Path:
    return Path(home) / ".gemini" / "antigravity-cli"


def _log_dir(home: str | Path) -> Path:
    return _gemini_root(home) / "log"


def _cache_dir(home: str | Path) -> Path:
    return _gemini_root(home) / "cache"


def _state_path(home: str | Path) -> Path:
    return _log_dir(home) / _MANAGED_RUNTIME_STATE_NAME


def _lock_path(home: str | Path) -> Path:
    return _log_dir(home) / _MANAGED_RUNTIME_LOCK_NAME


def _managed_log_path(home: str | Path) -> Path:
    return _log_dir(home) / _MANAGED_RUNTIME_LOG_NAME


def load_runtime_state(home: str | Path) -> AntigravityRuntimeState | None:
    """Load the managed runtime state file if it exists and parses."""

    path = _state_path(home)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return AntigravityRuntimeState.from_dict(payload)


def write_runtime_state(home: str | Path, state: AntigravityRuntimeState) -> None:
    """Persist runtime state atomically."""

    atomic_json_write(_state_path(home), state.to_dict(), sort_keys=True)


def resolve_antigravity_project_id(env: dict[str, str], home: str | Path) -> str:
    """Resolve the project id from env first, then the Antigravity cache."""

    project_id = str(env.get("ANTIGRAVITY_PROJECT_ID") or "").strip()
    if project_id:
        return project_id

    cache_file = _cache_dir(home) / "default_project_id.txt"
    try:
        cached = cache_file.read_text(encoding="utf-8").strip()
    except OSError:
        cached = ""
    if cached:
        return cached

    raise RuntimeError(
        "Antigravity `agy agentapi` requires ANTIGRAVITY_PROJECT_ID. Set the env var, "
        f"or write the default project id to {cache_file}. If you only need one-shot "
        "behavior, drop the `agentapi` arg to use `agy -p` print mode instead."
    )


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        return psutil.pid_exists(pid)
    except Exception:
        return False


def _windows_listener_is_agy(address: str | None, *, expected_pid: int | None = None) -> bool:
    """True when ``address`` is owned by a listening ``agy.exe`` process on Windows."""

    normalized = normalize_antigravity_address(address)
    if not normalized:
        return False
    parsed = urlparse(normalized)
    port = parsed.port
    if port is None:
        return False

    try:
        listeners = psutil.net_connections(kind="tcp")
    except Exception:
        return False

    expected = int(expected_pid or 0)
    for listener in listeners:
        if listener.status != psutil.CONN_LISTEN:
            continue
        laddr = getattr(listener, "laddr", ())
        listener_port = getattr(laddr, "port", None)
        if listener_port is None and isinstance(laddr, tuple) and len(laddr) >= 2:
            listener_port = laddr[1]
        if listener_port != port:
            continue

        pid = int(getattr(listener, "pid", 0) or 0)
        if expected > 0 and pid != expected:
            continue

        try:
            name = psutil.Process(pid).name().lower()
        except Exception:
            continue
        if name == "agy.exe":
            return True
    return False


def _tail_log_for_address(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        address = parse_antigravity_log_address(line)
        if address:
            return address
    return None


class _WindowsRuntimeLock:
    """Tiny file lock wrapper for the managed Windows daemon."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle = None

    def acquire(self, *, blocking: bool) -> bool:
        import msvcrt

        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._path, "a+b")
        try:
            handle.seek(0)
            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            try:
                msvcrt.locking(handle.fileno(), mode, 1)
            except OSError:
                return False

            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()).encode("ascii", errors="ignore"))
            handle.flush()
            self._handle = handle
            handle = None
            return True
        finally:
            if handle is not None:
                handle.close()

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        import msvcrt

        try:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        finally:
            handle.close()

    def __enter__(self) -> "_WindowsRuntimeLock":
        if not self.acquire(blocking=True):
            raise RuntimeError(f"Could not acquire runtime lock {self._path}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.release()


class AntigravityRuntimeManager:
    """Resolves a live Antigravity address and starts the managed daemon if needed."""

    def __init__(
        self,
        *,
        command: str,
        cwd: str,
        env_factory: Callable[[], dict[str, str]],
        platform: str | None = None,
        python_executable: str | None = None,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._command = command
        self._cwd = cwd
        self._env_factory = env_factory
        self._platform = platform or sys.platform
        self._python_executable = python_executable or sys.executable
        self._popen = popen
        self._sleep = sleep
        self._monotonic = monotonic
        self._prepare_lock = threading.RLock()
        self._current_env: dict[str, str] | None = None
        self._current_home: Path | None = None

    def prepare_env(self, timeout_seconds: float) -> dict[str, str]:
        """Return the child env with a live address and resolved project id."""

        with self._prepare_lock:
            env = dict(self._env_factory())
            home = Path(str(env.get("HOME") or "").strip() or os.path.expanduser("~"))
            env["HOME"] = str(home)
            self._current_env = env
            self._current_home = home

            project_id = resolve_antigravity_project_id(env, home)
            env["ANTIGRAVITY_PROJECT_ID"] = project_id

            deadline = self._monotonic() + max(float(timeout_seconds), 0.0)
            address = self._resolve_runtime_address(deadline=deadline)
            env["ANTIGRAVITY_LS_ADDRESS"] = address
            return env

    def _resolve_runtime_address(self, *, deadline: float) -> str:
        env = self._require_current_env()

        configured = str(env.get("ANTIGRAVITY_LS_ADDRESS") or "").strip()
        if configured and self._probe_live_address(configured, deadline=deadline):
            return configured

        managed = self._discover_live_address_from_state(deadline=deadline)
        if managed:
            return managed

        discovered = self._discover_live_address_from_logs(deadline=deadline)
        if discovered:
            return discovered

        if self._platform != "win32":
            raise RuntimeError(
                "Antigravity `agy agentapi` could not find a live language-server address. "
                "Automatic startup is only available on Windows. Start `agy -i` yourself "
                "so it publishes a fresh ANTIGRAVITY_LS_ADDRESS, or drop the `agentapi` "
                "arg to use `agy -p` print mode instead."
            )

        self._start_managed_runtime(deadline)
        ready = self._wait_for_runtime_address(deadline)
        if ready:
            return ready

        raise RuntimeError(
            "Antigravity `agy agentapi` could not discover a live language-server "
            "address before the caller timeout expired."
        )

    def _remaining_budget(self, deadline: float) -> float:
        return max(float(deadline) - self._monotonic(), 0.0)

    def _probe_live_address(
        self,
        address: str | None,
        *,
        deadline: float,
        expected_pid: int | None = None,
    ) -> bool:
        remaining = self._remaining_budget(deadline)
        if remaining <= 0:
            return False
        if not probe_antigravity_address(
            address,
            timeout_seconds=min(_DEFAULT_PROBE_TIMEOUT_SECONDS, remaining),
        ):
            return False
        if self._platform != "win32":
            return True
        return _windows_listener_is_agy(address, expected_pid=expected_pid)

    def _discover_live_address_from_state(self, *, deadline: float | None = None) -> str | None:
        state = load_runtime_state(self._require_current_home())
        if state is None or not state.address:
            return None
        limit = float("inf") if deadline is None else deadline
        if self._probe_live_address(state.address, deadline=limit, expected_pid=state.agy_pid):
            return state.address
        return None

    def _discover_live_address_from_logs(self, *, deadline: float | None = None) -> str | None:
        home = self._require_current_home()
        log_dir = _log_dir(home)
        try:
            candidates = sorted(
                (path for path in log_dir.glob("*.log*") if path.is_file()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[:_DEFAULT_RECENT_LOG_COUNT]
        except OSError:
            candidates = []
        for path in candidates:
            if deadline is not None and self._remaining_budget(deadline) <= 0:
                break
            address = _tail_log_for_address(path)
            limit = float("inf") if deadline is None else deadline
            if address and self._probe_live_address(address, deadline=limit):
                return address
        return None

    def _start_managed_runtime(self, deadline: float) -> None:
        home = self._require_current_home()

        with _DAEMON_STARTUP_LOCK:
            live = self._discover_live_address_from_state(deadline=deadline) or self._discover_live_address_from_logs(
                deadline=deadline
            )
            if live:
                return

            state = load_runtime_state(home)
            if state is not None and _pid_is_alive(state.daemon_pid):
                return

            lock = _WindowsRuntimeLock(_lock_path(home))
            if not lock.acquire(blocking=False):
                return
            try:
                state = load_runtime_state(home)
                if state is not None and _pid_is_alive(state.daemon_pid):
                    return
                self._spawn_managed_runtime_daemon()
            finally:
                lock.release()

    def _spawn_managed_runtime_daemon(self) -> None:
        env = dict(self._require_current_env())
        home = self._require_current_home()
        env["HOME"] = str(home)
        env.pop("ANTIGRAVITY_LS_ADDRESS", None)
        project_id = str(env.get("ANTIGRAVITY_PROJECT_ID") or "").strip()
        log_path = _managed_log_path(home)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            self._python_executable,
            "-m",
            "agent.antigravity_runtime",
            "--daemon",
            "--command",
            self._command,
            "--cwd",
            self._cwd,
            "--home",
            str(home),
        ]

        creationflags = 0
        for name in ("DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP"):
            creationflags |= int(getattr(subprocess, name, 0))

        proc = self._popen(
            command,
            cwd=self._cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
        )
        write_runtime_state(
            home,
            AntigravityRuntimeState(
                status="starting",
                address="",
                project_id=project_id,
                daemon_pid=int(getattr(proc, "pid", 0) or 0),
                agy_pid=0,
                log_path=str(log_path),
            ),
        )

    def _wait_for_runtime_address(self, deadline: float) -> str | None:
        while self._monotonic() < deadline:
            live = self._discover_live_address_from_state(deadline=deadline)
            if live:
                return live
            live = self._discover_live_address_from_logs(deadline=deadline)
            if live:
                return live
            remaining = self._remaining_budget(deadline)
            if remaining <= 0:
                break
            self._sleep(min(_DEFAULT_DISCOVERY_WAIT_SECONDS, remaining))
        return None

    def _require_current_env(self) -> dict[str, str]:
        if self._current_env is None:
            raise RuntimeError("runtime prepare_env() context missing env")
        return self._current_env

    def _require_current_home(self) -> Path:
        if self._current_home is None:
            raise RuntimeError("runtime prepare_env() context missing home")
        return self._current_home


class _ManagedAntigravityDaemon:
    """Long-lived Windows daemon that owns the interactive `agy -i` runtime."""

    def __init__(self, *, command: str, cwd: str, home: Path) -> None:
        self._command = command
        self._cwd = cwd
        self._home = home
        self._log_path = _managed_log_path(home)
        self._stop_event = threading.Event()
        self._child = None

    def run(self) -> int:
        if sys.platform != "win32":
            raise RuntimeError("Managed Antigravity daemon is only available on Windows.")

        with _WindowsRuntimeLock(_lock_path(self._home)):
            self._install_signal_handlers()
            while not self._stop_event.is_set():
                child = self._spawn_child()
                self._child = child
                project_id = self._resolve_project_id_for_state()
                write_runtime_state(
                    self._home,
                    AntigravityRuntimeState(
                        status="starting",
                        address="",
                        project_id=project_id,
                        daemon_pid=os.getpid(),
                        agy_pid=int(getattr(child, "pid", 0) or 0),
                        log_path=str(self._log_path),
                    ),
                )

                drain_thread = threading.Thread(
                    target=self._drain_child_output,
                    args=(child,),
                    name="antigravity-pty-drain",
                    daemon=True,
                )
                drain_thread.start()

                address = self._wait_for_child_address(child)
                if address:
                    write_runtime_state(
                        self._home,
                        AntigravityRuntimeState(
                            status="ready",
                            address=address,
                            project_id=project_id,
                            daemon_pid=os.getpid(),
                            agy_pid=int(getattr(child, "pid", 0) or 0),
                            log_path=str(self._log_path),
                        ),
                    )

                while not self._stop_event.is_set() and child.isalive():
                    time.sleep(0.25)

                self._terminate_child(child)
                self._child = None
                if self._stop_event.is_set():
                    break
                time.sleep(0.5)
        return 0

    def _spawn_child(self):
        env = os.environ.copy()
        env["HOME"] = str(self._home)
        env.pop("ANTIGRAVITY_LS_ADDRESS", None)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.write_text("", encoding="utf-8")
        command = [
            self._command,
            "-i",
            _DAEMON_IDLE_PROMPT,
            f"--log-file={self._log_path}",
        ]
        from winpty import PtyProcess

        return PtyProcess.spawn(command, cwd=self._cwd, env=env)

    def _drain_child_output(self, child) -> None:
        while not self._stop_event.is_set():
            try:
                _ = child.read(4096)
            except EOFError:
                return
            except Exception:
                if not child.isalive():
                    return
                time.sleep(0.1)

    def _wait_for_child_address(self, child) -> str | None:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and child.isalive() and not self._stop_event.is_set():
            address = _tail_log_for_address(self._log_path)
            if (
                address
                and probe_antigravity_address(address)
                and _windows_listener_is_agy(
                    address,
                    expected_pid=int(getattr(child, "pid", 0) or 0),
                )
            ):
                return address
            time.sleep(0.1)
        return None

    def _terminate_child(self, child) -> None:
        try:
            child.terminate(force=True)
        except Exception:
            try:
                child.kill(signal.SIGTERM)
            except Exception:
                pass

    def _resolve_project_id_for_state(self) -> str:
        try:
            return resolve_antigravity_project_id(os.environ.copy(), self._home)
        except RuntimeError:
            return ""

    def _install_signal_handlers(self) -> None:
        def _handle_signal(signum, frame) -> None:
            del signum, frame
            self._stop_event.set()
            if self._child is not None:
                self._terminate_child(self._child)

        for signum_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            signum = getattr(signal, signum_name, None)
            if signum is not None:
                try:
                    signal.signal(signum, _handle_signal)
                except Exception:
                    pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Managed Antigravity runtime daemon")
    parser.add_argument("--daemon", action="store_true", help="run the managed daemon loop")
    parser.add_argument("--command", default="agy")
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--home", default=os.path.expanduser("~"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.daemon:
        return 0
    daemon = _ManagedAntigravityDaemon(
        command=str(args.command),
        cwd=str(args.cwd),
        home=Path(str(args.home)),
    )
    return daemon.run()


if __name__ == "__main__":  # pragma: no cover - exercised via detached runtime startup
    raise SystemExit(main())
