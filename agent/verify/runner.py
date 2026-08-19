"""Verification runner: execute a Recipe's phases and smoke-test the app.

Scoped port of the execution flow grok-cli's verify sub-agent performs
(install/bootstrap -> build -> test -> start in background -> curl-style
readiness loop -> teardown), reimplemented as a plain subprocess runner.

Commands come from the project's own recipe (its package.json scripts,
Makefile targets, etc.) and are executed with ``shell=True`` on purpose:
this is a developer tool running the project's own build commands in the
project's own checkout — the same trust level as the terminal tool.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.verify.recipes import Recipe

DEFAULT_PHASE_TIMEOUT = 600.0
DEFAULT_READY_TIMEOUT = 60.0
_TAIL_CHARS = 2000
PHASE_ORDER = ("bootstrap", "build", "test")


@dataclass
class PhaseResult:
    phase: str
    command: str
    exit_code: int | None
    duration: float
    output_tail: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "command": self.command,
            "exitCode": self.exit_code,
            "duration": round(self.duration, 3),
            "ok": self.ok,
            "timedOut": self.timed_out,
            "outputTail": self.output_tail,
        }


@dataclass
class ReadinessResult:
    url: str
    ready: bool
    status_code: int | None
    duration: float
    error: str | None = None
    output_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "ready": self.ready,
            "statusCode": self.status_code,
            "duration": round(self.duration, 3),
            "error": self.error,
            "outputTail": self.output_tail,
        }


@dataclass
class VerifyResult:
    recipe_name: str
    phases: list[PhaseResult] = field(default_factory=list)
    readiness: ReadinessResult | None = None

    @property
    def ok(self) -> bool:
        phases_ok = all(p.ok for p in self.phases)
        readiness_ok = self.readiness.ready if self.readiness is not None else True
        return phases_ok and readiness_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe_name,
            "ok": self.ok,
            "phases": [p.to_dict() for p in self.phases],
            "readiness": self.readiness.to_dict() if self.readiness else None,
        }


def _tail(text: str, limit: int = _TAIL_CHARS) -> str:
    return text[-limit:] if len(text) > limit else text


class _BoundedOutputReader:
    """Drain a subprocess's stdout while it runs, keeping only the last
    ``limit`` characters.

    ``subprocess.run(..., stdout=subprocess.PIPE)`` buffers the ENTIRE
    output in memory before ``_tail()`` ever gets a chance to truncate it —
    a verbose phase command (a chatty test runner, a build tool with
    progress spam) can grow that buffer without bound for the whole
    ``timeout`` window. Draining concurrently with a bounded tail caps the
    worst case at ``limit`` regardless of how much the command prints.
    """

    def __init__(self, stream: Any, limit: int = _TAIL_CHARS) -> None:
        self._stream = stream
        self._limit = limit
        self._buf = ""
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._pump, daemon=True)

    def start(self) -> "_BoundedOutputReader":
        self._thread.start()
        return self

    def _pump(self) -> None:
        try:
            for line in self._stream:
                with self._lock:
                    self._buf = (self._buf + line)[-self._limit :]
        except (OSError, ValueError):
            # Pipe closed under us (process killed) — whatever we captured stands.
            pass

    def result(self, timeout: float = 5.0) -> str:
        """Captured tail. Never raises: diagnostics must not fail the run."""
        self._thread.join(timeout)
        with self._lock:
            return self._buf


def _run_phase_command(
    phase: str,
    command: str,
    root: Path,
    timeout: float,
    on_output: Callable[[str], None] | None = None,
) -> PhaseResult:
    started = time.monotonic()
    # start_new_session puts the command (and every process it spawns) in its
    # own process group, so a timeout can terminate the WHOLE tree via
    # _terminate_process_group — the same pattern _run_start_phase already
    # uses. Plain proc.kill() only signals the direct `sh` wrapper; a phase
    # that spawned children (jest workers, npm build chains, dev servers)
    # would leave them orphaned and still running, holding ports/fds.
    proc = subprocess.Popen(
        command,
        shell=True,  # project-authored commands; see module docstring
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group for clean teardown
        text=True,
        errors="replace",
    )
    reader = _BoundedOutputReader(proc.stdout).start() if proc.stdout is not None else None
    try:
        exit_code: int | None = proc.wait(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc)
        exit_code = None
        timed_out = True
    output = reader.result() if reader is not None else ""
    duration = time.monotonic() - started
    if on_output and output:
        on_output(output)
    return PhaseResult(
        phase=phase,
        command=command,
        exit_code=exit_code,
        duration=duration,
        output_tail=_tail(output),
        timed_out=timed_out,
    )


def _poll_readiness(url: str, timeout: float, interval: float = 1.0) -> tuple[bool, int | None, str | None]:
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return True, resp.status, None
        except urllib.error.HTTPError as exc:
            # The server answered — it is up, even if it returned 4xx/5xx.
            return True, exc.code, None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(interval)
    return False, None, last_error


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Terminate the started app and its whole process group cleanly.

    On POSIX the child is spawned with ``start_new_session=True`` so we can
    signal the whole group; on Windows (no ``os.killpg``) we fall back to
    terminating just the direct child.
    """
    if proc.poll() is not None:
        return
    killpg = getattr(os, "killpg", None)
    getpgid = getattr(os, "getpgid", None)
    pgid = None
    if killpg is not None and getpgid is not None:
        try:
            pgid = getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            pgid = None
    try:
        if pgid is not None and killpg is not None:
            killpg(pgid, signal.SIGTERM)  # windows-footgun: ok — POSIX-only branch (killpg checked above)
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if pgid is not None and killpg is not None:
                killpg(pgid, signal.SIGKILL)  # windows-footgun: ok — POSIX-only branch (killpg checked above)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _run_start_phase(
    recipe: Recipe,
    root: Path,
    ready_timeout: float,
    port_override: int | None = None,
) -> ReadinessResult:
    assert recipe.start is not None
    port = port_override or recipe.port or 8000
    url = f"http://127.0.0.1:{port}{recipe.readiness_path}"
    started = time.monotonic()
    proc = subprocess.Popen(
        recipe.start,
        shell=True,  # project-authored command; see module docstring
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group for clean teardown
        text=True,
        errors="replace",
    )
    output = ""
    try:
        ready, status, error = _poll_readiness(url, ready_timeout)
    finally:
        _terminate_process_group(proc)
        try:
            if proc.stdout is not None:
                output = proc.stdout.read() or ""
        except (OSError, ValueError):
            output = ""
    return ReadinessResult(
        url=url,
        ready=ready,
        status_code=status,
        duration=time.monotonic() - started,
        error=error,
        output_tail=_tail(output),
    )


def run_verify(
    root: Path,
    recipe: Recipe,
    phases: tuple[str, ...] | list[str] | None = None,
    phase_timeout: float = DEFAULT_PHASE_TIMEOUT,
    ready_timeout: float = DEFAULT_READY_TIMEOUT,
    skip_start: bool = False,
    port_override: int | None = None,
    stop_on_failure: bool = True,
    on_output: Callable[[str], None] | None = None,
) -> VerifyResult:
    """Run a verify pass for ``recipe`` at project ``root``.

    Executes the selected command phases sequentially, then (unless
    ``skip_start`` or a phase failed) launches ``recipe.start`` in the
    background, polls the readiness URL, and tears the process group down.
    """
    root = Path(root)
    selected = tuple(phases) if phases else PHASE_ORDER + ("start",)
    result = VerifyResult(recipe_name=recipe.name)

    failed = False
    for phase in PHASE_ORDER:
        if phase not in selected:
            continue
        for command in getattr(recipe, phase):
            phase_result = _run_phase_command(phase, command, root, phase_timeout, on_output)
            result.phases.append(phase_result)
            if not phase_result.ok:
                failed = True
                if stop_on_failure:
                    return result

    if skip_start or "start" not in selected or failed or not recipe.start:
        return result

    result.readiness = _run_start_phase(recipe, root, ready_timeout, port_override)
    return result
