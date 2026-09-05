"""Verification runner: bootstrap -> build -> test -> start in background ->
readiness loop -> teardown (scoped port of grok-cli's verify flow).

Commands come from the project's own recipe and run with ``shell=True`` on
purpose: a developer tool running the project's own build commands in its own
checkout — the same trust level as the terminal tool.
"""

from __future__ import annotations

import os
import signal
import subprocess
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
# Project-authored shell commands; see module docstring.
_SUBPROCESS_KW: dict[str, Any] = dict(
    shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, errors="replace",
)


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
            "phase": self.phase, "command": self.command, "exitCode": self.exit_code,
            "duration": round(self.duration, 3), "ok": self.ok, "timedOut": self.timed_out,
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
            "url": self.url, "ready": self.ready, "statusCode": self.status_code,
            "duration": round(self.duration, 3), "error": self.error, "outputTail": self.output_tail,
        }


@dataclass
class VerifyResult:
    recipe_name: str
    phases: list[PhaseResult] = field(default_factory=list)
    readiness: ReadinessResult | None = None

    @property
    def ok(self) -> bool:
        return all(p.ok for p in self.phases) and (self.readiness is None or self.readiness.ready)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe_name, "ok": self.ok,
            "phases": [p.to_dict() for p in self.phases],
            "readiness": self.readiness.to_dict() if self.readiness else None,
        }


def _tail(text: str, limit: int = _TAIL_CHARS) -> str:
    return text[-limit:] if len(text) > limit else text


def _run_phase_command(
    phase: str, command: str, root: Path, timeout: float,
    on_output: Callable[[str], None] | None = None,
) -> PhaseResult:
    started = time.monotonic()
    try:
        proc = subprocess.run(command, cwd=str(root), timeout=timeout, **_SUBPROCESS_KW)
        output, exit_code, timed_out = proc.stdout or "", proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        raw = exc.output
        output = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
        exit_code, timed_out = None, True
    duration = time.monotonic() - started
    if on_output and output:
        on_output(output)
    return PhaseResult(phase, command, exit_code, duration, _tail(output), timed_out)


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
    """SIGTERM the app's process group (``start_new_session=True`` on POSIX; just the
    direct child on Windows, which lacks ``os.killpg``), SIGKILL after 10s."""
    if proc.poll() is not None:
        return
    killpg = getattr(os, "killpg", None)
    getpgid = getattr(os, "getpgid", None)
    pgid = None
    if killpg is not None and getpgid is not None:
        try:
            pgid = getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            pass

    def stop(sig: int, fallback: Callable[[], None]) -> None:
        if pgid is not None:
            killpg(pgid, sig)  # windows-footgun: ok — POSIX-only branch (pgid only set when killpg exists)
        else:
            fallback()

    try:
        stop(signal.SIGTERM, proc.terminate)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            stop(getattr(signal, "SIGKILL", signal.SIGTERM), proc.kill)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _run_start_phase(
    recipe: Recipe, root: Path, ready_timeout: float, port_override: int | None = None
) -> ReadinessResult:
    assert recipe.start is not None
    port = port_override or recipe.port or 8000
    url = f"http://127.0.0.1:{port}{recipe.readiness_path}"
    started = time.monotonic()
    # start_new_session: own process group for clean teardown.
    proc = subprocess.Popen(recipe.start, cwd=str(root), start_new_session=True, **_SUBPROCESS_KW)
    output = ""
    try:
        ready, status, error = _poll_readiness(url, ready_timeout)
    finally:
        _terminate_process_group(proc)
        try:
            output = proc.stdout.read() or "" if proc.stdout is not None else ""
        except (OSError, ValueError):
            output = ""
    return ReadinessResult(url, ready, status, time.monotonic() - started, error, _tail(output))


def run_verify(
    root: Path, recipe: Recipe, phases: tuple[str, ...] | list[str] | None = None,
    phase_timeout: float = DEFAULT_PHASE_TIMEOUT, ready_timeout: float = DEFAULT_READY_TIMEOUT,
    skip_start: bool = False, port_override: int | None = None, stop_on_failure: bool = True,
    on_output: Callable[[str], None] | None = None,
) -> VerifyResult:
    """Run the selected command phases sequentially, then (unless ``skip_start`` or a
    phase failed) boot ``recipe.start``, poll readiness, and tear the process group down."""
    root = Path(root)
    selected = tuple(phases) if phases else PHASE_ORDER + ("start",)
    result = VerifyResult(recipe_name=recipe.name)

    for phase in PHASE_ORDER:
        if phase not in selected:
            continue
        for command in getattr(recipe, phase):
            phase_result = _run_phase_command(phase, command, root, phase_timeout, on_output)
            result.phases.append(phase_result)
            if not phase_result.ok and stop_on_failure:
                return result

    if not skip_start and "start" in selected and recipe.start and all(p.ok for p in result.phases):
        result.readiness = _run_start_phase(recipe, root, ready_timeout, port_override)
    return result
