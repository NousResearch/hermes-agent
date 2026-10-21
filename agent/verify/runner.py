"""Verification runner: bootstrap -> build -> test -> start in background ->
readiness loop -> teardown (scoped port of grok-cli's verify flow).

Commands come from the project's own recipe and run with ``shell=True`` on
purpose: a developer tool running the project's own build commands in its own
checkout — the same trust level as the terminal tool.
"""

from __future__ import annotations

import os
import secrets
import re
import math
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
_SANDBOX_HELPER = "/usr/local/sbin/hermes-verify-sandbox"
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



def _run_sandbox_helper(*args: str, timeout: float) -> subprocess.CompletedProcess[str]:
    """Call the root-owned verification sandbox; never fall back to host shell."""
    return subprocess.run(
        ["sudo", "-n", _SANDBOX_HELPER, *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=timeout,
    )


def _run_phase_command(
    phase: str, command: str, root: Path, timeout: float,
    on_output: Callable[[str], None] | None = None,
) -> PhaseResult:
    started = time.monotonic()
    try:
        timeout_seconds = min(3600, max(1, math.ceil(timeout)))
        proc = _run_sandbox_helper(
            "run", str(root), phase, str(timeout_seconds), command,
            timeout=float(timeout_seconds + 15),
        )
        output = proc.stdout or ""
        timed_out = proc.returncode == 124
        exit_code = None if timed_out else proc.returncode
    except subprocess.TimeoutExpired as exc:
        raw = exc.output
        output = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
        exit_code, timed_out = None, True
    except OSError as exc:
        output = f"Failed to invoke verification sandbox: {type(exc).__name__}: {exc}"
        exit_code, timed_out = 127, False
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
    token = secrets.token_hex(16)
    start_timeout = min(3600, max(15, math.ceil(ready_timeout) + 15))
    start_attempted = False
    ready = False
    status: int | None = None
    error: str | None = None
    probe_output = ""
    start_output = ""
    stop_output = ""
    stop_failed = False

    try:
        start_attempted = True
        launch = _run_sandbox_helper(
            "start", str(root), str(start_timeout), token, recipe.start,
            timeout=20.0,
        )
        start_output = launch.stdout or ""
        if launch.returncode != 0 or f"STARTED {token}" not in start_output:
            error = start_output.strip() or f"Sandbox start failed (exit {launch.returncode})"
        else:
            deadline = time.monotonic() + max(0.0, ready_timeout)
            last_probe_error = ""
            while time.monotonic() < deadline:
                try:
                    probe = _run_sandbox_helper("probe", token, url, timeout=6.0)
                    probe_output = probe.stdout or ""
                    match = re.search(r"(?m)^READY (\d+)\s*$", probe_output)
                    if probe.returncode == 0 and match:
                        ready = True
                        status = int(match.group(1))
                        break
                    last_probe_error = probe_output.strip() or f"probe exited {probe.returncode}"
                except subprocess.TimeoutExpired:
                    last_probe_error = "readiness probe helper timed out"
                except OSError as exc:
                    last_probe_error = f"readiness probe helper failed: {exc}"

                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(1.0, remaining))

            if not ready:
                error = f"Readiness timed out after {ready_timeout:g}s"
                if last_probe_error:
                    error += f": {last_probe_error}"
    except subprocess.TimeoutExpired:
        error = "Sandbox start helper timed out"
    except OSError as exc:
        error = f"Sandbox start helper failed: {type(exc).__name__}: {exc}"
    finally:
        if start_attempted:
            try:
                stopped = _run_sandbox_helper("stop", token, timeout=15.0)
                stop_output = stopped.stdout or ""
                if stopped.returncode != 0:
                    stop_failed = True
                    detail = stop_output.strip() or f"stop exited {stopped.returncode}"
                    error = f"Sandbox cleanup failed: {detail}"
                    ready = False
                    status = None
            except (subprocess.TimeoutExpired, OSError) as exc:
                stop_failed = True
                error = f"Sandbox cleanup failed: {type(exc).__name__}: {exc}"
                ready = False
                status = None

    output_parts = []
    if start_output and (error is not None or not ready):
        output_parts.append(start_output)
    if probe_output:
        output_parts.append(probe_output)
    if stop_output:
        output_parts.append(stop_output)
    output = "\n".join(part.rstrip() for part in output_parts if part.strip())
    return ReadinessResult(
        url, ready, status, time.monotonic() - started, error, _tail(output)
    )


def _compose_live_state_reason(root: Path) -> str | None:
    """Why ``docker compose build``/``up`` must not run at *root*, or ``None`` to proceed.

    Read-only ``docker compose ps`` probe. Only a missing docker binary proceeds -- the
    build phase would fail the same way, so there is nothing to protect. A hung daemon
    or a non-zero probe refuses: containers may be live and unobservable, which is
    exactly the #103567 loss window.
    """
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--status", "running", "--format", "{{.Name}}"],
            cwd=root, capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return "docker compose ps timed out after 15s; live containers cannot be ruled out"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return f"docker compose ps failed (exit {result.returncode}): {detail[-1] if detail else 'no output'}"
    names = [line for line in result.stdout.splitlines() if line.strip()]
    return f"this compose project already has running container(s): {', '.join(names)}" if names else None


def run_verify(
    root: Path, recipe: Recipe, phases: tuple[str, ...] | list[str] | None = None,
    phase_timeout: float = DEFAULT_PHASE_TIMEOUT, ready_timeout: float = DEFAULT_READY_TIMEOUT,
    skip_start: bool = False, port_override: int | None = None, stop_on_failure: bool = True,
    on_output: Callable[[str], None] | None = None,
) -> VerifyResult:
    """Run the selected command phases sequentially, then (unless ``skip_start`` or a
    phase failed) boot ``recipe.start``, poll readiness, and tear the process group down.

    A ``compose`` recipe refuses outright when the project already has running
    containers: ``docker compose build`` + ``up`` replaces them on an image-hash
    change, destroying any container-local state they carry -- this has caused a
    real state-loss incident (#103567). The check is best-effort and read-only
    (``docker compose ps``); when it cannot run at all, verify proceeds rather than
    blocking on an unrelated environment gap, matching every other recipe kind's
    behavior when its own tooling is unavailable."""
    root = Path(root)
    selected = tuple(phases) if phases else PHASE_ORDER + ("start",)
    result = VerifyResult(recipe_name=recipe.name)

    mutating = ("build" in selected) or ("start" in selected and not skip_start)
    if recipe.kind == "compose" and mutating:
        reason = _compose_live_state_reason(root)
        if reason:
            result.phases.append(PhaseResult(
                phase="build", command=recipe.build[0] if recipe.build else "docker compose build",
                exit_code=1, duration=0.0, output_tail=(
                    f"Refusing to run: {reason}. `docker compose build` + `up` would replace "
                    "live containers on an image-hash change, destroying any container-local "
                    "state they carry. If you intend to rebuild this live deployment, run "
                    "`docker compose build`/`up` yourself."
                ),
            ))
            return result

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
