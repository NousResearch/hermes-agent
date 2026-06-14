"""Thin runtime wrapper for invoking the official Hermes Agent process."""

import os
import subprocess
import time

from .contracts import AgentResponse
from .errors import PROCESS_ERROR, RUNTIME_TIMEOUT, RUNTIME_UNAVAILABLE, adapter_error


def default_launcher_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "hermes-agent")


class RuntimeWrapper:
    def __init__(self, launcher_path=None):
        self.launcher_path = launcher_path or default_launcher_path()

    def run(self, request):
        started = time.monotonic()
        if request.dry_run:
            return AgentResponse(
                task_id=request.task_id,
                status="dry_run",
                output="Dry-run delegation accepted; runtime was not invoked.",
                duration_ms=0,
            )

        if not os.path.exists(self.launcher_path):
            return AgentResponse(
                task_id=request.task_id,
                status="unavailable",
                errors=[adapter_error(RUNTIME_UNAVAILABLE, "hermes-agent launcher not found")],
            )

        command = [
            self.launcher_path,
            "--help",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=request.working_directory,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return AgentResponse(
                task_id=request.task_id,
                status="timeout",
                errors=[adapter_error(RUNTIME_TIMEOUT, "Runtime invocation timed out")],
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                duration_ms=_duration_ms(started),
            )
        except OSError as exc:
            return AgentResponse(
                task_id=request.task_id,
                status="unavailable",
                errors=[adapter_error(RUNTIME_UNAVAILABLE, str(exc))],
                duration_ms=_duration_ms(started),
            )

        status = "completed" if completed.returncode == 0 else "failed"
        errors = []
        if completed.returncode != 0:
            errors.append(adapter_error(PROCESS_ERROR, "Runtime exited with code %s" % completed.returncode))
        return AgentResponse(
            task_id=request.task_id,
            status=status,
            output=completed.stdout,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            errors=errors,
            duration_ms=_duration_ms(started),
        )


def _duration_ms(started):
    return int((time.monotonic() - started) * 1000)
