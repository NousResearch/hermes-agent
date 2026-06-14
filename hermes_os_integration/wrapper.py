"""Thin runtime wrapper for invoking the official Hermes Agent process."""

import os
import subprocess
import time

from .contracts import AgentResponse
from .errors import PROCESS_ERROR, RUNTIME_TIMEOUT, RUNTIME_UNAVAILABLE, adapter_error
from .architecture_first import load_constitution


def default_launcher_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "hermes-agent")


class RuntimeWrapper:
    def __init__(self, launcher_path=None, max_retries=0):
        self.launcher_path = launcher_path or default_launcher_path()
        self.max_retries = int(max_retries)

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

        command = build_oneshot_command(self.launcher_path, request)
        attempts = 0
        last_response = None
        while attempts <= self.max_retries:
            last_response = self._run_command(command, request, started)
            if last_response.status != "timeout":
                return last_response
            attempts += 1
        return last_response

    def _run_command(self, command, request, started):
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


def build_runtime_prompt(request):
    constitution = load_constitution()
    sections = [
        "Hermes OS delegated task",
        "",
        "Task ID: " + request.task_id,
        "Project ID: " + request.project_id,
        "Agent kind: " + request.agent_kind,
        "",
        "Objective:",
        request.prompt,
        "",
        "Source-of-truth boundary:",
        "Hermes OS owns projects, tasks, workflows, dashboards, approvals, state, and memory.",
        "You are a worker runtime. Produce artifacts; do not claim ownership of source-of-truth state.",
        "",
        "Constitution:",
    ]
    sections.extend("- " + rule for rule in constitution["rules"])
    sections.extend([
        "",
        "Tool policy:",
        "Allowed tools: " + ", ".join(request.tool_policy.allowed_tools or ["none specified"]),
        "Denied tools: " + ", ".join(request.tool_policy.denied_tools or ["none specified"]),
        "Requires approval: " + str(request.tool_policy.require_approval).lower(),
    ])
    if request.context:
        sections.extend(["", "Context:"])
        for key in sorted(request.context):
            sections.append(str(key) + ": " + str(request.context[key]))
    return "\n".join(sections)


def build_oneshot_command(launcher_path, request):
    return [
        launcher_path,
        "--oneshot",
        build_runtime_prompt(request),
    ]


def _duration_ms(started):
    return int((time.monotonic() - started) * 1000)
