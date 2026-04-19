from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


DEFAULT_OMX_COMMAND = ["npx", "-y", "oh-my-codex"]
# Default to the plain/direct OMX exec lane for Hermes handoff.
# Live validation showed this is a safer inner-harness default than
# Ralph, which introduces extra OMX-native planning/runtime behavior.
DEFAULT_WORKFLOW = "plain"
DEFAULT_TIMEOUT_SECONDS = 600



def _detect_auth_failure(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    return any(
        signal in text
        for signal in (
            "401 unauthorized",
            "could not parse your authentication token",
            "missing bearer or basic authentication",
            "failed to connect to websocket: http error: 401",
        )
    )



def _auth_error_message(stderr: str) -> str:
    detail = (stderr or "").strip().splitlines()
    preview = detail[-1] if detail else "authentication failed"
    return f"codex authentication failed: {preview}"



_TOOL_VERSION_CACHE: dict[str, str | None] = {}


def _resolve_tool_path(name: str) -> str:
    # Hermes can be launched from a shell whose PATH is missing the npm-global
    # bin dir even though omx/codex live there. Fall back to the standard user
    # install location so version probes don't silently return None.
    found = shutil.which(name)
    if found:
        return found
    npm_global = Path.home() / ".npm-global" / "bin" / name
    if npm_global.exists():
        return str(npm_global)
    return name


def _tool_version(name: str, args: tuple[str, ...] = ("--version",)) -> str | None:
    if name in _TOOL_VERSION_CACHE:
        return _TOOL_VERSION_CACHE[name]
    try:
        proc = subprocess.run(
            [_resolve_tool_path(name), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = (proc.stdout or proc.stderr or "").strip().splitlines()[0] if (proc.stdout or proc.stderr) else None
    except Exception:
        # Best-effort tool probe; never break execution if version lookup fails
        # (e.g., binary missing, monkeypatched subprocess in tests, OS error).
        version = None
    _TOOL_VERSION_CACHE[name] = version
    return version


def _persist_execution_result(result_path: Path, payload: Mapping[str, Any]) -> None:
    try:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip noisy stdout/stderr so the on-disk artifact stays small;
        # they are still in the in-memory result returned to the caller.
        slim = {k: v for k, v in payload.items() if k not in ("stdout", "stderr")}
        result_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    except Exception as exc:
        # Best-effort persistence — must not mask the real result. Surface to
        # stderr so journalctl shows why downstream executor metadata went
        # missing instead of silently swallowing the failure.
        print(
            f"[omx_executor] failed to persist {result_path}: {exc!r}",
            file=sys.stderr,
        )


def _default_executor_policy() -> dict[str, Any]:
    return {
        "engine": "omx",
        "workflow": DEFAULT_WORKFLOW,
        "launch_mode": "exec",
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
    }



def build_execution_request(
    *,
    request_id: str,
    task_type: str,
    mode: str,
    directive: str,
    routing_artifact_version: str,
    session_id: str | None,
    state_dir: Path | str,
    targets_context: str,
    task_text: str | None = None,
    context_brief_path: Path | str | None = None,
    executor_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state_path = Path(state_dir)
    request_path = state_path / "execution_request.json"
    output_path = state_path / "output.md"
    result_path = state_path / "execution_result.json"

    policy = _default_executor_policy()
    if executor_policy:
        policy.update(dict(executor_policy))

    payload: dict[str, Any] = {
        "request_id": request_id,
        "task_type": task_type,
        "mode": mode,
        "directive": directive,
        "routing_artifact_version": routing_artifact_version,
        "session_id": session_id,
        "state_dir": str(state_path),
        "request_path": str(request_path),
        "output_path": str(output_path),
        "result_path": str(result_path),
        "targets_context": targets_context,
        "executor_policy": policy,
    }
    if task_text:
        payload["task_text"] = task_text
    if context_brief_path:
        payload["context_brief_path"] = str(Path(context_brief_path))
    return payload



def write_execution_request(request: Mapping[str, Any]) -> Path:
    state_dir = Path(str(request["state_dir"]))
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "execution_request.json"
    path.write_text(json.dumps(dict(request), indent=2), encoding="utf-8")
    return path



def render_omx_prompt(request: Mapping[str, Any], *, request_path: Path | str | None = None) -> str:
    request_file = Path(request_path) if request_path else Path(str(request["request_path"]))
    workflow = str(request.get("executor_policy", {}).get("workflow", DEFAULT_WORKFLOW))
    task_text = str(request.get("task_text") or "").strip()
    context_brief_path = request.get("context_brief_path")

    lines = [
        "[OMX EXECUTION REQUEST]",
        f"Read the execution contract from: {request_file}",
        f"Workflow: {workflow}",
        f"Directive: {request.get('directive', '')}",
        f"Task type: {request.get('task_type', '')}",
        "",
        "Follow the Hermes-generated targets/context below while executing the coding task.",
        str(request.get("targets_context", "")).strip(),
    ]
    if context_brief_path:
        lines.extend([
            "",
            f"If present and useful, read additional context from: {context_brief_path}",
        ])
    if task_text:
        lines.extend([
            "",
            "Task:",
            task_text,
        ])
    lines.extend([
        "",
        f"Write your main task output to: {request.get('output_path')}",
        f"Write structured execution metadata to: {request.get('result_path')}",
        "Do not change the contract file path. Keep your result machine-readable.",
        "",
        "Required output.md structure (use Markdown `##` headings for each section):",
        "- Status",
        "- Exact files changed",
        "- Live verification",
        "- Edge cases considered (explicitly address null/empty inputs, boundary conditions, `is None`, `len(...) == 0`, `if not`, `raise`, `try:`/`except`, and `fallback`; say 'not applicable' when truly none)",
        "- Security / safe defaults",
        "- Notes for early-intermediate reader",
        "When writing code files or code snippets, preserve requested type hints/signatures exactly.",
        "For tiny Python helpers, prefer explicit return type hints and short docstrings when they do not conflict with the task.",
        "If error handling or validation is unnecessary, say so explicitly using the concrete keywords above rather than implying it.",
        "Avoid unsupported absolute claims; tie status statements to the runtime verification evidence you observed.",
        "For small code changes, include the final file contents or exact code snippets in fenced code blocks so Hermes Phase 2 can score correctness and clarity from the artifact itself.",
    ])
    return "\n".join(line for line in lines if line is not None).strip()



def resolve_omx_command(command_override: str | None = None) -> list[str]:
    if command_override and command_override.strip():
        return shlex.split(command_override)

    local_omx = shutil.which("omx")
    if local_omx:
        return [local_omx]

    return list(DEFAULT_OMX_COMMAND)



def build_omx_exec_command(
    request: Mapping[str, Any],
    *,
    request_path: Path | str | None = None,
    command_override: str | None = None,
) -> list[str]:
    cmd = resolve_omx_command(command_override)
    prompt = render_omx_prompt(request, request_path=request_path)
    state_dir = str(request.get("state_dir") or "")

    argv = [*cmd, "exec"]
    argv.extend(["--sandbox", "workspace-write"])
    # State dirs (e.g. /home/.../rql/state/<id>) are not git repos; without this
    # flag codex aborts with "Not inside a trusted directory".
    argv.append("--skip-git-repo-check")
    if state_dir:
        argv.extend(["--add-dir", state_dir])
    argv.append(prompt)
    return argv



def execute_request(
    request: Mapping[str, Any],
    *,
    workdir: Path | str | None = None,
    command_override: str | None = None,
) -> dict[str, Any]:
    request_path = write_execution_request(request)
    cmd = build_omx_exec_command(request, request_path=request_path, command_override=command_override)
    cwd = str(Path(workdir) if workdir is not None else Path(str(request["state_dir"])))
    timeout = int(request.get("executor_policy", {}).get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))

    workflow = str(request.get("executor_policy", {}).get("workflow", DEFAULT_WORKFLOW))
    launch_mode = str(request.get("executor_policy", {}).get("launch_mode", "exec"))
    output_path = Path(str(request["output_path"]))
    result_path = Path(str(request["result_path"]))

    # Capture tool versions up-front so every return path — including the
    # failure paths — can persist them. Otherwise routing_outcomes.jsonl loses
    # executor identity whenever Codex exits without writing artifacts.
    omx_version = _tool_version("omx")
    codex_version = _tool_version("codex")

    base: dict[str, Any] = {
        "request_id": request.get("request_id"),
        "engine": "omx",
        "workflow": workflow,
        "launch_mode": launch_mode,
        "output_path": str(output_path),
    }
    if omx_version:
        base["omx_version"] = omx_version
    if codex_version:
        base["codex_version"] = codex_version

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        result = {
            **base,
            "status": "failed",
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "auth_failed": False,
            "executor_unavailable": True,
            "error": f"omx executor not installed or not on PATH: {exc}",
        }
        _persist_execution_result(result_path, result)
        return result
    except subprocess.TimeoutExpired as exc:
        result = {
            **base,
            "status": "failed",
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "auth_failed": False,
            "executor_unavailable": False,
            "error": f"omx execution timed out after {timeout}s",
        }
        _persist_execution_result(result_path, result)
        return result

    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}
        for key, value in base.items():
            result.setdefault(key, value)
        result.setdefault("returncode", proc.returncode)
        _persist_execution_result(result_path, result)
        return result

    if output_path.exists():
        result = {
            **base,
            "status": "completed" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "auth_failed": _detect_auth_failure(proc.stdout, proc.stderr),
        }
        _persist_execution_result(result_path, result)
        return result

    auth_failed = _detect_auth_failure(proc.stdout, proc.stderr)
    error_message = _auth_error_message(proc.stderr) if auth_failed else "omx execution did not produce output artifacts"
    result = {
        **base,
        "status": "failed",
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "auth_failed": auth_failed,
        "error": error_message,
    }
    _persist_execution_result(result_path, result)
    return result
