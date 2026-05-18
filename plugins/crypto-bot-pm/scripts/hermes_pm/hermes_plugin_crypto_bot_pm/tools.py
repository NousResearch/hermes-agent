from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/Users/preston/robinhood/crypto_bot")
TIMEOUT_SECONDS = 60
OUTPUT_FORMATS = ("json", "text")

NON_ACTION_BOOLEANS = {
    "calls_gitea_write_api": False,
    "creates_issues": False,
    "creates_labels": False,
    "creates_comments": False,
    "mutates_projects": False,
    "starts_runners": False,
    "starts_workflows": False,
    "runs_workflows": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
    "branch_writer_invoked": False,
    "issue_executor_invoked": False,
}

PLUGIN_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
    "PYTHONPATH": str(REPO_ROOT),
}

COMMAND_TEMPLATES: dict[str, tuple[str, ...]] = {
    "crypto_bot_pm_issue_lifecycle": (
        "python3",
        "scripts/hermes_pm/hermes_pm_issue_lifecycle.py",
        "--issue-index",
        "1",
        "--expected-title",
        "[Hermes PM] Establish initial PM-managed backlog item",
        "--format",
        "{output_format}",
    ),
    "crypto_bot_pm_status": (
        "python3",
        "scripts/hermes_pm/hermes_pm_status.py",
        "--repo-root",
        ".",
        "--live-gitea-read",
        "--format",
        "{output_format}",
    ),
    "crypto_bot_pm_kanban_packet": (
        "python3",
        "scripts/hermes_pm/generate_kanban_proposal_packet.py",
        "--repo-root",
        ".",
        "--project-id",
        "crypto_bot",
        "--live-gitea-read",
        "--format",
        "{output_format}",
    ),
    "crypto_bot_pm_backlog_expansion": (
        "python3",
        "scripts/hermes_pm/generate_backlog_expansion_proposal.py",
        "--repo-root",
        ".",
        "--project-id",
        "crypto_bot",
        "--live-gitea-read",
        "--issue-index",
        "1",
        "--expected-title",
        "[Hermes PM] Establish initial PM-managed backlog item",
        "--format",
        "{output_format}",
    ),
    "crypto_bot_pm_backlog_selection": (
        "python3",
        "scripts/hermes_pm/generate_backlog_selection_packet.py",
        "--repo-root",
        ".",
        "--project-id",
        "crypto_bot",
        "--live-gitea-read",
        "--issue-index",
        "1",
        "--expected-title",
        "[Hermes PM] Establish initial PM-managed backlog item",
        "--format",
        "{output_format}",
    ),
    "crypto_bot_pm_candidate_approval_scope": (
        "python3",
        "scripts/hermes_pm/generate_backlog_candidate_approval_scope.py",
        "--repo-root",
        ".",
        "--project-id",
        "crypto_bot",
        "--candidate-id",
        "pm8-002",
        "--live-gitea-read",
        "--format",
        "{output_format}",
    ),
    "crypto_bot_pm_forge_plan": (
        "python3",
        "scripts/hermes_pm/generate_forge_write_plan.py",
        "--repo-root",
        ".",
        "--project-id",
        "crypto_bot",
        "--format",
        "{output_format}",
    ),
    "crypto_bot_pm_forge_approval_packet": (
        "python3",
        "scripts/hermes_pm/generate_forge_approval_packet.py",
        "--repo-root",
        ".",
        "--project-id",
        "crypto_bot",
        "--format",
        "{output_format}",
    ),
    "crypto_bot_pm_capability_map": (
        "python3",
        "scripts/hermes_pm/map_gitea_forge_capabilities.py",
        "--base-url",
        "http://127.0.0.1:3005",
        "--owner",
        "preston",
        "--repo",
        "crypto_bot",
        "--format",
        "{output_format}",
    ),
    "crypto_bot_pm_development_workstream": (
        "python3",
        "scripts/hermes_pm/generate_development_workstream_packet.py",
        "--repo-root",
        ".",
        "--project-id",
        "crypto_bot",
        "--live-gitea-read",
        "--format",
        "{output_format}",
    ),
    "crypto_bot_pm_development_slice": (
        "python3",
        "scripts/hermes_pm/generate_development_slice_packet.py",
        "--repo-root",
        ".",
        "--project-id",
        "crypto_bot",
        "--format",
        "{output_format}",
    ),
}

ALLOWED_COMMANDS = {
    tuple(
        part.format(output_format=output_format)
        for part in template
    )
    for template in COMMAND_TEMPLATES.values()
    for output_format in OUTPUT_FORMATS
}

FORBIDDEN_COMMAND_FRAGMENTS = (
    "execute_forge_issue_create",
    "forge_issue_executor",
    "apply_approved_write_plan",
    "branch_local_writer",
    "workflow",
    "runner",
    "broker",
    "trading",
    "robinhood",
    "order",
    "position",
    "wallet",
    "runtime",
    "deploy",
)

SECRET_PATTERNS = (
    (
        re.compile(
            r"(?i)(authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9._~+/=-]+"
        ),
        lambda match: f"{match.group(1)}<redacted>",
    ),
    (
        re.compile(
            r"(?i)((?:token|secret|password|passwd|api[_-]?key|"
            r"private[_ -]?key|credential|authorization)\s*[:=]\s*)"
            r"[^\s,;\"']+"
        ),
        lambda match: f"{match.group(1)}<redacted>",
    ),
    (
        re.compile(
            r"(?i)(\"(?:token|secret|password|api[_-]?key|private[_ -]?key|"
            r"credential|authorization)\"\s*:\s*\")[^\"]+(\")"
        ),
        lambda match: f"{match.group(1)}<redacted>{match.group(2)}",
    ),
)


def redact_text(value: str) -> str:
    redacted = value
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _json_result(payload: dict[str, Any]) -> str:
    with_flags = {**payload, **NON_ACTION_BOOLEANS}
    return json.dumps(with_flags, sort_keys=True)


def _coerce_format(args: dict[str, Any] | None) -> str:
    if not isinstance(args, dict):
        return "json"
    raw = str(args.get("output_format") or "json").strip().lower()
    return raw if raw in OUTPUT_FORMATS else "json"


def build_command(tool_name: str, output_format: str = "json") -> list[str]:
    if tool_name not in COMMAND_TEMPLATES:
        raise ValueError(f"Unknown crypto_bot PM tool: {tool_name}")
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}")
    argv = [
        part.format(output_format=output_format)
        for part in COMMAND_TEMPLATES[tool_name]
    ]
    enforce_command_allowlist(argv)
    return argv


def enforce_command_allowlist(argv: list[str] | tuple[str, ...]) -> None:
    command = tuple(argv)
    if command not in ALLOWED_COMMANDS:
        raise ValueError("Command is not in the crypto_bot PM plugin allowlist.")
    joined = " ".join(command).lower()
    for fragment in FORBIDDEN_COMMAND_FRAGMENTS:
        if fragment in joined:
            raise ValueError(
                f"Command contains forbidden fragment: {fragment}"
            )


def _parse_output(stdout: str, output_format: str) -> dict[str, Any]:
    if not stdout:
        return {"stdout_text": ""}
    if output_format == "json":
        try:
            return {"stdout_json": json.loads(stdout)}
        except json.JSONDecodeError:
            return {"stdout_text": stdout, "stdout_json_parse_error": True}
    return {"stdout_text": stdout}


def run_pm_tool(tool_name: str, args: dict[str, Any] | None = None) -> str:
    output_format = _coerce_format(args)
    try:
        argv = build_command(tool_name, output_format)
        completed = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            env=PLUGIN_ENV,
            text=True,
            capture_output=True,
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        stdout = redact_text(completed.stdout.strip())
        stderr = redact_text(completed.stderr.strip())
        payload: dict[str, Any] = {
            "success": completed.returncode == 0,
            "tool": tool_name,
            "repo_root": str(REPO_ROOT),
            "cwd": str(REPO_ROOT),
            "pythonpath": PLUGIN_ENV["PYTHONPATH"],
            "command": argv,
            "returncode": completed.returncode,
            "timeout_seconds": TIMEOUT_SECONDS,
            "output_format": output_format,
            **_parse_output(stdout, output_format),
        }
        if stderr:
            payload["stderr"] = stderr
        if completed.returncode != 0 and "error" not in payload:
            payload["error"] = stderr or "PM command returned a non-zero exit code."
        return _json_result(payload)
    except subprocess.TimeoutExpired as exc:
        return _json_result(
            {
                "success": False,
                "tool": tool_name,
                "repo_root": str(REPO_ROOT),
                "cwd": str(REPO_ROOT),
                "pythonpath": PLUGIN_ENV["PYTHONPATH"],
                "command": list(exc.cmd) if isinstance(exc.cmd, list) else [],
                "timeout_seconds": TIMEOUT_SECONDS,
                "error": "PM command timed out.",
                "stdout_text": redact_text((exc.stdout or "").strip()),
                "stderr": redact_text((exc.stderr or "").strip()),
            }
        )
    except Exception as exc:
        return _json_result(
            {
                "success": False,
                "tool": tool_name,
                "repo_root": str(REPO_ROOT),
                "cwd": str(REPO_ROOT),
                "pythonpath": PLUGIN_ENV["PYTHONPATH"],
                "timeout_seconds": TIMEOUT_SECONDS,
                "error": redact_text(f"{type(exc).__name__}: {exc}"),
            }
        )


def crypto_bot_pm_status(args: dict[str, Any] | None = None, **_: Any) -> str:
    return run_pm_tool("crypto_bot_pm_status", args)


def crypto_bot_pm_issue_lifecycle(
    args: dict[str, Any] | None = None, **_: Any
) -> str:
    return run_pm_tool("crypto_bot_pm_issue_lifecycle", args)


def crypto_bot_pm_kanban_packet(args: dict[str, Any] | None = None, **_: Any) -> str:
    return run_pm_tool("crypto_bot_pm_kanban_packet", args)


def crypto_bot_pm_backlog_expansion(
    args: dict[str, Any] | None = None, **_: Any
) -> str:
    return run_pm_tool("crypto_bot_pm_backlog_expansion", args)


def crypto_bot_pm_backlog_selection(
    args: dict[str, Any] | None = None, **_: Any
) -> str:
    return run_pm_tool("crypto_bot_pm_backlog_selection", args)


def crypto_bot_pm_candidate_approval_scope(
    args: dict[str, Any] | None = None, **_: Any
) -> str:
    return run_pm_tool("crypto_bot_pm_candidate_approval_scope", args)


def crypto_bot_pm_forge_plan(args: dict[str, Any] | None = None, **_: Any) -> str:
    return run_pm_tool("crypto_bot_pm_forge_plan", args)


def crypto_bot_pm_forge_approval_packet(
    args: dict[str, Any] | None = None, **_: Any
) -> str:
    return run_pm_tool("crypto_bot_pm_forge_approval_packet", args)


def crypto_bot_pm_capability_map(args: dict[str, Any] | None = None, **_: Any) -> str:
    return run_pm_tool("crypto_bot_pm_capability_map", args)


def crypto_bot_pm_development_workstream(
    args: dict[str, Any] | None = None, **_: Any
) -> str:
    return run_pm_tool("crypto_bot_pm_development_workstream", args)


def crypto_bot_pm_development_slice(
    args: dict[str, Any] | None = None, **_: Any
) -> str:
    return run_pm_tool("crypto_bot_pm_development_slice", args)


TOOL_HANDLERS = {
    "crypto_bot_pm_issue_lifecycle": crypto_bot_pm_issue_lifecycle,
    "crypto_bot_pm_status": crypto_bot_pm_status,
    "crypto_bot_pm_kanban_packet": crypto_bot_pm_kanban_packet,
    "crypto_bot_pm_backlog_expansion": crypto_bot_pm_backlog_expansion,
    "crypto_bot_pm_backlog_selection": crypto_bot_pm_backlog_selection,
    "crypto_bot_pm_candidate_approval_scope": crypto_bot_pm_candidate_approval_scope,
    "crypto_bot_pm_forge_plan": crypto_bot_pm_forge_plan,
    "crypto_bot_pm_forge_approval_packet": crypto_bot_pm_forge_approval_packet,
    "crypto_bot_pm_capability_map": crypto_bot_pm_capability_map,
    "crypto_bot_pm_development_workstream": crypto_bot_pm_development_workstream,
    "crypto_bot_pm_development_slice": crypto_bot_pm_development_slice,
}


def crypto_bot_pm_status_slash(raw_args: str = "") -> str:
    output_format = "text" if "text" in (raw_args or "").lower() else "json"
    result = json.loads(crypto_bot_pm_status({"output_format": output_format}))
    if result.get("success") and isinstance(result.get("stdout_text"), str):
        return result["stdout_text"]
    return json.dumps(result, sort_keys=True)
