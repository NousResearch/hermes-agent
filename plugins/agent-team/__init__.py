from __future__ import annotations

import json
from typing import Any


HELP_TEXT = """\
/team — run a small Hermes subagent team

Usage:
  /team <task>
  /team full <task>
  /team research <task>
  /team build <task>
  /team review <task>

Modes:
  full      Run researcher, builder, and reviewer in parallel.
  research  Run a web-focused research worker.
  build     Run a coding/implementation worker.
  review    Run a review/risk-analysis worker.
"""


MODE_ALIASES = {
    "full": "full",
    "all": "full",
    "research": "research",
    "r": "research",
    "build": "build",
    "b": "build",
    "review": "review",
    "v": "review",
}


ROLE_LABELS = {
    "researcher": "Researcher",
    "builder": "Builder",
    "reviewer": "Reviewer",
}


FULL_MODE_MAX_ITERATIONS = 20


def _split_mode(raw_args: str) -> tuple[str, str]:
    text = raw_args.strip()
    if not text:
        return "help", ""
    first, _, rest = text.partition(" ")
    lowered = first.lower()
    if lowered in {"help", "-h", "--help"}:
        return "help", ""
    if lowered in MODE_ALIASES:
        return MODE_ALIASES[lowered], rest.strip()
    return "full", text


def _base_context(task: str) -> str:
    return (
        "User request:\n"
        f"{task}\n\n"
        "Return a concise, actionable summary. Match the user's language when clear. "
        "If you changed or inspected files, include exact paths and verification steps."
    )


def _task(role: str, goal: str, task: str, toolsets: list[str]) -> dict[str, Any]:
    return {
        "goal": goal,
        "context": _base_context(task),
        "toolsets": toolsets,
        "role": "leaf",
        "team_role": role,
    }


def _build_tasks(mode: str, task: str) -> list[dict[str, Any]]:
    presets = {
        "research": [
            _task(
                "researcher",
                "Act as a research agent. Gather relevant facts, constraints, options, and open questions for the user request.",
                task,
                ["web"],
            )
        ],
        "build": [
            _task(
                "builder",
                "Act as an implementation agent. Inspect the project, propose or make the smallest justified implementation, and report exact changes.",
                task,
                ["terminal", "file"],
            )
        ],
        "review": [
            _task(
                "reviewer",
                "Act as a review agent. Look for correctness, safety, maintainability, missing tests, and operational risks.",
                task,
                ["terminal", "file"],
            )
        ],
    }
    if mode == "full":
        return presets["research"] + presets["build"] + presets["review"]
    return presets[mode]


def _format_delegate_result(raw: str, role_order: list[str] | None = None) -> str:
    try:
        data = json.loads(raw)
    except Exception:
        return raw

    if isinstance(data, dict) and data.get("error"):
        error = str(data["error"])
        if "parent agent context" in error or "active Hermes agent context" in error:
            return "Cannot run /team here: delegate_task requires an active Hermes agent context. Start an interactive Hermes session and try again."
        return f"Agent team failed: {error}"

    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return raw

    total = data.get("total_duration_seconds")
    lines = ["Agent team result" + (f" ({total}s)" if total is not None else "")]

    for index, entry in enumerate(results, start=1):
        if not isinstance(entry, dict):
            lines.append(f"\n{index}. Unknown result: {entry}")
            continue
        role = entry.get("team_role") or entry.get("role")
        if not role and role_order and index <= len(role_order):
            role = role_order[index - 1]
        role = role or f"agent-{index}"
        label = ROLE_LABELS.get(str(role), str(role).title())
        status = entry.get("status", "unknown")
        duration = entry.get("duration_seconds")
        heading = f"\n{index}. {label} — {status}"
        if duration is not None:
            heading += f" ({duration}s)"
        lines.append(heading)
        summary = entry.get("summary") or entry.get("error") or "No summary returned."
        lines.append(str(summary).strip())

    return "\n".join(lines)


def register(ctx) -> None:
    def handle_team(raw_args: str) -> str:
        mode, task = _split_mode(raw_args)
        if mode == "help":
            return HELP_TEXT
        if not task:
            return HELP_TEXT
        tasks = _build_tasks(mode, task)
        dispatch_args: dict[str, Any] = {"tasks": tasks}
        if mode == "full":
            dispatch_args["max_iterations"] = FULL_MODE_MAX_ITERATIONS
        raw_result = ctx.dispatch_tool("delegate_task", dispatch_args)
        role_order = [str(task.get("team_role") or "") for task in tasks]
        return _format_delegate_result(raw_result, role_order)

    ctx.register_command(
        "team",
        handler=handle_team,
        description="Run a small researcher/builder/reviewer subagent team.",
        args_hint="[full|research|build|review] <task>",
    )
