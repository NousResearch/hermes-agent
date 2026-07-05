"""Session trace harvesting helpers for SkillOpt."""

from __future__ import annotations

import re
from typing import Any

_EXCLUDED_SOURCES = {"cron", "subagent", "delegation"}
_COMMAND_RE = re.compile(r"(?:^|\n)\s*((?:python3?|uv|poetry|pytest|npm|pnpm|yarn|bun|git|bash|sh)\b[^\n]*)", re.IGNORECASE)


def harvest_skill_traces(
    db: Any,
    skill_name: str,
    *,
    limit: int = 20,
    window: int = 4,
    exclude_sources: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return context windows anchored near mentions of a skill.

    ``db`` is expected to be a ``SessionDB``-like object exposing
    ``search_messages`` and ``get_messages_around``. The helper is deliberately
    read-only and excludes autonomous cron/subagent scaffolding by default.
    """

    name = str(skill_name or "").strip()
    if not name:
        raise ValueError("skill_name is required")
    excluded = set(exclude_sources or _EXCLUDED_SOURCES)
    phrase = name.replace('"', ' ')
    token = re.sub(r"[^A-Za-z0-9_.:-]+", " ", name).strip() or phrase
    query = f'"{phrase}" OR {token} OR skill_view OR skill_manage'
    hits = db.search_messages(
        query,
        role_filter=["user", "assistant", "tool"],
        limit=limit,
        sort="newest",
        include_inactive=False,
    )
    traces: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for hit in hits:
        source = str(hit.get("source") or "")
        if source in excluded:
            continue
        session_id = str(hit.get("session_id") or "")
        msg_id_raw = hit.get("message_id", hit.get("id"))
        try:
            msg_id = int(msg_id_raw)
        except (TypeError, ValueError):
            continue
        key = (session_id, msg_id)
        if not session_id or key in seen:
            continue
        seen.add(key)
        window_result = db.get_messages_around(session_id, msg_id, window=window)
        messages = window_result.get("window", []) if isinstance(window_result, dict) else window_result
        traces.append(
            {
                "skill_name": name,
                "session_id": session_id,
                "anchor_message_id": msg_id,
                "source": source,
                "anchor": hit,
                "messages": messages,
            }
        )
    return traces


def _message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    return ""


def _extract_commands(messages: list[dict[str, Any]]) -> list[str]:
    commands: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        if str(msg.get("tool_name") or "") not in {"terminal", "process"} and msg.get("role") != "tool":
            continue
        for match in _COMMAND_RE.finditer(_message_text(msg)):
            command = match.group(1).strip()
            if command and command not in seen:
                seen.add(command)
                commands.append(command)
    return commands


def distill_trace_to_skill(trace: dict[str, Any]) -> dict[str, Any]:
    """Distill a successful coding trajectory into a compact skill draft.

    This is a deterministic CODESKILL-style first pass: extract commands and
    coarse workflow phases. LLM rewriting can refine the staged markdown later,
    but this function stays safe, local, and testable.
    """

    name = str(trace.get("skill_name") or "distilled-skill").strip() or "distilled-skill"
    messages = [m for m in trace.get("messages") or [] if isinstance(m, dict)]
    commands = _extract_commands(messages)
    saw_patch = any(str(m.get("tool_name") or "") == "patch" or "*** Begin Patch" in _message_text(m) for m in messages)
    saw_failure = any("fail" in _message_text(m).lower() or "error" in _message_text(m).lower() for m in messages)
    saw_success = any("passed" in _message_text(m).lower() or "success" in _message_text(m).lower() for m in messages)
    steps = ["Reproduce the task or failure with the narrowest command available."]
    if saw_failure:
        steps.append("Read the failure output and isolate the root cause before editing.")
    if saw_patch:
        steps.append("Apply the smallest targeted patch that addresses the root cause.")
    if commands:
        steps.append("Re-run the same verification command until it passes.")
    if saw_success:
        steps.append("Report the exact verification evidence in the final response.")
    skill_markdown = (
        f"---\nname: {name}\ndescription: Distilled from a successful Hermes coding trajectory.\n---\n\n"
        f"# {name}\n\n"
        "## Workflow\n"
        + "\n".join(f"{idx}. {step}" for idx, step in enumerate(steps, 1))
        + "\n\n## Commands observed\n"
        + "\n".join(f"- `{cmd}`" for cmd in commands)
        + "\n\n<!-- Save as SKILL.md after human/eval review. -->\n"
    )
    return {"name": name, "commands": commands, "steps": steps, "skill_markdown": skill_markdown}
