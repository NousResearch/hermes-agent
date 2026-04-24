from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import logging
import re

from gateway.config import TopicResumeConfig
from gateway.session import SessionEntry, SessionSource


logger = logging.getLogger(__name__)


_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)


@dataclass
class TopicResumeContext:
    workspace_id: str | None
    workspace_path: str | None
    summary: str | None
    current_state: str | None
    decisions: list[str]
    open_loops: list[str]
    next_actions: list[str]
    operating_contract: list[str]
    topic_specific_instructions: list[str]
    recent_messages: list[dict[str, str]]
    source_of_truth: list[str]
    was_auto_resume: bool


def _normalize_bullets(body: str | None) -> list[str]:
    if not body:
        return []
    items: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("- "):
            items.append(line[2:].strip())
        else:
            items.append(line)
    return items


def _extract_section(text: str, heading: str) -> str | None:
    matches = list(_SECTION_RE.finditer(text))
    for index, match in enumerate(matches):
        if match.group(1).strip() != heading:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        return text[start:end].strip()
    return None


def resolve_topic_workspace(source: SessionSource, hermes_home: Path) -> Path | None:
    if not source.thread_id:
        return None

    root = hermes_home / "topic-workspaces"
    if not root.exists():
        return None

    for topic_json in root.glob("*/meta/topic.json"):
        try:
            data = json.loads(topic_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            str(data.get("platform", "")).lower() == source.platform.value.lower()
            and str(data.get("chat_id")) == str(source.chat_id)
            and str(data.get("thread_id")) == str(source.thread_id)
        ):
            return topic_json.parent.parent
    return None


def parse_workspace_state(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    summary = _extract_section(text, "Summary") or ""
    current_state = _extract_section(text, "Current State") or ""
    decisions = _normalize_bullets(_extract_section(text, "Decisions"))
    open_loops = _normalize_bullets(_extract_section(text, "Open Loops"))
    next_actions = _normalize_bullets(_extract_section(text, "Next Actions"))
    operating_contract = _normalize_bullets(_extract_section(text, "Operating Contract"))
    topic_specific_instructions = _normalize_bullets(_extract_section(text, "Topic-Specific Instructions"))
    return {
        "summary": summary,
        "current_state": current_state,
        "decisions": decisions,
        "open_loops": open_loops,
        "next_actions": next_actions,
        "operating_contract": operating_contract,
        "topic_specific_instructions": topic_specific_instructions,
    }


def load_recent_topic_messages(session_store, session_id: str, limit: int = 8, max_chars: int = 400) -> list[dict[str, str]]:
    transcript = session_store.load_transcript(session_id) or []
    filtered = [
        {"role": msg.get("role", "unknown"), "content": (msg.get("content") or "")}
        for msg in transcript
        if msg.get("role") in {"user", "assistant"} and msg.get("content")
    ]
    recent = filtered[-limit:]
    results: list[dict[str, str]] = []
    for item in recent:
        content = item["content"]
        if len(content) > max_chars:
            content = content[:max_chars] + "…"
        results.append({"role": item["role"], "content": content})
    return results


def build_topic_resume_context(
    *,
    source: SessionSource,
    session_store,
    session_entry: SessionEntry,
    config: TopicResumeConfig,
    hermes_home: Path,
    is_new_session: bool,
) -> TopicResumeContext | None:
    if not config.enabled:
        return None
    if not source.thread_id:
        return None

    is_auto_reset = bool(getattr(session_entry, "was_auto_reset", False))
    should_trigger = (
        config.trigger_on_auto_reset if is_auto_reset
        else (is_new_session and config.trigger_on_new_session)
    )
    if not should_trigger:
        return None

    workspace = resolve_topic_workspace(source, hermes_home)
    if not workspace:
        return None

    state = parse_workspace_state(workspace / "state.md")
    recent_messages: list[dict[str, str]] = []
    recent_session_id = session_entry.session_id
    if is_new_session and getattr(session_entry, "previous_session_id", None):
        recent_session_id = session_entry.previous_session_id
    if config.include_recent_messages:
        recent_messages = load_recent_topic_messages(
            session_store,
            recent_session_id,
            limit=config.recent_message_count,
            max_chars=config.max_message_chars,
        )

    logger.info(
        "Topic resume: workspace=%s trigger=%s recent_source_session=%s recent_count=%d",
        workspace.name,
        "auto_reset" if bool(getattr(session_entry, "was_auto_reset", False)) else "new_session",
        recent_session_id,
        len(recent_messages),
    )

    source_of_truth = ["state.md"]
    if recent_messages:
        source_of_truth.append(f"recent_messages(session_id={recent_session_id})")

    return TopicResumeContext(
        workspace_id=workspace.name,
        workspace_path=str(workspace),
        summary=state["summary"],
        current_state=state["current_state"],
        decisions=state["decisions"] if config.include_workspace_decisions else [],
        open_loops=state["open_loops"] if config.include_open_loops else [],
        next_actions=state["next_actions"] if config.include_next_actions else [],
        operating_contract=state["operating_contract"],
        topic_specific_instructions=state["topic_specific_instructions"],
        recent_messages=recent_messages,
        source_of_truth=source_of_truth,
        was_auto_resume=bool(getattr(session_entry, "was_auto_reset", False)),
    )


def build_topic_resume_prompt(ctx: TopicResumeContext) -> str:
    lines = [
        "## Topic Resume Context",
        "",
        "This conversation belongs to an ongoing topic workspace. Treat the following as the current source of truth unless the user explicitly changes it.",
        "",
        f"**Workspace:** {ctx.workspace_id or 'unknown'}",
        f"**Resume source:** {' + '.join(ctx.source_of_truth)}",
    ]
    if ctx.summary:
        lines.extend(["", f"**Summary:** {ctx.summary}"])
    if ctx.current_state:
        lines.extend(["", f"**Current State:** {ctx.current_state}"])
    if ctx.operating_contract:
        lines.extend(["", "**Operating Contract:**"])
        lines.extend([f"- {item}" for item in ctx.operating_contract])
    if ctx.topic_specific_instructions:
        lines.extend(["", "**Topic-Specific Instructions:**"])
        lines.extend([f"- {item}" for item in ctx.topic_specific_instructions])
    if ctx.open_loops:
        lines.extend(["", "**Open Loops:**"])
        lines.extend([f"- {item}" for item in ctx.open_loops])
    if ctx.next_actions:
        lines.extend(["", "**Next Actions:**"])
        lines.extend([f"- {item}" for item in ctx.next_actions])
    if ctx.recent_messages:
        lines.extend(["", "**Recent Topic Messages:**"])
        for item in ctx.recent_messages:
            lines.append(f"- {item['role'].title()}: {item['content']}")
    lines.extend([
        "",
        "If your planned reply would violate the operating contract or recent topic workflow, rewrite it to stay consistent with this topic's established behavior.",
    ])
    return "\n".join(lines)
