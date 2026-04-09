"""compress-summary — pre-compression task-state anchor.

Before context compression discards the middle of a conversation, this plugin
scans messages and appends a structured task-state summary to the tail.
Because the compressor protects tail messages, the summary survives and gives
the model a clear picture of completed work and pending goals.

No LLM call is needed — the summary is built heuristically from tool_calls
(the most reliable signal) and assistant message text.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_MARKER = "[TASK-STATE-SUMMARY]"


def register(ctx):
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_hook("pre_compress", _on_pre_compress)


def _on_pre_compress(
    session_id: str = "",
    messages: Optional[List[Dict[str, Any]]] = None,
    approx_tokens: Optional[int] = None,
    **kwargs,
) -> None:
    """Build and inject a task-state summary before compression."""
    if not messages or len(messages) < 10:
        return

    # Guard against double injection
    for msg in messages[-5:]:
        if _MARKER in (msg.get("content") or ""):
            return

    summary = _build_summary(messages)
    if not summary:
        return

    messages.append({"role": "user", "content": summary})
    logger.info(
        "compress-summary: injected %d-char summary into tail (session=%s)",
        len(summary),
        session_id,
    )


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


def _build_summary(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Assemble a structured summary from conversation messages."""
    original_goal = _find_original_goal(messages)
    actions = _extract_actions(messages)
    notes = _extract_progress_notes(messages)
    recent = _extract_recent_user(messages)
    files = _extract_key_files(messages)

    if not original_goal and not actions:
        return None

    parts: List[str] = [_MARKER, ""]

    if original_goal:
        parts += [f"Original request: {original_goal}", ""]

    if actions:
        parts.append("Actions taken:")
        for a in actions:
            parts.append(f"  - {a}")
        parts.append("")

    if notes:
        parts.append("Progress:")
        for n in notes:
            parts.append(f"  - {n}")
        parts.append("")

    if recent:
        parts.append("Recent user instructions:")
        for u in recent:
            parts.append(f"  - {u}")
        parts.append("")

    if files:
        parts.append("Key files:")
        for f in sorted(files):
            parts.append(f"  - {f}")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def _find_original_goal(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Return the first substantive user message."""
    for msg in messages[:10]:
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if len(content) < 8:
            continue
        # Skip injected system content
        if content.startswith(("[", "<!--", "MEMORY", "{")):
            continue
        return _trunc(content, 500)
    return None


def _extract_actions(messages: List[Dict[str, Any]]) -> List[str]:
    """Build a chronological action log from tool_calls."""
    actions: List[str] = []
    seen: Set[str] = set()

    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", "")

            line = _describe_tool_call(name, args)
            if not line:
                continue

            key = line[:80].lower()
            if key not in seen:
                seen.add(key)
                actions.append(line)

    # Cap: keep first 5 + last 15 when too many
    if len(actions) > 20:
        actions = actions[:5] + ["(...earlier actions omitted...)"] + actions[-14:]

    return actions


def _describe_tool_call(name: str, args_str: str) -> Optional[str]:
    """Human-readable one-liner for a tool call."""
    if name in ("terminal", "execute_code"):
        cmd = _jfield(args_str, "command") or _jfield(args_str, "code")
        if cmd:
            return f"[{name}] {_trunc(cmd, 120)}"
    elif name in ("write_file", "patch"):
        path = _jfield(args_str, "path") or "?"
        return f"[{name}] {path}"
    elif name == "read_file":
        return f"[read] {_jfield(args_str, 'path') or '?'}"
    elif name == "search_files":
        pat = _jfield(args_str, "pattern") or "?"
        return f"[search] pattern={_trunc(pat, 60)}"
    elif name == "delegate_task":
        goal = _jfield(args_str, "goal") or _trunc(args_str, 100)
        return f"[delegate] {_trunc(goal, 120)}"
    elif name == "browser_navigate":
        return f"[browse] {_trunc(_jfield(args_str, 'url') or '?', 100)}"
    elif name == "memory":
        action = _jfield(args_str, "action") or "?"
        return f"[memory:{action}]"
    elif name == "todo":
        return "[todo] updated"
    elif name:
        return f"[{name}]"
    return None


def _extract_progress_notes(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract the first sentence of each assistant message as a progress note."""
    notes: List[str] = []
    seen: Set[str] = set()

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = (msg.get("content") or "").strip()
        if not content or len(content) < 8:
            continue

        # First line, then first sentence
        line = content.split("\n")[0].strip()
        for sep in ("\u3002", ".", "\uff01", "!", "\uff1b", ";"):
            idx = line.find(sep)
            if idx > 0:
                line = line[: idx + 1]
                break

        line = _trunc(line, 150)
        key = line[:60].lower()
        if key not in seen and len(line) > 5:
            seen.add(key)
            notes.append(line)

    return notes[-10:]


def _extract_recent_user(messages: List[Dict[str, Any]]) -> List[str]:
    """Last few user messages for current-context awareness."""
    recent: List[str] = []
    for msg in messages[-20:]:
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if len(content) < 5 or content.startswith("[") or _MARKER in content:
            continue
        recent.append(_trunc(content, 200))
    return recent[-5:]


def _extract_key_files(messages: List[Dict[str, Any]]) -> Set[str]:
    """Collect file paths from write_file / patch / read_file tool calls."""
    files: Set[str] = set()
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function", {})
            name = func.get("name", "")
            if name in ("write_file", "patch", "read_file"):
                path = _jfield(func.get("arguments", ""), "path")
                if path:
                    files.add(path)
    return files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trunc(text: str, n: int) -> str:
    """Collapse whitespace and truncate."""
    text = text.replace("\n", " ").strip()
    return text[:n] + "..." if len(text) > n else text


def _jfield(json_str: str, field: str) -> Optional[str]:
    """Quick regex extraction of a JSON string field value."""
    if not json_str:
        return None
    m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str)
    if m:
        return (
            m.group(1)
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\\\", "\\")
        )
    return None
