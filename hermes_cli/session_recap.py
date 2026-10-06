"""Session recap — summarize the current session from in-memory history. Pure local computation: no
LLM call, no auxiliary model, no prompt-cache invalidation. A recap should be instant and free."""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from tools.ansi_strip import sanitize_display_text

_RECENT_TURN_WINDOW = 20  # user/assistant turns considered "recent activity"
_PROMPT_PREVIEW_CHARS = 140
_ASSISTANT_PREVIEW_CHARS = 200
_MAX_FILES_LISTED = 5

# File-touching tool name -> argument key holding the path.
_FILE_EDIT_TOOLS: Mapping[str, str] = {
    "write_file": "path", "patch": "path", "read_file": "path",
    "skill_manage": "file_path", "skill_view": "file_path",
}


def _coerce_text(value: Any) -> str:
    """Flatten ``content`` (string, or list of blocks whose text-like parts are joined) into a string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return str(value)
    parts: List[str] = []
    for block in value:
        text = block if isinstance(block, str) else block.get("text") if isinstance(block, Mapping) else None
        if isinstance(text, str) and (text or isinstance(block, str)):
            parts.append(text)
    return "\n".join(parts)


def _tool_call_name_and_args(tool_call: Any) -> Tuple[str, Mapping[str, Any]]:
    """``(name, arguments_dict)`` from a tool_call; ``arguments`` may be a JSON string or dict ({} if unparsable)."""
    fn = tool_call.get("function") if isinstance(tool_call, Mapping) else None
    if not isinstance(fn, Mapping):
        return "", {}
    name = str(fn.get("name") or "")
    raw_args = fn.get("arguments")
    if isinstance(raw_args, str) and raw_args:
        try:
            raw_args = json.loads(raw_args)
        except Exception:
            return name, {}
    return name, raw_args if isinstance(raw_args, Mapping) else {}


def _iter_assistant_tool_calls(messages: Sequence[Mapping[str, Any]]) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    for msg in messages:
        if not isinstance(msg, Mapping) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls if isinstance(tool_calls, list) else ():
            name, args = _tool_call_name_and_args(tc)
            if name:
                yield name, args


def _count_visible_turns(messages: Sequence[Mapping[str, Any]]) -> Tuple[int, int, int]:
    """Return ``(user_turn_count, assistant_turn_count, tool_message_count)``."""
    roles = Counter(msg.get("role") for msg in messages if isinstance(msg, Mapping))
    return roles["user"], roles["assistant"], roles["tool"]


def _latest_text(messages: Sequence[Mapping[str, Any]], role: str) -> Optional[str]:
    """Most recent non-empty ``content`` text for *role*, or None."""
    for msg in reversed(messages):
        if isinstance(msg, Mapping) and msg.get("role") == role:
            text = _coerce_text(msg.get("content")).strip()
            if text:
                return text
    return None


def _recent_window(
    messages: Sequence[Mapping[str, Any]], window: int = _RECENT_TURN_WINDOW
) -> List[Mapping[str, Any]]:
    """Tail slice covering at most ``window`` user+assistant turns (tool messages ride along)."""
    count = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], Mapping) and messages[i].get("role") in {"user", "assistant"}:
            count += 1
            if count >= window:
                return list(messages[i:])
    return list(messages)


def _shortened_path(path: str) -> str:
    """Show a path relative to cwd when possible, otherwise with ~ expansion."""
    if not path:
        return path
    try:
        abs_path = os.path.abspath(os.path.expanduser(path))
        cwd, home = os.getcwd(), os.path.expanduser("~")
        if abs_path == cwd:
            return "."
        if abs_path.startswith(cwd + os.sep):
            return abs_path[len(cwd) + 1 :]
        if abs_path.startswith(home + os.sep):
            return "~/" + abs_path[len(home) + 1 :]
        return abs_path
    except Exception:
        return path


def _summarise_tool_activity(
    tool_calls: Sequence[Tuple[str, Mapping[str, Any]]],
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """``(tool_counts_sorted_desc, recently_edited_files)`` — files are distinct paths, newest first."""
    counter: Counter[str] = Counter()
    files_seen: dict[str, str] = {}  # raw path -> shortened, insertion ordered
    for name, args in reversed(list(tool_calls)):  # reversed so files_seen comes out newest -> oldest
        counter[name] += 1
        path = args.get(_FILE_EDIT_TOOLS[name]) if name in _FILE_EDIT_TOOLS else None
        if isinstance(path, str) and path and path not in files_seen:
            files_seen[path] = _shortened_path(path)
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])), list(files_seen.values())


def _join_capped(items: List[str], limit: int) -> str:
    """``a, b, c (+N more)`` — comma-join the first *limit* items and count the rest."""
    text = ", ".join(items[:limit])
    extra = len(items) - limit
    return f"{text} (+{extra} more)" if extra > 0 else text


def _truncate(text: str, limit: int) -> str:
    # Stored history is untrusted for display: strip escapes/control chars so a recap line can't clear
    # the screen or retitle the window when echoed to a terminal.
    text = " ".join(sanitize_display_text(text).split())  # collapse newlines for a compact one-liner
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_recap(
    messages: Sequence[Mapping[str, Any]], *, session_title: Optional[str] = None, session_id: Optional[str] = None,
    platform: Optional[str] = None,
) -> str:
    """Multi-line plain-text recap of recent activity (80-col terminal / gateway bubble friendly).
    ``platform`` is accepted for forward compat and does not change behavior."""
    lines: List[str] = ["Session recap"]
    if session_title or session_id:
        lines[0] += f" — {session_title or session_id[:8]}"
    if not messages:
        lines.append("  (nothing to recap — no messages yet)")
        return "\n".join(lines)
    users, assistants, tool_msgs = _count_visible_turns(messages)
    window = _recent_window(messages)
    win_users, win_assistants, _ = _count_visible_turns(window)
    scope = (
        f"{win_users} user turn{'s' if win_users != 1 else ''} / "
        f"{win_assistants} assistant repl{'ies' if win_assistants != 1 else 'y'}"
    )
    if (users, assistants) != (win_users, win_assistants):
        scope += f" (of {users}/{assistants} total)"
    lines.append(f"  Recent: {scope}, {tool_msgs} tool result{'s' if tool_msgs != 1 else ''}")
    tool_calls = list(_iter_assistant_tool_calls(window))
    tool_counts, files = _summarise_tool_activity(tool_calls)
    if tool_counts:
        top = _join_capped([f"{name}×{count}" for name, count in tool_counts], 5)
        lines.append(f"  Tools used: {top}")
    if files:
        lines.append(f"  Files touched: {_join_capped(files, _MAX_FILES_LISTED)}")
    previews = (("user", "Last ask", _PROMPT_PREVIEW_CHARS), ("assistant", "Last reply", _ASSISTANT_PREVIEW_CHARS))
    for role, label, limit in previews:
        latest = _latest_text(window, role)
        if latest:
            lines.append(f"  {label}: {_truncate(latest, limit)}")
    if len(lines) == 2:  # only header + scope line: nothing substantive to show
        lines.append("  (no assistant activity yet in this window)")
    return "\n".join(lines)


__all__ = ["build_recap"]
