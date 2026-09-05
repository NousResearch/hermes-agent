"""CLI presentation -- spinner, kawaii faces, tool preview formatting.

Pure display functions with no AIAgent dependency; used for CLI feedback.
"""

import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

from utils import safe_json_loads
from agent.redact import redact_sensitive_text
from agent.tool_result_classification import (
    file_mutation_result_landed,
    is_skill_view_dedup_result,
)

logger = logging.getLogger(__name__)

_ANSI_RESET = "\033[0m"
_MAX_INLINE_DIFF_FILES = 6
_MAX_INLINE_DIFF_LINES = 80


def _display_url(value: Any) -> str:
    """Extract a display-only URL without assuming model argument types."""
    value = (value.get("url") or value.get("href")) if isinstance(value, dict) else value
    return value.strip() if isinstance(value, str) else ""


def _http_url(candidate: str) -> str | None:
    """``candidate`` when it parses as an absolute http(s) URL, else None."""
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    return candidate if parsed.scheme.lower() in {"http", "https"} and parsed.netloc else None


def _hex_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def _get_skin():
    """Active skin config, or None when unavailable (lazy import avoids cycles)."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        return get_active_skin()
    except Exception:
        return None


# Diff colors resolve lazily from the skin engine (light/dark aware) and are
# cached after the first resolution.
_diff_colors_cached: dict[str, str] | None = None


# Foreground diff colors: key -> (skin color key, dark-terminal fallback RGB).
_DIFF_FG = {
    "dim": ("banner_dim", (150, 150, 150)), "file": ("session_label", (180, 160, 255)),
    "hunk": ("session_border", (120, 120, 140)),
}
# Background diff colors (white text on a dark tint of the skin's ui_error/ui_ok):
# key -> (skin key, default hex, dominant channel index, dark-terminal fallback).
_DIFF_BG = {
    "minus": ("ui_error", "#ef5350", 0, "\033[38;2;255;255;255;48;2;120;20;20m"),
    "plus": ("ui_ok", "#4caf50", 1, "\033[38;2;255;255;255;48;2;20;90;20m"),
}


def _fg(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def _tinted_bg(rgb: tuple[int, int, int], dominant: int) -> str:
    r, g, b = (max(v // 2, 20) if i == dominant else max(v // 4, 10) for i, v in enumerate(rgb))
    return f"\033[38;2;255;255;255;48;2;{r};{g};{b}m"


def _diff_ansi() -> dict[str, str]:
    """Return ANSI escapes for diff display, resolved from the active skin."""
    global _diff_colors_cached
    if _diff_colors_cached is not None:
        return _diff_colors_cached
    colors = {k: _fg(*rgb) for k, (_, rgb) in _DIFF_FG.items()} | {k: v[3] for k, v in _DIFF_BG.items()}
    try:
        skin = _get_skin()
        for key, (skin_key, fallback) in _DIFF_FG.items():
            h = skin.get_color(skin_key, "")
            colors[key] = _fg(*(_hex_rgb(h) if h and len(h) == 7 and h[0] == "#" else fallback))
        bg_hex = {key: skin.get_color(skin_key, default) for key, (skin_key, default, _, _) in _DIFF_BG.items()}
        for key, (_, _, dominant, _) in _DIFF_BG.items():
            h = bg_hex[key]
            if h and len(h) == 7:
                colors[key] = _tinted_bg(_hex_rgb(h), dominant)
    except Exception:
        pass
    _diff_colors_cached = colors
    return colors


@dataclass
class LocalEditSnapshot:
    """Pre-tool filesystem snapshot used to render diffs locally after writes."""
    paths: list[Path] = field(default_factory=list)
    before: dict[str, str | None] = field(default_factory=dict)


# Configurable tool preview length; set once at startup from display.tool_preview_length.
_tool_preview_max_len: int = 0  # 0 = unlimited
_friendly_tool_labels: bool = True


def set_tool_preview_max_len(n: int) -> None:
    """Set the global max length for tool call previews. 0 = no limit."""
    global _tool_preview_max_len
    _tool_preview_max_len = max(int(n), 0) if n else 0


def get_tool_preview_max_len() -> int:
    """Return the configured max preview length (0 = unlimited)."""
    return _tool_preview_max_len


def set_friendly_tool_labels(enabled: bool) -> None:
    """Toggle friendly human-phrased tool labels (display.friendly_tool_labels)."""
    global _friendly_tool_labels
    _friendly_tool_labels = bool(enabled)


def get_skin_tool_prefix() -> str:
    """Get tool output prefix character from active skin."""
    skin = _get_skin()
    return skin.tool_prefix if skin else "┊"


def get_tool_emoji(tool_name: str, default: str = "⚡") -> str:
    """Display emoji for a tool: skin ``tool_emojis`` override, then registry, then *default*."""
    skin = _get_skin()
    override = skin.tool_emojis.get(tool_name) if skin and skin.tool_emojis else None
    if override:
        return override
    try:
        from tools.registry import registry
        return registry.get_emoji(tool_name, default="") or default
    except Exception:
        return default


# ── Tool preview (one-line summary of a tool call's primary argument) ─────

def _oneline(text: str) -> str:
    """Collapse whitespace (including newlines) to single spaces."""
    return " ".join(text.split())


def _tail_trunc(text: str, limit: int) -> str:
    """Tail-truncate to ``limit`` chars with ``...`` (0 = unlimited; no guard for limit <= 3)."""
    return text[:limit - 3] + "..." if limit > 0 and len(text) > limit else text


def _truncate_preview(text: str, max_len: int | None) -> str:
    if max_len and max_len > 0 and len(text) > max_len:
        return "." * max_len if max_len <= 3 else text[:max_len - 3] + "..."
    return text


def _clip(text: str, n: int) -> str:
    """``text[:n]`` plus ``...`` when longer (used inside quoted previews)."""
    return f"{text[:n]}{'...' if len(text) > n else ''}"


@dataclass(frozen=True)
class ToolPreview:
    """A compact tool preview plus presentation facts lost to truncation."""

    text: str
    truncated: bool = False
    url: str | None = None


# ── Shell command summarisation ──────────────────────────────────────────

_SHELL_SILENT_HEADS = {"cd", "pushd", "popd", "export", "set", "unset", "source", ".", "true", "false", ":"}
_SHELL_PIPE_TAIL_HEADS = {"head", "tail", "wc", "sort", "uniq"}


def _shell_basename(head: str) -> str:
    return head.rsplit("/", 1)[-1] if head else ""


def _scan_quoted(text: str) -> Iterator[tuple[int, str, bool]]:
    """Yield ``(index, char, inside_quotes)``; a quote closes unless backslash-escaped."""
    quote: str | None = None
    for i, ch in enumerate(text):
        if quote:
            yield i, ch, True
            if ch == quote and (i == 0 or text[i - 1] != "\\"):
                quote = None
        elif ch in {"'", '"'}:
            quote = ch
            yield i, ch, True
        else:
            yield i, ch, False


def _split_shell_words(segment: str) -> list[str]:
    parts: list[list[str]] = [[]]
    for _, ch, quoted in _scan_quoted(segment):
        if not quoted and ch.isspace():
            parts.append([])
        else:
            parts[-1].append(ch)
    return ["".join(p) for p in parts if p]


def _strip_shell_pipe_tail(segment: str) -> str:
    """Drop a trailing ``| head/tail/wc/sort/uniq ...`` pipeline stage."""
    words = _split_shell_words(segment)
    for i, word in enumerate(words):
        if word == "|" and _shell_basename(words[i + 1] if i + 1 < len(words) else "") in _SHELL_PIPE_TAIL_HEADS:
            words = words[:i]
            break
    return " ".join(words).strip()


def _split_shell_compound(command: str) -> list[str]:
    """Split on unquoted ``&&`` / ``||`` / ``;`` / newline, dropping pipe tails per segment."""
    raw: list[list[str]] = [[]]
    skip = False
    for i, ch, quoted in _scan_quoted(command):
        if skip:
            skip = False
        elif not quoted and (command.startswith("&&", i) or command.startswith("||", i)):
            raw.append([])
            skip = True
        elif not quoted and ch in {";", "\n"}:
            raw.append([])
        else:
            raw[-1].append(ch)
    segments = (_strip_shell_pipe_tail("".join(buf).strip()) for buf in raw)
    return [s for s in segments if s]


def _shell_head_word(segment: str) -> str:
    """Command name of a segment, skipping leading ``VAR=value`` assignments."""
    words = _split_shell_words(segment)
    while words and re.match(r"^[A-Za-z_]\w*=", words[0]):
        words.pop(0)
    return _shell_basename(words[0] if words else "")


def _clean_shell_segment(segment: str) -> str:
    """Drop redirections (``> file``, ``2>&1``) from a segment."""
    words = _split_shell_words(segment)
    out: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        if re.match(r"^\d*(?:>>?|<)$", word):
            i += 2  # operator + target
        elif re.match(r"^\d*(?:>&|<&)\d+$", word):
            i += 1
        else:
            out.append(word)
            i += 1
    return " ".join(out).strip()


def _is_shell_boundary_echo(segment: str) -> bool:
    words = _split_shell_words(segment)
    if _shell_basename(words[0] if words else "") != "echo":
        return False
    return bool(re.search(r"-{2,}|_exit=|(?:^|\s|=)\$[?{]|PIPESTATUS", " ".join(words[1:])))


def summarize_shell_command(command: str) -> str:
    """Compact shell wrapper/plumbing for display while preserving raw command elsewhere."""
    original = _oneline(command)
    if not original:
        return ""
    segments = _split_shell_compound(original)
    if len(segments) <= 1:
        return _clean_shell_segment(segments[0] if segments else original) or original
    core: list[str] = []
    for segment in segments:
        cleaned = _clean_shell_segment(segment)
        if cleaned and _shell_head_word(cleaned) not in _SHELL_SILENT_HEADS and not _is_shell_boundary_echo(cleaned):
            core.append(cleaned)
    if not core:
        return original
    count = len(core) - 1
    return core[0] if not count else f"{core[0]} + {count} {'command' if count == 1 else 'commands'}"


def _read_file_line_label(args: dict) -> str:
    offset, limit = args.get("offset"), args.get("limit")
    if not isinstance(offset, int) or offset <= 0:
        return ""
    return f"L{offset}-{offset + limit - 1}" if isinstance(limit, int) and limit > 1 else f"L{offset}"


def redact_browser_typed_text_for_display(value: Any, typed_text: Any) -> Any:
    """Replace every occurrence of a secret-looking browser_type value with its redacted form.

    Backends echo the attempted input in error strings/metadata, so it is swapped before
    reaching logs, callbacks, the model, or chat history. Forced regardless of
    ``security.redact_secrets``: a leaked typed credential is a security boundary.
    """
    needle = "" if typed_text is None else str(typed_text)
    redacted = redact_sensitive_text(needle, force=True) if needle else needle
    if redacted == needle:
        return value
    if isinstance(value, str):
        return value.replace(needle, redacted)
    if isinstance(value, dict):
        return {key: redact_browser_typed_text_for_display(item, typed_text) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_browser_typed_text_for_display(item, typed_text) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_browser_typed_text_for_display(item, typed_text) for item in value)
    return value


def redact_tool_args_for_display(tool_name: str, args: dict | None) -> dict | None:
    """Return a copy of tool args safe for logs/progress UI (masks ``browser_type`` secrets)."""
    if not isinstance(args, dict):
        return args
    if tool_name == "browser_type" and isinstance(args.get("text"), str):
        return {**args, "text": redact_sensitive_text(args["text"], force=True)}
    return args


def _delegate_task_goals(tasks: Any, *, per_goal_len: int) -> list[str]:
    """One truncated goal string per dict task (``?`` when missing)."""
    if not isinstance(tasks, list):
        return []
    raw_goals = (task.get("goal") for task in tasks if isinstance(task, dict))
    return [_truncate_preview(("?" if g is None else _oneline(str(g))) or "?", per_goal_len) for g in raw_goals]


def _browser_exec_step_label(args: dict, max_chars: int = 80) -> str | None:
    """User-friendly step label from browser_exec code's leading comment."""
    first = str(args.get("code", "") or "").strip().split("\n", 1)[0].strip()
    label = first.lstrip("#").strip() if first.startswith("#") else ""
    if not label:
        return None
    return label[: max_chars - 1] + "…" if len(label) > max_chars else label


_PRIMARY_ARGS = {
    "terminal": "command", "web_search": "query", "web_extract": "urls", "read_file": "path",
    "write_file": "path", "patch": "path", "search_files": "pattern", "browser_navigate": "url",
    "browser_click": "ref", "browser_type": "text", "image_generate": "prompt", "text_to_speech": "text",
    "vision_analyze": "question", "skill_view": "name", "skills_list": "category", "cronjob_manage": "action",
    "execute_code": "code", "browser_exec": "code", "delegate_task": "goal", "clarify": "question",
    "skill_manage": "name",
}
_FALLBACK_PREVIEW_KEYS = ("query", "text", "command", "path", "name", "prompt", "code", "goal")


def _delegate_action_preview(args: dict) -> str | None:
    """Shared ``list/steer/stop <id>`` preview for delegate_task, or None for spawn calls."""
    action = str(args.get("action") or "").strip().lower()
    if action in ("list", "steer", "stop"):
        return f"{action} {str(args.get('subagent_id') or '').strip()}".strip()
    return None


def _preview_browser_exec(args: dict, max_len: int) -> str | None:
    label = _browser_exec_step_label(args)
    if label is not None:
        return _truncate_preview(label, max_len)
    return _truncate_preview(_oneline(str(args.get("code", "") or "")), max_len) or None


def _preview_delegate_task(args: dict, max_len: int) -> str | None:
    action_preview = _delegate_action_preview(args)
    tasks = args.get("tasks")
    if action_preview is not None:
        return _truncate_preview(action_preview, max_len)
    if tasks and isinstance(tasks, list):
        goals = _delegate_task_goals(tasks, per_goal_len=40)
        preview = f"{len(goals)} tasks: " + " | ".join(goals) if goals else f"{len(tasks)} parallel tasks"
        return _truncate_preview(preview, max_len)
    goal = args.get("goal", "")
    return None if goal is None else _truncate_preview(_oneline(str(goal)), max_len) or None


def _preview_process_manage(args: dict, _max_len: int) -> str | None:
    action, sid, data, timeout_val = (args.get(k) for k in ("action", "session_id", "data", "timeout"))
    parts = [str(action) if action else "", str(sid)[:16] if sid else "",
             f'"{_oneline(str(data)[:20])}"' if data else "", f"{timeout_val}s" if timeout_val and action == "wait" else ""]
    return " ".join(p for p in parts if p) or None


def _preview_todo_list(args: dict, _max_len: int) -> str:
    todos_arg = args.get("todos")
    verb = "updating" if args.get("merge", False) else "planning"
    return "reading task list" if todos_arg is None else f"{verb} {len(todos_arg)} task(s)"


def _preview_shell(key: str):
    def _build(args: dict, max_len: int) -> str | None:
        command = args.get(key)
        return None if command is None else _truncate_preview(summarize_shell_command(str(command)), max_len) or None
    return _build


def _preview_read_file(args: dict, max_len: int) -> str | None:
    path = args.get("path") or args.get("file") or args.get("filepath")
    label = (Path(str(path).replace("\\", "/")).name or str(path)) if path is not None else None
    return None if label is None else _truncate_preview(f"{label} {_read_file_line_label(args)}".strip(), max_len) or None


def _preview_memory(args: dict, _max_len: int) -> str:
    action, target = args.get("action", ""), args.get("target", "")
    if action == "add":
        return f"+{target}: \"{_clip(_oneline(args.get('content', '')), 25)}\""
    if action in ("replace", "remove"):
        old = _oneline(args.get("old_text") or "") or "<missing old_text>"
        return f"{'~' if action == 'replace' else '-'}{target}: \"{old[:20]}\""
    return action


def _preview_send_message(args: dict, _max_len: int) -> str:
    return f"to {args.get('target', '?')}: \"{_tail_trunc(_oneline(args.get('message', '')), 20)}\""


def _preview_skill_view(args: dict, max_len: int) -> str | None:
    name = _oneline(str(args.get("name") or ""))
    file_path = args.get("file_path")
    label = (f"{name} → {_oneline(str(file_path))}" if name else _oneline(str(file_path))) if file_path else name
    return _truncate_preview(label, max_len) or None


# Tool-specific preview builders: f(args, max_len) -> preview. Tools not listed
# fall through to the primary-argument lookup in build_tool_preview.
_PREVIEW_BUILDERS = {
    "browser_exec": _preview_browser_exec, "delegate_task": _preview_delegate_task,
    "process_manage": _preview_process_manage, "todo_list": _preview_todo_list,
    "terminal": _preview_shell("command"), "execute_code": _preview_shell("code"),
    "read_file": _preview_read_file, "memory": _preview_memory, "send_message": _preview_send_message,
    "skill_view": _preview_skill_view,
    "session_search": lambda args, _m: f"recall: \"{_clip(_oneline(args.get('query', '')), 25)}\"",
}


def build_tool_preview(tool_name: str, args: dict, max_len: int | None = None) -> str | None:
    """Build a short preview of a tool call's primary argument for display.

    *max_len* ``None`` defers to the global ``_tool_preview_max_len``; ``0`` means unlimited.
    """
    if max_len is None:
        max_len = _tool_preview_max_len
    if not args:
        return None
    args = redact_tool_args_for_display(tool_name, args) or args
    builder = _PREVIEW_BUILDERS.get(tool_name)
    if builder is not None:
        return builder(args, max_len)
    key = _PRIMARY_ARGS.get(tool_name) or next((k for k in _FALLBACK_PREVIEW_KEYS if k in args), None)
    if not key or key not in args:
        return None
    value = args[key]
    preview = _oneline(str((value[0] if value else "") if isinstance(value, list) else value))
    return _tail_trunc(preview, max_len) if preview else None


def prepare_tool_preview(tool_name: str, args: dict | None, *, fallback: str, max_len: int) -> ToolPreview:
    """Compact preview plus explicit truncation/URL facts (the uncapped preview is
    rebuilt from the arguments so an upstream display cap cannot drop its link target)."""
    full_text = build_tool_preview(tool_name, args, max_len=0) or fallback
    text = _truncate_preview(full_text, max_len)
    truncated = text != full_text
    url = _http_url(_display_url(full_text)) if truncated else None
    return ToolPreview(text=text, truncated=truncated, url=url)


# ── Friendly tool labels: "web_search <q>" -> "Searching the web for <q>" ──
# Curated built-ins only — we know each core tool's semantics so the verb is fixed,
# not computed; custom/plugin/MCP tools have no entry and fall back to the raw preview.

_TOOL_VERBS: dict[str, str] = {
    "web_search": "Searching the web", "web_extract": "Reading",
    "browser_navigate": "Browsing", "browser_click": "Clicking", "browser_type": "Typing",
    "read_file": "Reading", "write_file": "Writing", "patch": "Editing", "search_files": "Searching files",
    "terminal": "Running", "execute_code": "Running code",
    "image_generate": "Generating image", "video_generate": "Generating video",
    "text_to_speech": "Generating speech", "vision_analyze": "Looking at the image",
    "session_search": "Searching past sessions",
    "skill_view": "Reading skill", "skills_list": "Listing skills", "skill_manage": "Updating skill",
    "delegate_task": "Delegating", "cronjob_manage": "Scheduling", "clarify": "Asking",
    "memory": "Updating memory", "todo_list": "Updating tasks",
}
# Verbs that read better without the argument preview appended.
_TOOL_VERBS_NO_PREVIEW: frozenset[str] = frozenset({"skills_list", "session_search"})
# Verbs joined to the preview with " for " (search-style phrasing).
_TOOL_VERBS_FOR_CONNECTOR: frozenset[str] = frozenset({"web_search", "search_files"})

def get_tool_verb(tool_name: str) -> str | None:
    """Friendly verb for a built-in tool, or None (labels disabled / no curated verb);
    callers compose ``f"{verb}{tool_verb_connector(tool)}{preview}"`` themselves."""
    return _TOOL_VERBS.get(tool_name) if _friendly_tool_labels else None


def tool_verb_connector(tool_name: str) -> str:
    """Return the connector between a verb and its preview (" for " or " ")."""
    return " for " if tool_name in _TOOL_VERBS_FOR_CONNECTOR else " "


def verb_drops_preview(tool_name: str) -> bool:
    """Whether the verb should render alone, without the argument preview."""
    return tool_name in _TOOL_VERBS_NO_PREVIEW


def build_status_phrase(tool_name: str, args: dict | None, max_len: int = 49) -> str | None:
    """Lowercase "is <verb> <preview>…" phrase following the bot's display name (Slack setStatus).

    ``args=None`` gives a verb-only phrase (``display.live_status: verb`` keeps previews out
    of shared channels). None for ``_thinking`` or disabled labels so callers use their
    static default. Default ``max_len`` stays under Slack's ~50-char status truncation.
    """
    if not tool_name or tool_name == "_thinking" or not _friendly_tool_labels:
        return None
    verb = _TOOL_VERBS.get(tool_name)
    phrase = f"is {verb[0].lower()}{verb[1:]}" if verb else f"is using {tool_name}"
    with_preview = args and verb and tool_name not in _TOOL_VERBS_NO_PREVIEW
    preview = build_tool_preview(tool_name, args, max_len=None) if with_preview else None
    if preview:  # previews can contain newlines (terminal commands); keep the first line
        phrase = f"{phrase}{tool_verb_connector(tool_name)}{preview.splitlines()[0].strip()}"
    return phrase[: max_len - 2].rstrip() + "…" if len(phrase) > max_len - 1 else phrase + "…"


def build_tool_label(tool_name: str, args: dict, max_len: int | None = None) -> str | None:
    """Human-phrased label ("Searching the web for ...") for curated built-ins; other
    tools (or labels disabled) get the raw preview, so it is a drop-in for build_tool_preview."""
    verb = get_tool_verb(tool_name)
    if verb and tool_name in _TOOL_VERBS_NO_PREVIEW:
        return verb
    preview = build_tool_preview(tool_name, args, max_len=max_len)
    if not verb:
        return preview
    return f"{verb}{tool_verb_connector(tool_name)}{preview}" if preview else verb


# ── Inline diff previews for write actions ────────────────────────────────

def _resolved_path(path: str) -> Path:
    """Resolve a possibly-relative filesystem path against the current cwd."""
    return Path.cwd() / Path(os.path.expanduser(path))  # a `/` with an absolute rhs keeps the rhs


def _snapshot_text(path: Path) -> str | None:
    """Return UTF-8 file content, or None for missing/unreadable files."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):  # FileNotFoundError/IsADirectoryError are OSErrors
        return None


def _display_diff_path(path: Path) -> str:
    """Prefer cwd-relative paths in diffs when available."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def _resolve_skill_manage_paths(args: dict) -> list[Path]:
    """Resolve skill_manage write targets to filesystem paths."""
    action = args.get("action")
    name = args.get("name")
    if not action or not name:
        return []
    from tools.skill_manager_tool import _find_skill, _resolve_skill_dir

    if action == "create":
        return [_resolve_skill_dir(name, args.get("category")) / "SKILL.md"]
    existing = _find_skill(name)
    if not existing:
        return []
    skill_dir = Path(existing["path"])
    file_path = args.get("file_path")
    if action == "delete":
        return [path for path in sorted(skill_dir.rglob("*")) if path.is_file()]
    if file_path and action in {"edit", "patch", "write_file", "remove_file"}:
        return [skill_dir / file_path]
    return [skill_dir / "SKILL.md"] if action in {"edit", "patch"} else []


def _resolve_local_edit_paths(tool_name: str, function_args: dict | None) -> list[Path]:
    """Resolve local filesystem targets for write-capable tools."""
    if not isinstance(function_args, dict):
        return []
    if tool_name == "skill_manage":
        return _resolve_skill_manage_paths(function_args)
    path = function_args.get("path") if tool_name in {"write_file", "patch"} else None
    return [_resolved_path(path)] if path else []


def capture_local_edit_snapshot(tool_name: str, function_args: dict | None) -> LocalEditSnapshot | None:
    """Capture before-state for local write previews."""
    paths = _resolve_local_edit_paths(tool_name, function_args)
    if not paths:
        return None
    return LocalEditSnapshot(paths=paths, before={str(path): _snapshot_text(path) for path in paths})


def _result_succeeded(result: str | None) -> bool:
    """Conservatively detect whether a tool result represents success."""
    data = safe_json_loads(result) if result else None
    if not isinstance(data, dict) or data.get("error"):
        return False
    return bool(data.get("success")) if "success" in data else True


def _diff_from_snapshot(snapshot: LocalEditSnapshot | None) -> str | None:
    """Generate unified diff text from a stored before-state and current files."""
    if not snapshot:
        return None
    chunks: list[str] = []
    for path in snapshot.paths:
        before, after = snapshot.before.get(str(path)), _snapshot_text(path)
        if before == after:
            continue
        display_path = _display_diff_path(path)
        diff = "".join(unified_diff(
            (before or "").splitlines(keepends=True), (after or "").splitlines(keepends=True),
            fromfile=f"a/{display_path}", tofile=f"b/{display_path}",
        ))
        if diff:
            chunks.append(diff if diff.endswith("\n") else diff + "\n")
    return "".join(chunks) or None


def extract_edit_diff(
    tool_name: str, result: str | None, *,
    function_args: dict | None = None, snapshot: LocalEditSnapshot | None = None,
) -> str | None:
    """Extract a unified diff from a file-edit tool result."""
    if tool_name == "patch" and result:
        data = safe_json_loads(result)
        diff = data.get("diff") if isinstance(data, dict) else None
        if isinstance(diff, str) and diff.strip():
            return diff
    if tool_name not in {"write_file", "patch", "skill_manage"} or not _result_succeeded(result):
        return None
    return _diff_from_snapshot(snapshot)


def _emit_inline_diff(diff_text: str, print_fn) -> bool:
    """Emit rendered diff text through the CLI's prompt_toolkit-safe printer."""
    if print_fn is None or not diff_text:
        return False
    try:
        for line in ["  ┊ review diff", *diff_text.rstrip("\n").splitlines()]:
            print_fn(line)
        return True
    except Exception:
        return False


# Unified-diff line prefix -> diff color key (checked in order; "--- "/"+++ " handled first).
_DIFF_LINE_COLORS = (("@@", "hunk"), ("-", "minus"), ("+", "plus"), (" ", "dim"))


def _render_inline_unified_diff(diff: str) -> list[str]:
    """Render unified diff lines in Hermes' inline transcript style."""
    rendered: list[str] = []
    from_file = to_file = None
    for raw_line in diff.splitlines():
        if raw_line.startswith("--- "):
            from_file = raw_line[4:].strip()
            continue
        if raw_line.startswith("+++ "):
            to_file = raw_line[4:].strip()
            if from_file or to_file:
                rendered.append(f"{_diff_ansi()['file']}{from_file or 'a/?'} → {to_file or 'b/?'}{_ANSI_RESET}")
            continue
        color = next((c for prefix, c in _DIFF_LINE_COLORS if raw_line.startswith(prefix)), None)
        if color:
            rendered.append(f"{_diff_ansi()[color]}{raw_line}{_ANSI_RESET}")
        elif raw_line:
            rendered.append(raw_line)
    return rendered


def _split_unified_diff_sections(diff: str) -> list[str]:
    """Split a unified diff into per-file sections."""
    sections: list[list[str]] = [[]]
    for line in diff.splitlines():
        if line.startswith("--- ") and sections[-1]:
            sections.append([])
        sections[-1].append(line)
    return ["\n".join(section) for section in sections if section]


def _summarize_rendered_diff_sections(
    diff: str, *, max_files: int = _MAX_INLINE_DIFF_FILES, max_lines: int = _MAX_INLINE_DIFF_LINES,
) -> list[str]:
    """Render diff sections while capping file count and total line count."""
    sections = _split_unified_diff_sections(diff)
    rendered: list[str] = []
    omitted_files = omitted_lines = 0
    for idx, section in enumerate(sections):
        section_lines = _render_inline_unified_diff(section)
        remaining_budget = max_lines - len(rendered)
        if idx >= max_files or remaining_budget <= 0:
            omitted_files += 1
            omitted_lines += len(section_lines)
            continue
        if len(section_lines) <= remaining_budget:
            rendered.extend(section_lines)
            continue
        rendered.extend(section_lines[:remaining_budget])
        omitted_lines += len(section_lines) - remaining_budget
        omitted_files += 1 + max(0, len(sections) - idx - 1)
        for leftover in sections[idx + 1:]:
            omitted_lines += len(_render_inline_unified_diff(leftover))
        break
    if omitted_files or omitted_lines:
        summary = f"… omitted {omitted_lines} diff line(s)"
        if omitted_files:
            summary += f" across {omitted_files} additional file(s)/section(s)"
        rendered.append(f"{_diff_ansi()['hunk']}{summary}{_ANSI_RESET}")
    return rendered


def render_edit_diff_with_delta(
    tool_name: str, result: str | None, *,
    function_args: dict | None = None, snapshot: LocalEditSnapshot | None = None, print_fn=None,
) -> bool:
    """Render an edit diff inline without taking over the terminal UI."""
    diff = extract_edit_diff(tool_name, result, function_args=function_args, snapshot=snapshot)
    if not diff:
        return False
    try:
        rendered_lines = _summarize_rendered_diff_sections(diff)
    except Exception as exc:
        logger.debug("Could not render inline diff: %s", exc)
        return False
    return _emit_inline_diff("\n".join(rendered_lines), print_fn)


# ── KawaiiSpinner ─────────────────────────────────────────────────────────

class KawaiiSpinner:
    """Animated spinner with kawaii faces for CLI feedback during tool execution."""

    SPINNERS = {
        'dots': list('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'), 'bounce': list('⠁⠂⠄⡀⢀⠠⠐⠈'), 'grow': list('▁▂▃▄▅▆▇█▇▆▅▄▃▂'),
        'arrows': list('←↖↑↗→↘↓↙'), 'star': list('✶✷✸✹✺✹✸✷'), 'moon': list('🌑🌒🌓🌔🌕🌖🌗🌘'),
        'pulse': list('◜◠◝◞◡◟'), 'brain': list('🧠💭💡✨💫🌟💡💭'), 'sparkle': list('⁺˚*✧✦✧*˚'),
    }
    KAWAII_WAITING = [
        "(｡◕‿◕｡)", "(◕‿◕✿)", "٩(◕‿◕｡)۶", "(✿◠‿◠)", "( ˘▽˘)っ", "♪(´ε` )", "(◕ᴗ◕✿)", "ヾ(＾∇＾)", "(≧◡≦)", "(★ω★)",
    ]
    KAWAII_THINKING = [
        "(｡•́︿•̀｡)", "(◔_◔)", "(¬‿¬)", "( •_•)>⌐■-■", "(⌐■_■)", "(´･_･`)", "◉_◉", "(°ロ°)", "( ˘⌣˘)♡", "ヽ(>∀<☆)☆",
        "٩(๑❛ᴗ❛๑)۶", "(⊙_⊙)", "(¬_¬)", "( ͡° ͜ʖ ͡°)", "ಠ_ಠ",
    ]
    THINKING_VERBS = [
        "pondering", "contemplating", "musing", "cogitating", "ruminating", "deliberating", "mulling",
        "reflecting", "processing", "reasoning", "analyzing", "computing", "synthesizing", "formulating",
        "brainstorming",
    ]

    @staticmethod
    def _skin_spinner_list(key: str, fallback: list) -> list:
        """Return the active skin's ``spinner[key]`` list, or *fallback* when absent/empty."""
        try:
            skin = _get_skin()
            return (skin.spinner.get(key, []) if skin else None) or fallback
        except Exception:
            return fallback

    get_waiting_faces = classmethod(lambda cls: cls._skin_spinner_list("waiting_faces", cls.KAWAII_WAITING))
    get_thinking_faces = classmethod(lambda cls: cls._skin_spinner_list("thinking_faces", cls.KAWAII_THINKING))
    get_thinking_verbs = classmethod(lambda cls: cls._skin_spinner_list("thinking_verbs", cls.THINKING_VERBS))

    def __init__(self, message: str = "", spinner_type: str = 'dots', print_fn=None):
        self.message = message
        self.spinner_frames = self.SPINNERS.get(spinner_type, self.SPINNERS['dots'])
        self.running = False
        self.thread = self.start_time = None
        self.frame_idx = self.last_line_len = 0
        self._print_fn = print_fn  # when set, bypasses self._out so silenced agents stay silent
        self._out = sys.stdout  # captured NOW, before any child redirect_stdout(devnull) replaces it

    def _write(self, text: str, end: str = '\n', flush: bool = False):
        """Write via print_fn when supplied, else to the stdout captured at creation."""
        if self._print_fn is not None:
            try:
                self._print_fn(text)
            except Exception:
                pass
            return
        try:
            self._out.write(text + end)
            if flush:
                self._out.flush()
        except (ValueError, OSError):
            pass

    @property
    def _is_tty(self) -> bool:
        """Check if output is a real terminal, safe against closed streams."""
        try:
            return hasattr(self._out, 'isatty') and self._out.isatty()
        except (ValueError, OSError):
            return False

    def _is_patch_stdout_proxy(self) -> bool:
        """True for prompt_toolkit's StdoutProxy: it injects newlines around each flush so the
        \\r overwrite never lands, and the CLI drives its own TUI spinner widget in that mode."""
        try:
            from prompt_toolkit.patch_stdout import StdoutProxy
            return isinstance(self._out, StdoutProxy)
        except ImportError:
            return False

    def _animate(self):
        tty = self._is_tty
        # Non-TTY (Docker, systemd, pipe): log once instead of spamming frames.
        if not tty:
            self._write(f"  [tool] {self.message}", flush=True)
        # Under patch_stdout the \r animation would overdraw the TUI status bar.
        if not tty or self._is_patch_stdout_proxy():
            while self.running:
                time.sleep(0.5 if not tty else 0.1)
            return
        skin = _get_skin()
        wings = skin.get_spinner_wings() if skin else []
        while self.running:
            if os.getenv("HERMES_SPINNER_PAUSE"):
                time.sleep(0.1)
                continue
            frame = self.spinner_frames[self.frame_idx % len(self.spinner_frames)]
            elapsed = f"({time.time() - self.start_time:.1f}s)"
            left, right = wings[self.frame_idx % len(wings)] if wings else ("", "")
            line = f"  {left} {frame} {self.message} {right} {elapsed}" if wings else f"  {frame} {self.message} {elapsed}"
            self._write(f"\r{line}{' ' * max(self.last_line_len - len(line), 0)}", end='', flush=True)
            self.last_line_len = len(line)
            self.frame_idx += 1
            time.sleep(0.12)

    def start(self):
        if self.running:
            return
        self.running, self.start_time = True, time.time()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def update_text(self, new_message: str):
        self.message = new_message

    def _clear_line_blanks(self) -> str:
        return ' ' * max(self.last_line_len + 5, 40)  # spaces, not \033[K: garbles under patch_stdout

    def print_above(self, text: str):
        """Print a line above the spinner (next tick redraws it below). Works inside
        redirect_stdout(devnull) because _write targets the stdout captured at creation."""
        self._write(f"\r{self._clear_line_blanks()}\r  {text}" if self.running else f"  {text}", flush=True)

    def stop(self, final_message: str = None):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        is_tty = self._is_tty
        if is_tty:
            self._write(f"\r{self._clear_line_blanks()}\r", end='', flush=True)
        if final_message:
            elapsed = f" ({time.time() - self.start_time:.1f}s)" if self.start_time else ""
            self._write(f"  {final_message}" if is_tty else f"  [done] {final_message}{elapsed}", flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


# ── Cute tool message (completion line that replaces the spinner) ─────────

_ERROR_SUFFIX_MAX_LEN = 48


def _trim_error(msg: str) -> str:
    """Shrink an error message for inline display (long 'File not found' paths -> filename)."""
    msg = msg.strip()
    if "File not found:" in msg:
        tail = msg.partition("File not found:")[2].strip()
        if "/" in tail:
            msg = f"File not found: {tail.rsplit('/', 1)[-1]}"
    return _tail_trunc(msg, _ERROR_SUFFIX_MAX_LEN)


def _detect_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Return ``(is_failure, suffix)`` for a tool result, e.g. ``(True, " [exit 1]")``."""
    if result is None or file_mutation_result_landed(tool_name, result):
        return False, ""
    if is_skill_view_dedup_result(tool_name, result):
        return False, ""
    data = safe_json_loads(result)

    # Terminal: non-zero exit code is the canonical failure signal.
    if tool_name == "terminal":
        exit_code = data.get("exit_code") if isinstance(data, dict) else None
        if exit_code is None or exit_code == 0:
            return False, ""
        err_msg = data.get("error")
        return True, f" [{_trim_error(str(err_msg))}]" if err_msg else f" [exit {exit_code}]"

    if isinstance(data, dict):
        failed = data.get("success") is False
        # Memory: distinguish "store full" from real errors.
        if tool_name == "memory" and failed and "exceed the limit" in data.get("error", ""):
            return True, " [full]"
        err = data.get("error") or data.get("message")
        if err and (failed or "error" in data):
            return True, f" [{_trim_error(str(err))}]"
    # Multimodal results (dicts) are successes; failures arrive as JSON-encoded strings.
    if isinstance(result, str) and (
        '"error"' in result[:500].lower() or '"failed"' in result[:500].lower() or result.startswith("Error")
    ):
        return True, " [error]"
    return False, ""


def _domain(url: str) -> str:
    return url.replace("https://", "").replace("http://", "").split("/")[0]


def _cute_trunc(s) -> str:
    """Tail-truncate to the configured preview cap (0 = unlimited)."""
    return _tail_trunc(str(s), _tool_preview_max_len)


def _cute_path(p) -> str:
    """Head-truncate a path to the configured preview cap, keeping the filename end."""
    p = str(p)
    limit = _tool_preview_max_len
    return ("..." + p[-(limit-3):]) if limit and len(p) > limit else p


def _cute_web_extract(a: dict, _r) -> str:
    urls = a.get("urls", [])
    url = _display_url(urls[0] if isinstance(urls, list) else urls) if urls else ""
    if not url:
        return "┊ 📄 fetch     pages"
    extra = f" +{len(urls)-1}" if isinstance(urls, list) and len(urls) > 1 else ""
    return f"┊ 📄 fetch     {_cute_trunc(_domain(url))}{extra}"


def _cute_todo_list(a: dict, result) -> str:
    todos_arg = a.get("todos")
    total = done = 0
    try:
        summary = (safe_json_loads(result) or {}).get("summary", {}) if result else {}
        total, done = summary.get("total", 0), summary.get("completed", 0)
    except Exception:
        pass
    if todos_arg is None:
        detail = f"{done}/{total} task(s)" if total > 0 else "reading tasks"
    elif a.get("merge", False):
        detail = f"update {done}/{total} ✓" if total > 0 and done > 0 else f"update {len(todos_arg)} task(s)"
    else:
        detail = f"{done}/{total} task(s)" if total > 0 and done > 0 else f"{len(todos_arg)} task(s)"
    return f"┊ 📋 plan      {detail}"


def _cute_memory(a: dict, _r) -> str:
    action, target = a.get("action", "?"), a.get("target", "")
    if action == "add":
        return f"┊ 🧠 memory    +{target}: \"{_cute_trunc(a.get('content', ''))}\""
    if action in ("replace", "remove"):
        old = a.get("old_text") or "<missing old_text>"
        return f"┊ 🧠 memory    {'~' if action == 'replace' else '-'}{target}: \"{_cute_trunc(old)}\""
    return f"┊ 🧠 memory    {action}"


def _cute_skill_view(a: dict, _r) -> str:
    label, file_path = a.get("name", ""), a.get("file_path")
    label = (f"{label} → {file_path}" if label else str(file_path)) if file_path else label
    return f"┊ 📚 skill     {_cute_trunc(label)}"


def _cute_cronjob(a: dict, _r) -> str:
    action = a.get("action", "?")
    if action == "create":
        skills = a.get("skills") or ([a.get("skill")] if a.get("skill") else [])
        label = a.get("name") or (skills[0] if skills else None) or a.get("prompt", "task")
        return f"┊ ⏰ cron      create {_cute_trunc(label)}"
    return "┊ ⏰ cron      listing" if action == "list" else f"┊ ⏰ cron      {action} {a.get('job_id', '')}"


def _cute_execute_code(a: dict, _r) -> str:
    code = a.get("code", "").strip()
    return f"┊ 🐍 exec      {_cute_trunc(code.split(chr(10))[0] if code else '')}"


def _cute_browser_exec(a: dict, _r) -> str:
    # Leading `# …` comment becomes the step label; code stays collapsed behind the preview cap.
    label = _browser_exec_step_label(a)
    return f"┊ 🌐 browser   {_cute_trunc(_oneline(str(a.get('code', '') or ''))) if label is None else label}"


def _cute_delegate(a: dict, _r) -> str:
    action_preview = _delegate_action_preview(a)
    tasks = a.get("tasks")
    if action_preview is not None:
        return f"┊ 🔀 delegate  {_cute_trunc(action_preview)}"
    if tasks and isinstance(tasks, list):
        goals = _delegate_task_goals(tasks, per_goal_len=30)
        return f"┊ 🔀 delegate  {len(goals) or len(tasks)}x: {_cute_trunc(' | '.join(goals) if goals else 'parallel')}"
    return f"┊ 🔀 delegate  {_cute_trunc(a.get('goal', ''))}"


def _cute_process_manage(a: dict, _r) -> str:
    action, sid = a.get("action", "?"), a.get("session_id", "")[:12]
    return f"┊ ⚙️  proc      {'ls processes' if action == 'list' else f'{action} {sid}'}"


_SCROLL_ARROWS = {"down": "↓", "up": "↑", "right": "→", "left": "←"}

# Completion-line renderers: tool -> f(args, result) -> "┊ {emoji} {verb:9} {detail}" (duration appended by caller).
_CUTE_LINES = {
    "web_search": lambda a, r: f"┊ 🔍 search    {_cute_trunc(a.get('query', ''))}",
    "web_extract": _cute_web_extract,
    "terminal": lambda a, r: f"┊ 💻 $         {_cute_trunc(build_tool_preview('terminal', a) or a.get('command', ''))}",
    "process_manage": _cute_process_manage,
    "read_file": lambda a, r: f"┊ 📖 read      {_cute_trunc(build_tool_preview('read_file', a) or a.get('path', ''))}",
    "write_file": lambda a, r: f"┊ ✍️  write     {_cute_path(a.get('path', ''))}",
    "patch": lambda a, r: f"┊ 🔧 patch     {_cute_path(a.get('path', ''))}",
    "search_files": lambda a, r: f"┊ 🔎 {'find' if a.get('target', 'content') == 'files' else 'grep':9} {_cute_trunc(a.get('pattern', ''))}",
    "browser_navigate": lambda a, r: f"┊ 🌐 navigate  {_cute_trunc(_domain(a.get('url', '')))}",
    "browser_snapshot": lambda a, r: f"┊ 📸 snapshot  {'full' if a.get('full') else 'compact'}",
    "browser_click": lambda a, r: f"┊ 👆 click     {a.get('ref', '?')}",
    "browser_type": lambda a, r: f"┊ ⌨️  type      \"{_cute_trunc(a.get('text', ''))}\"",
    "browser_scroll": lambda a, r: f"┊ {_SCROLL_ARROWS.get(a.get('direction', 'down'), '↓')}  scroll    {a.get('direction', 'down')}",
    "browser_back": lambda a, r: "┊ ◀️  back    ",
    "browser_press": lambda a, r: f"┊ ⌨️  press     {a.get('key', '?')}",
    "browser_get_images": lambda a, r: "┊ 🖼️  images    extracting",
    "browser_vision": lambda a, r: "┊ 👁️  vision    analyzing page",
    "todo_list": _cute_todo_list,
    "session_search": lambda a, r: f"┊ 🔍 recall    \"{_cute_trunc(a.get('query', ''))}\"",
    "memory": _cute_memory,
    "skills_list": lambda a, r: f"┊ 📚 skills    list {a.get('category', 'all')}",
    "skill_view": _cute_skill_view,
    "image_generate": lambda a, r: f"┊ 🎨 create    {_cute_trunc(a.get('prompt', ''))}",
    "text_to_speech": lambda a, r: f"┊ 🔊 speak     {_cute_trunc(a.get('text', ''))}",
    "vision_analyze": lambda a, r: f"┊ 👁️  vision    {_cute_trunc(a.get('question', ''))}",
    "send_message": lambda a, r: f"┊ 📨 send      {a.get('target', '?')}: \"{_cute_trunc(a.get('message', ''))}\"",
    "cronjob_manage": _cute_cronjob,
    "execute_code": _cute_execute_code,
    "browser_exec": _cute_browser_exec,
    "delegate_task": _cute_delegate,
}


def _get_cute_tool_message(tool_name: str, args: dict, duration: float, result: str | None = None) -> str:
    """Tool completion line for CLI quiet mode: ``| {emoji} {verb:9} {detail}  {duration}``, plus a
    failure suffix from :func:`_detect_tool_failure`; the leading ``┊`` becomes the skin's tool prefix."""
    args = redact_tool_args_for_display(tool_name, args) or args
    is_failure, failure_suffix = _detect_tool_failure(tool_name, result)
    render = _CUTE_LINES.get(tool_name)
    body = render(args, result) if render else f"┊ ⚡ {tool_name[:9]:9} {_cute_trunc(build_tool_preview(tool_name, args) or '')}"
    line = f"{body}  {duration:.1f}s".replace("┊", get_skin_tool_prefix(), 1)
    return f"{line}{failure_suffix}" if is_failure else line


def get_cute_tool_message(tool_name: str, args: dict, duration: float, result: str | None = None) -> str:
    """Render a completion label without letting cosmetic failures escape."""
    try:
        return _get_cute_tool_message(tool_name, args, duration, result=result)
    except Exception as exc:  # noqa: BLE001 — display must never abort a turn
        logger.debug("Tool completion label failed for %s: %s", tool_name, exc)
        safe_name = tool_name[:9] if isinstance(tool_name, str) and tool_name else "tool"
        safe_duration = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "done"
        return f"┊ ⚡ {safe_name:9} completed  {safe_duration}"


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def get_friendly_tool_labels() -> bool:
    """Return whether friendly tool labels are enabled."""
    return _friendly_tool_labels
# ---- END PLUGIN-COMPAT ----
