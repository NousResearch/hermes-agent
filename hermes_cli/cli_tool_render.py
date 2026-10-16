"""Classic-CLI transcript rows for completed tool calls: the status-glyph row, the dim ``⎿``
result preview, the grouped "explored" row for read-only calls and the collapsed reasoning line.

Pure string builders (ANSI out, no printing, no CLI state); ``cli_tool_activity_mixin`` decides
when they are printed. Nothing here changes what the model sees.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from agent.i18n import t

RST = "\033[0m"
DIM = "\033[2m"
GUTTER_FIRST = "    ⎿  "
GUTTER_REST = "       "
EXPAND_KEY = "ctrl+o"

# Tools whose completion is a "look at something" step: consecutive ones fold into one row.
# value = (part key, how the call's subject is read from its args)
_EXPLORE_TOOLS = {
    "read_file": ("read", "path"),
    "search_files": ("search", "pattern"),
    "web_search": ("search", "query"),
    "web_extract": ("fetch", "urls"),
    "skill_view": ("skill", "name"),
}
# The diff (or the todo panel) is the result for these; a text preview would repeat it.
_NO_PREVIEW = frozenset({"write_file", "patch", "skill_manage", "todo", "todo_list", "memory", "clarify",
                         "delegate_task"})
_COMMAND_TOOLS = frozenset({"terminal", "execute_code"})
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-_]")
_BOLD_HEADING_RE = re.compile(r"^\s*\*\*(.+?)\*\*")
_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+)")


def skin_hex(key: str, fallback: str) -> str:
    """``#rrggbb`` for a skin colour key (``fallback`` when unset or malformed)."""
    from hermes_cli.skin_engine import get_active_skin
    value = get_active_skin().get_color(key, fallback)
    return value if isinstance(value, str) and len(value) == 7 and value.startswith("#") else fallback


def _hex_fg(key: str, fallback: str) -> str:
    value = skin_hex(key, fallback)
    return f"\033[38;2;{int(value[1:3], 16)};{int(value[3:5], 16)};{int(value[5:7], 16)}m"


def ok_color() -> str:
    return _hex_fg("ui_ok", "#4caf50")


def error_color() -> str:
    return _hex_fg("ui_error", "#ef5350")


def clean_text(text: str) -> str:
    """Terminal-safe text: escape sequences and control characters removed, tabs expanded."""
    text = _ANSI_RE.sub("", text.replace("\r\n", "\n"))
    return "".join(c for c in text.expandtabs(4) if c == "\n" or c.isprintable())


def clip(text: str, width: int) -> str:
    from prompt_toolkit.utils import get_cwidth
    if width <= 0 or get_cwidth(text) <= width:
        return text
    out = ""
    for char in text:
        if get_cwidth(out + char) > width - 1:
            break
        out += char
    return out + "…"


def format_duration(seconds: float) -> str:
    if seconds >= 60:
        return t("cli.shared.duration_ms_compact", minutes=int(seconds // 60), seconds=f"{int(seconds % 60):02d}")
    return t("cli.shared.duration_s", seconds=f"{seconds:.1f}" if seconds < 10 else f"{seconds:.0f}")


def result_data(result: Any) -> Any:
    """The tool result as structured data when it is JSON (the gateway already decodes it)."""
    if isinstance(result, (dict, list)):
        return result
    if isinstance(result, str) and result[:1] in "{[":
        try:
            return json.loads(result)
        except ValueError:
            return None
    return None


def result_text(result: Any) -> str:
    """The full result as text, for the transcript viewer."""
    if result is None:
        return ""
    data = result_data(result)
    if isinstance(data, dict):
        body = data.get("output") if isinstance(data.get("output"), str) else data.get("content")
        if isinstance(body, str):
            error = data.get("error")
            return body + (f"\n{error}" if isinstance(error, str) and error and error not in body else "")
    if data is not None:
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
    return str(result)


def exit_code_of(result: Any) -> int | None:
    data = result_data(result)
    code = data.get("exit_code") if isinstance(data, dict) else None
    return code if isinstance(code, int) else None


# ── result preview ──────────────────────────────────────────────────────────────────────────

def _command_lines(data: dict) -> list[str]:
    out, error = data.get("output"), data.get("error")
    text = out if isinstance(out, str) and out else error if isinstance(error, str) else ""
    return str(text).splitlines()


def _read_lines(data: dict) -> list[str]:
    total = data.get("total_lines")
    if isinstance(total, int):
        return [t("cli.tool_render.read_lines_one" if total == 1 else "cli.tool_render.read_lines_other",
                  count=total)]
    return []


def _search_lines(data: dict) -> list[str]:
    total = data.get("total_count")
    rows = []
    if isinstance(total, int):
        rows.append(t("cli.tool_render.matches_one" if total == 1 else "cli.tool_render.matches_other", count=total))
    for match in data.get("matches") or []:
        if isinstance(match, dict):
            rows.append(f"{match.get('path')}:{match.get('line')}: {str(match.get('content') or '').strip()}")
    if isinstance(data.get("matches_text"), str):
        rows.extend(data["matches_text"].splitlines())
    rows.extend(str(f) for f in data.get("files") or [])
    return rows


def _web_lines(data: dict) -> list[str]:
    web = (data.get("data") or {}).get("web") if isinstance(data.get("data"), dict) else None
    return [str(item.get("title") or item.get("url")) for item in web or [] if isinstance(item, dict)]


_STRUCTURED_PREVIEW = {
    "terminal": _command_lines, "execute_code": _command_lines, "read_file": _read_lines,
    "search_files": _search_lines, "web_search": _web_lines,
}


def _generic_lines(data: Any, result: Any) -> list[str]:
    if isinstance(data, dict):
        for key in ("error", "output", "content", "result", "message", "text"):
            if isinstance(data.get(key), str) and data[key].strip():
                return data[key].splitlines()
        return []
    if data is None and isinstance(result, str):
        return result.splitlines()
    return []


def preview_source(name: str, result: Any) -> list[str]:
    """Plain lines worth previewing for ``name``'s result (may be empty)."""
    if name in _NO_PREVIEW or result is None:
        return []
    data = result_data(result)
    extractor = _STRUCTURED_PREVIEW.get(name)
    lines = extractor(data) if extractor and isinstance(data, dict) else _generic_lines(data, result)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def fold(lines: list[str], max_lines: int) -> tuple[list[str], int]:
    """First ``max_lines`` lines plus the hidden count; a single hidden line is shown instead."""
    if len(lines) <= max_lines + 1:
        return lines, 0
    return lines[:max_lines], len(lines) - max_lines


def render_preview(name: str, result: Any, *, failed: bool, max_lines: int, width: int,
                   expand_hint: bool) -> list[str]:
    """The ``⎿`` block under a completed tool row: up to ``max_lines`` dim lines (red on failure),
    ``(no output)`` for an empty command, then ``… +N lines``."""
    if max_lines <= 0 or name in _NO_PREVIEW or result is None:
        return []
    lines = preview_source(name, result)
    if not lines:
        return [f"{DIM}{GUTTER_FIRST}{t('cli.tool_render.no_output')}{RST}"] if name in _COMMAND_TOOLS else []
    shown, hidden = fold(lines, max_lines)
    color = error_color() if failed else DIM
    body = width - len(GUTTER_FIRST)
    from agent.redact import redact_sensitive_text
    rendered = [f"{color}{GUTTER_FIRST if i == 0 else GUTTER_REST}{clip(clean_text(redact_sensitive_text(line)), body)}{RST}"
                for i, line in enumerate(shown)]
    if hidden:
        hint = "cli.tool_render.more_lines_hint" if expand_hint else "cli.tool_render.more_lines"
        rendered.append(f"{DIM}{GUTTER_REST}{t(hint, count=hidden, shortcut=EXPAND_KEY)}{RST}")
    return rendered


# ── completion row ──────────────────────────────────────────────────────────────────────────

def render_tool_row(name: str, args: dict, duration: float, result: Any, *, is_error: bool = False) -> tuple[list[str], bool]:
    """``  ┊ ✓ 💻 $         ls -la  0.4s`` (green ✓ / red ✗ + red failure suffix); one row per
    bridge sub-call. Returns ``(rows, failed)``."""
    from agent.display import cute_tool_rows, get_skin_tool_prefix
    rows, suffixes, failed = cute_tool_rows(name, args or {}, result)
    failed = failed or bool(is_error)
    prefix = get_skin_tool_prefix()
    ok, bad = ok_color(), error_color()
    out = []
    for index, (row, suffix) in enumerate(zip(rows, suffixes)):
        row_failed = bool(suffix) or (failed and len(rows) == 1)
        glyph = f"{bad}✗{RST}" if row_failed else f"{ok}✓{RST}"
        body = row.removeprefix("┊ ")
        tail = f"  {DIM}{format_duration(duration)}{RST}" if index == len(rows) - 1 else ""
        out.append(f"  {prefix} {glyph} {body}{f'{bad}{suffix}{RST}' if suffix else ''}{tail}")
    return out, failed


# ── explored group ──────────────────────────────────────────────────────────────────────────

def explore_kind(name: str) -> str | None:
    entry = _EXPLORE_TOOLS.get(name)
    return entry[0] if entry else None


def _subject(name: str, args: dict) -> str:
    key = _EXPLORE_TOOLS[name][1]
    value = (args or {}).get(key)
    if isinstance(value, list):
        value = value[0] if value else ""
    text = str(value or "").strip()
    if name == "read_file":
        return text.rstrip("/").rsplit("/", 1)[-1] or text
    if name == "web_extract":
        return text.split("://", 1)[-1].split("/", 1)[0]
    return text


@dataclass
class ExploreGroup:
    """Read-only completions held back so a run of them prints as one row."""
    calls: list[tuple[str, dict, float, Any]] = field(default_factory=list)

    def add(self, name: str, args: dict, duration: float, result: Any) -> None:
        self.calls.append((name, args or {}, duration, result))

    def __len__(self) -> int:
        return len(self.calls)


def _part_items(items: list[str], limit: int = 3) -> str:
    unique = list(dict.fromkeys(i for i in items if i))
    shown = ", ".join(unique[:limit])
    return shown + (f" +{len(unique) - limit}" if len(unique) > limit else "")


def render_explore_row(group: ExploreGroup, width: int) -> str:
    """``  ┊ ✓ 🔎 explored  read a.py, b.py · searched "foo"  1.2s``."""
    from agent.display import get_skin_tool_prefix
    parts: dict[str, list[str]] = {}
    for name, args, _, _ in group.calls:
        parts.setdefault(explore_kind(name) or "read", []).append(_subject(name, args))
    detail = " · ".join(t(f"cli.tool_render.explored_{kind}", items=_part_items(items)) for kind, items in parts.items())
    duration = sum(call[2] or 0.0 for call in group.calls)
    head = f"  {get_skin_tool_prefix()} {ok_color()}✓{RST} 🔎 {t('cli.tool_render.explored'):9} "
    tail = f"  {DIM}{format_duration(duration)}{RST}"
    return head + clip(detail, max(20, width - 40)) + tail


# ── reasoning ───────────────────────────────────────────────────────────────────────────────

def reasoning_headline(text: str) -> str:
    """First bold heading (``**Planning**``), else first markdown heading, else first line."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    for pattern in (_BOLD_HEADING_RE, _MD_HEADING_RE):
        for line in lines:
            match = pattern.match(line)
            if match:
                return match.group(1).strip().strip("*").strip()
    return lines[0].strip() if lines else ""


def render_reasoning_line(text: str, seconds: float | None, width: int) -> str:
    """``  ✻ Thought for 8s · Planning the refactor`` (dim; no duration under a second)."""
    label = (t("cli.tool_render.thought_for", duration=format_duration(seconds)) if seconds and seconds >= 1
             else t("cli.tool_render.thought"))
    headline = clean_text(reasoning_headline(text)).replace("\n", " ")
    line = f"✻ {label}" + (f" · {headline}" if headline else "")
    return f"  {DIM}{clip(line, max(20, width - 4))}{RST}"
