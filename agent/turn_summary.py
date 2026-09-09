"""Per-turn accounting for the interactive CLI (display-only, pure).

:class:`TurnSummaryCollector` rides the ``tool_progress_callback`` feed (``tool.completed``
events carry the tool name and raw result) and tallies what a turn did — no agent-loop state
is threaded through. :func:`format_turn_summary` renders a tally plus wall-clock duration
into one dim line: ``⋯ 12.4s · edited 2 files +18 -3 · read 4 files · ran 3 commands``.
:func:`format_token_flow` is the spinner-side cumulative token readout (``↓ 1.2k tok``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TurnSummaryCollector",
    "TurnTally",
    "format_turn_summary",
    "format_token_flow",
    "format_elapsed",
]


SUMMARY_PREFIX = "⋯"  # terminal chrome, deliberately not an emoji
# A tool-less turn faster than this is a plain chat reply: formatter returns "".
_MIN_TOOLLESS_SECONDS = 2.0
# Max "verb + count" segments before collapsing the rest into "+N more".
_MAX_SEGMENTS = 4

# Tool name -> (verb, singular noun, plural noun), past tense. Unlisted tools (plugin/MCP)
# fall into a generic "called N tools" bucket.
_VERB_GROUPS: dict[str, tuple[str, str, str]] = {
    "write_file": ("edited", "file", "files"),
    "patch": ("edited", "file", "files"),
    "read_file": ("read", "file", "files"),
    "web_extract": ("read", "page", "pages"),
    "terminal": ("ran", "command", "commands"),
    "execute_code": ("ran", "script", "scripts"),
    "search_files": ("searched", "path", "paths"),
    "web_search": ("searched the web", "time", "times"),
    "session_search": ("searched sessions", "time", "times"),
    "browser_navigate": ("browsed", "page", "pages"),
    "skill_view": ("read", "skill", "skills"),
    "skill_manage": ("updated", "skill", "skills"),
    "skills_list": ("listed skills", "time", "times"),
    "todo_list": ("updated", "task list", "task lists"),
    "delegate_task": ("delegated", "task", "tasks"),
    "memory": ("updated", "memory", "memories"),
}

_EDIT_VERB = "edited"  # verb group that carries file-edit line deltas (+X -Y) when known
# Render order: edits first, then reads, then commands; others in first-seen order.
_VERB_PRIORITY: tuple[str, ...] = ("edited", "read", "ran")
# Tools whose results may report a unified diff we can count lines from.
_DIFF_RESULT_TOOLS = frozenset({"patch"})


@dataclass
class TurnTally:
    """What a single turn did, as observed from the tool-progress feed."""

    # verb -> {noun_plural: count}; insertion order gives stable rendering.
    verbs: dict[str, dict[str, int]] = field(default_factory=dict)
    other_tools: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    # True once an edit tool reported a countable diff ("+0 -0" vs "unknown").
    has_line_deltas: bool = False

    @property
    def total_tools(self) -> int:
        return sum(sum(nouns.values()) for nouns in self.verbs.values()) + self.other_tools


def _extract_line_deltas(tool_name: str, result: Any) -> tuple[int, int] | None:
    """(added, removed) from a tool result that already reports a diff, else None.
    Never shells out to git or re-reads files; ``+++``/``---`` headers excluded."""
    if tool_name not in _DIFF_RESULT_TOOLS:
        return None
    payload: Any = result
    if isinstance(payload, str):
        text = payload.strip()
        if not text.startswith("{"):
            return None
        try:
            # strict=False tolerates raw control chars inside an embedded diff.
            payload = json.loads(text, strict=False)
        except Exception:
            return None
    if not isinstance(payload, dict):
        return None
    diff = payload.get("diff")
    if not isinstance(diff, str) or not diff.strip():
        return None
    body = [ln for ln in diff.splitlines() if not ln.startswith(("+++", "---"))]
    added = sum(ln.startswith("+") for ln in body)
    removed = sum(ln.startswith("-") for ln in body)
    # A diff with no +/- content lines tells us nothing: unknown, not "+0 -0".
    if added == 0 and removed == 0:
        return None
    return added, removed


class TurnSummaryCollector:
    """Accumulate per-turn tool tallies from the CLI's ``_on_tool_progress`` feed."""

    def __init__(self) -> None:
        self._tally = TurnTally()

    def begin(self) -> None:
        """Start a fresh turn (drops any prior tally)."""
        self._tally = TurnTally()

    def record_tool(self, tool_name: str | None, *, result: Any = None, is_error: bool = False) -> None:
        """Record one completed tool call. Failed calls are skipped: "edited 2 files"
        when one write was denied is exactly the over-claim to avoid."""
        # Internal/pseudo tools (``_thinking``) are not user-visible work.
        if not tool_name or is_error or tool_name.startswith("_"):
            return

        group = _VERB_GROUPS.get(tool_name)
        if group is None:
            self._tally.other_tools += 1
            return

        verb, _singular, plural = group
        nouns = self._tally.verbs.setdefault(verb, {})
        nouns[plural] = nouns.get(plural, 0) + 1

        if verb == _EDIT_VERB:
            deltas = _extract_line_deltas(tool_name, result)
            if deltas is not None:
                added, removed = deltas
                self._tally.lines_added += added
                self._tally.lines_removed += removed
                self._tally.has_line_deltas = True

    @property
    def tally(self) -> TurnTally:
        return self._tally

    def render(self, elapsed_seconds: float) -> str:
        return format_turn_summary(elapsed_seconds, self._tally)


def format_elapsed(seconds: float) -> str:
    """``12.4s`` / ``2m05s``."""
    seconds = max(seconds, 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes}m{rest:02d}s"


def _pluralize(count: int, plural_noun: str) -> str:
    """``"1 file"`` / ``"3 files"`` from a plural noun form."""
    if count != 1:
        return f"{count} {plural_noun}"
    for suffix, singular_tail in (("ies", "y"), ("ses", "s"), ("s", "")):
        if plural_noun.endswith(suffix):
            return f"1 {plural_noun[:-len(suffix)]}{singular_tail}"
    return f"1 {plural_noun}"


def format_turn_summary(elapsed_seconds: float, tally: TurnTally | None, *, max_segments: int = _MAX_SEGMENTS) -> str:
    """Render the per-turn accounting line, or ``""`` when there's nothing to say.
    Pure; gating (``display.turn_summary``, quiet mode, CLI-only) is the caller's job."""
    if tally is None:
        tally = TurnTally()

    ordered = [v for v in _VERB_PRIORITY if v in tally.verbs] + [v for v in tally.verbs if v not in _VERB_PRIORITY]
    segments: list[str] = []
    for verb in ordered:
        parts = [_pluralize(count, plural) for plural, count in tally.verbs[verb].items() if count]
        if not parts:
            continue
        segment = f"{verb} {', '.join(parts)}"
        if verb == _EDIT_VERB and tally.has_line_deltas:
            segment += f" +{tally.lines_added} -{tally.lines_removed}"
        segments.append(segment)

    if tally.other_tools:
        segments.append(f"called {_pluralize(tally.other_tools, 'tools')}")

    if not segments and tally.total_tools == 0 and elapsed_seconds < _MIN_TOOLLESS_SECONDS:
        return ""

    if max_segments > 0 and len(segments) > max_segments:
        hidden = len(segments) - max_segments
        segments = segments[:max_segments] + [f"+{hidden} more"]

    return f"{SUMMARY_PREFIX} " + " · ".join([format_elapsed(elapsed_seconds)] + segments)


def format_token_flow(output_tokens: Any, *, arrow: str = "↓") -> str:
    """Cumulative turn tokens for the live spinner (``↓ 1.2k tok``); ``""`` for a
    non-positive count so nothing misleading shows before the first response."""
    try:
        count = int(output_tokens)
    except (TypeError, ValueError):
        return ""
    if count <= 0:
        return ""
    if count < 1000:
        return f"{arrow} {count} tok"
    if count < 1_000_000:
        return f"{arrow} {count / 1000:.1f}k tok"
    return f"{arrow} {count / 1_000_000:.1f}M tok"
