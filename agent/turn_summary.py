"""Per-turn accounting for the interactive CLI.

Two display-only pieces live here:

*   :class:`TurnSummaryCollector` — a tiny observer that rides the existing
    ``tool_progress_callback`` feed (``tool.completed`` events already carry
    the tool name and its raw result) and tallies what a turn actually did.
    It holds **no** agent-loop state: the display layer already sees every
    tool call, so nothing new is threaded through the conversation loop.
*   :func:`format_turn_summary` — a pure formatter that turns a tally plus a
    wall-clock duration into one dim line, e.g.::

        ⋯ 12.4s · edited 2 files +18 -3 · read 4 files · ran 3 commands

    Ported from Claude Code's post-turn accounting line
    ("Edited 1 file +6 -2, read 1 file … Worked for 10s").

:func:`format_token_flow` is the spinner-side counterpart: a cumulative
token readout appended to the live elapsed timer (``↓ 1.2k tok``).

Everything in this module is pure/side-effect free apart from the
collector's own counters, which makes it directly unit-testable without a
terminal, an agent, or a network call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TurnSummaryCollector",
    "TurnTally",
    "format_turn_summary",
    "format_token_flow",
    "format_elapsed",
]


# Leading glyph for the summary line. Deliberately not an emoji — the line is
# meant to read as terminal chrome, not as agent speech.
SUMMARY_PREFIX = "⋯"

# A turn that called no tools and finished this fast has nothing worth
# reporting (plain chat reply). Below the threshold the formatter returns "".
_MIN_TOOLLESS_SECONDS = 2.0

# Max number of "verb + count" segments rendered before collapsing the rest
# into a "+N more" tail, so a 12-tool turn cannot blow past one line.
_MAX_SEGMENTS = 4


# Tool name -> (verb, singular noun, plural noun).
#
# Verbs are past tense because the line is printed *after* the turn. Tools not
# listed here fall into a generic "called N tools" bucket rather than inventing
# phrasing for plugin/MCP tools whose semantics we don't know.
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

# Verb groups that carry file-edit line deltas (+X -Y) when known.
_EDIT_VERB = "edited"

# Render order: edits first (the thing users most want confirmed), then reads,
# then commands. Anything else follows in first-seen order.
_VERB_PRIORITY: tuple[str, ...] = ("edited", "read", "ran")

# Tools whose results may report a unified diff we can count lines from.
_DIFF_RESULT_TOOLS = frozenset({"patch"})


@dataclass
class TurnTally:
    """What a single turn did, as observed from the tool-progress feed."""

    # verb -> {noun_plural: count}; keeps insertion order for stable rendering.
    verbs: dict[str, dict[str, int]] = field(default_factory=dict)
    # Tools with no curated verb, counted together.
    other_tools: int = 0
    # Aggregated unified-diff line deltas across edit tools, when reported.
    lines_added: int = 0
    lines_removed: int = 0
    # True once at least one edit tool reported a countable diff, so the
    # formatter knows the difference between "+0 -0" and "unknown".
    has_line_deltas: bool = False

    @property
    def total_tools(self) -> int:
        counted = sum(sum(nouns.values()) for nouns in self.verbs.values())
        return counted + self.other_tools


def _count_diff_lines(diff: str) -> tuple[int, int]:
    """Count added/removed lines in unified-diff text.

    File headers (``+++``/``---``) are excluded so a one-line edit does not
    read as three additions.
    """
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _extract_line_deltas(tool_name: str, result: Any) -> tuple[int, int] | None:
    """Pull (added, removed) from a tool result, or None when unavailable.

    Only tools that already report a diff in their result payload are
    inspected — we never shell out to git and never re-read files to
    synthesise a delta.
    """
    if tool_name not in _DIFF_RESULT_TOOLS:
        return None
    payload: Any = result
    if isinstance(payload, str):
        text = payload.strip()
        if not text.startswith("{"):
            return None
        try:
            import json

            # strict=False tolerates literal control characters inside strings
            # (raw newlines in an embedded diff), which some tool serialisers
            # emit. A tally line is never worth failing over formatting.
            payload = json.loads(text, strict=False)
        except Exception:
            return None
    if not isinstance(payload, dict):
        return None
    diff = payload.get("diff")
    if not isinstance(diff, str) or not diff.strip():
        return None
    added, removed = _count_diff_lines(diff)
    # A diff that carries no +/- content lines (e.g. a bare hunk header) tells
    # us nothing — report it as unknown rather than rendering a misleading
    # "+0 -0" next to a real edit.
    if added == 0 and removed == 0:
        return None
    return added, removed


class TurnSummaryCollector:
    """Accumulate per-turn tool tallies from the tool-progress feed.

    Wired into the CLI's existing ``_on_tool_progress`` handler: the display
    layer already receives every ``tool.completed`` event with the tool name
    and raw result, so no agent-loop bookkeeping is added.
    """

    def __init__(self) -> None:
        self._tally = TurnTally()

    def begin(self) -> None:
        """Start a fresh turn (drops any prior tally)."""
        self._tally = TurnTally()

    def record_tool(
        self,
        tool_name: str | None,
        *,
        result: Any = None,
        is_error: bool = False,
    ) -> None:
        """Record one completed tool call.

        Failed calls are skipped: a summary claiming "edited 2 files" when one
        write was denied would be exactly the over-claim the file-mutation
        verifier exists to catch.
        """
        if not tool_name or is_error:
            return
        # Internal/pseudo tools (``_thinking``) are not user-visible work.
        if tool_name.startswith("_"):
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
        """Render this turn's summary line (see :func:`format_turn_summary`)."""
        return format_turn_summary(elapsed_seconds, self._tally)


def format_elapsed(seconds: float) -> str:
    """Format a wall-clock duration compactly (``12.4s`` / ``2m05s``)."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes}m{rest:02d}s"


def _pluralize(count: int, plural_noun: str) -> str:
    """Return ``"1 file"`` / ``"3 files"`` from a plural noun form."""
    if count == 1:
        singular = plural_noun
        if plural_noun.endswith("ies"):
            singular = plural_noun[:-3] + "y"
        elif plural_noun.endswith("ses"):
            singular = plural_noun[:-2]
        elif plural_noun.endswith("s"):
            singular = plural_noun[:-1]
        return f"1 {singular}"
    return f"{count} {plural_noun}"


def _ordered_verbs(tally: TurnTally) -> list[str]:
    """Verbs in render order: priority verbs first, then first-seen order."""
    seen = list(tally.verbs.keys())
    ranked = [v for v in _VERB_PRIORITY if v in tally.verbs]
    ranked += [v for v in seen if v not in _VERB_PRIORITY]
    return ranked


def format_turn_summary(
    elapsed_seconds: float,
    tally: TurnTally | None,
    *,
    max_segments: int = _MAX_SEGMENTS,
) -> str:
    """Render the per-turn accounting line, or ``""`` when there's nothing to say.

    Pure function — no config lookups, no terminal access, no I/O. Gating
    (``display.turn_summary``, quiet mode, CLI-only) is the caller's job.
    """
    if tally is None:
        tally = TurnTally()

    segments: list[str] = []
    for verb in _ordered_verbs(tally):
        nouns = tally.verbs[verb]
        parts = [_pluralize(count, plural) for plural, count in nouns.items() if count]
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

    pieces = [format_elapsed(elapsed_seconds)] + segments
    return f"{SUMMARY_PREFIX} " + " · ".join(pieces)


def format_token_flow(output_tokens: Any, *, arrow: str = "↓") -> str:
    """Render cumulative turn tokens for the live spinner (``↓ 1.2k tok``).

    Returns ``""`` for a non-positive count so the spinner shows nothing
    rather than a misleading ``↓ 0 tok`` before the first API response lands.
    """
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
