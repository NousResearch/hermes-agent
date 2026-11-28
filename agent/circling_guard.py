"""Detect a model that keeps restating its own sentences while one reply streams.

``agent.repetition_guard`` cuts a stream once exact contiguous repeats dominate a 64k-char tail.
A reasoning model can circle long before that, and in shapes that check never sees: a plan
of several paragraphs restated every few thousand characters with other text in between (one
recorded turn restated one plan 8 times over 54k chars and was never flagged), or a slow
spiral that only tightens into an exact loop after tens of thousands of characters (another
ran to 409k chars; the tail check would have cut it at 64k, this one at 27k).

The unit here is the line (a long line is split into sentences), compared exactly after case
and whitespace folding. The watch fires when the latest ``window_units`` units are almost all
restatements of units already written in the same reply, and at least half of them were already
written ``repeat_count`` times or more. Text that is new keeps the watch quiet however long the
reply grows; short units ("Let me check.", "}", "else:") are ignored. State is bounded: one
counter per distinct unit, up to a cap.

It is meant for the reasoning channel only. Visible text and tool output repeat lines for good
reasons (a sentence the user asked for 500 times, a log): 4 of 304 recorded tool results over
8k chars would trip it. None of 619 recorded tool-call argument payloads did.

``ReasoningStreamWatch`` bundles it with the exact-loop ``RunawayStreamWatch`` for one streamed
reply and hands a cut to the length path (``agent.turn_truncation``), which re-issues the call
with ``recovery_feedback`` up to ``repetition_guard.recovery.max_attempts`` times per turn.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agent.repetition_guard import RunawayStreamWatch

# Defaults picked from a grid replayed over 131 recorded agent sessions of one hosted reasoning model
# (7,625 reasoning channels, 53.6M chars): the three runaway turns are cut at 25k-33k chars.
DEFAULT_WINDOW_UNITS = 60
DEFAULT_STALE_RATIO = 0.95
DEFAULT_REPEAT_COUNT = 3
DEFAULT_HEAVY_RATIO = 0.5
DEFAULT_MIN_UNIT_CHARS = 30
# Below this much streamed text a cut saves little; the earliest fire in the recorded sessions was at 12.7k chars.
DEFAULT_MIN_CHARS = 8_000
# Recoveries per turn after a repetition cut; then the turn ends with the repetition copy.
DEFAULT_RECOVERY_ATTEMPTS = 1

_MAX_DISTINCT_UNITS = 50_000
# Text held while waiting for a line break; past it the buffer is released in sentences or slices.
_MAX_BUFFER_CHARS = 4_000
DEFAULT_LONG_LINE_CHARS = 400
_LINE_BREAK = re.compile(r"\n+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class CirclingSettings:
    window_units: int = DEFAULT_WINDOW_UNITS
    stale_ratio: float = DEFAULT_STALE_RATIO
    repeat_count: int = DEFAULT_REPEAT_COUNT
    heavy_ratio: float = DEFAULT_HEAVY_RATIO
    min_unit_chars: int = DEFAULT_MIN_UNIT_CHARS
    min_chars: int = DEFAULT_MIN_CHARS
    long_line_chars: int = DEFAULT_LONG_LINE_CHARS


class CirclingStreamWatch:
    """Feed one channel's deltas; :meth:`feed` returns True once the channel is circling."""

    __slots__ = ("_buffer", "_counts", "_heavy", "_recent", "_settings", "_stale", "chars", "fired_at")

    def __init__(self, settings: CirclingSettings | None = None) -> None:
        self._settings = settings or CirclingSettings()
        self._counts: dict[str, int] = {}
        self._recent: deque[tuple[bool, bool]] = deque(maxlen=self._settings.window_units)
        self._buffer = ""
        self._stale = self._heavy = 0
        self.chars = 0
        self.fired_at: int | None = None

    def feed(self, text: str) -> bool:
        if not text:
            return False
        self.chars += len(text)
        self._buffer += text
        lines = _LINE_BREAK.split(self._buffer)
        self._buffer = lines.pop()
        if len(self._buffer) > _MAX_BUFFER_CHARS:
            # No line break for a long stretch: release the complete sentences, or a slice if there are none.
            sentences = _SENTENCE_BREAK.split(self._buffer)
            self._buffer = sentences.pop()
            lines.extend(sentences)
            if len(self._buffer) > _MAX_BUFFER_CHARS:
                lines.append(self._buffer[:_MAX_BUFFER_CHARS])
                self._buffer = self._buffer[_MAX_BUFFER_CHARS:]
        fired = False
        for line in lines:
            # A line is one unit, so rows that differ only in a leading field stay distinct; a long
            # paragraph on one line is judged sentence by sentence.
            for unit in (_SENTENCE_BREAK.split(line) if len(line) > self._settings.long_line_chars else (line,)):
                if unit and self._observe(unit):
                    fired = True
        if fired and self.fired_at is None:
            self.fired_at = self.chars
        return fired

    def _observe(self, raw: str) -> bool:
        settings = self._settings
        unit = _WHITESPACE.sub(" ", raw).strip().lower()
        if len(unit) < settings.min_unit_chars:
            return False
        count = self._counts.get(unit, 0) + 1
        if count > 1 or len(self._counts) < _MAX_DISTINCT_UNITS:
            self._counts[unit] = count
        recent = self._recent
        if len(recent) == settings.window_units:
            was_stale, was_heavy = recent[0]
            self._stale -= was_stale
            self._heavy -= was_heavy
        flags = (count >= 2, count >= settings.repeat_count)
        recent.append(flags)
        self._stale += flags[0]
        self._heavy += flags[1]
        return (
            len(recent) == settings.window_units
            and self.chars >= settings.min_chars
            and self._stale >= settings.stale_ratio * settings.window_units
            and self._heavy >= settings.heavy_ratio * settings.window_units
        )

    def repeated_units(self, limit: int = 3) -> list[tuple[int, str]]:
        """The most repeated units so far as ``(count, text)``, for the recovery message and the log."""
        repeated = sorted(((c, u) for u, c in self._counts.items() if c >= 2), reverse=True)
        return repeated[:limit]


# ---- configuration (``repetition_guard`` in config.yaml) ----------------------------------------

_BOOL_WORDS = {w: True for w in ("1", "true", "yes", "on")} | {w: False for w in ("0", "false", "no", "off")}


def _flag(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return _BOOL_WORDS.get(value.strip().lower(), default) if isinstance(value, str) else default


def _int_at_least(value: Any, default: int, minimum: int) -> int:
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        return default  # YAML ``yes``/``no``, ``.inf`` and fractions are not counts
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _ratio(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if 0.0 < parsed <= 1.0 else default


def settings_from_config(data: Any) -> CirclingSettings | None:
    """``repetition_guard.circling`` as watch settings, or None when ``enabled`` is false.

    A missing or malformed section, and any bad value, falls back to the default for that value."""
    if not isinstance(data, Mapping):
        return CirclingSettings()
    if not _flag(data.get("enabled"), True):
        return None
    d = CirclingSettings()
    return CirclingSettings(
        window_units=_int_at_least(data.get("window_units"), d.window_units, 2),
        stale_ratio=_ratio(data.get("stale_ratio"), d.stale_ratio),
        repeat_count=_int_at_least(data.get("repeat_count"), d.repeat_count, 2),
        min_chars=_int_at_least(data.get("min_chars"), d.min_chars, 0),
    )


@dataclass(frozen=True)
class RecoverySettings:
    """``repetition_guard.recovery``: re-issues per turn after a cut (0 ends the turn on the first
    cut), and whether the re-issued call keeps the configured reasoning or goes out with it off."""

    max_attempts: int = DEFAULT_RECOVERY_ATTEMPTS
    reasoning: Literal["keep", "off"] = "keep"

    @classmethod
    def from_mapping(cls, data: Any) -> RecoverySettings:
        if not isinstance(data, Mapping):
            return cls()
        raw = data.get("reasoning")
        # Unquoted YAML ``off`` arrives as False.
        reasoning = {False: "off", True: "keep"}.get(raw) if isinstance(raw, bool) else str(raw or "").strip().lower()
        return cls(
            max_attempts=_int_at_least(data.get("max_attempts"), DEFAULT_RECOVERY_ATTEMPTS, 0),
            reasoning="off" if reasoning == "off" else "keep",
        )


def apply_repetition_guard_config(agent: Any, section: Any) -> None:
    """Set the agent attributes the stream watch and the length path read from ``repetition_guard``."""
    section = section if isinstance(section, Mapping) else {}
    agent._circling_settings = settings_from_config(section.get("circling"))
    recovery = RecoverySettings.from_mapping(section.get("recovery"))
    agent._repetition_recovery_limit = recovery.max_attempts
    agent._repetition_recovery_reasoning_off = recovery.reasoning == "off"


# ---- one streamed reply ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepetitionCut:
    """What the stream watch saw when it cut a reply: the channel, which detector fired
    (``circling`` or the exact-loop ``exact_loop``), the reasoning characters streamed before the
    cut (None for a visible-text cut) and the most repeated units as ``(count, text)``."""

    channel: str
    detector: str
    chars: int | None
    repeated: tuple[tuple[int, str], ...] = ()

    def describe(self) -> str:
        counts = [count for count, _ in self.repeated[:2]]  # counts only: reasoning stays out of WARNING logs
        return f"detector={self.detector}, chars={self.chars}, repeated_counts={counts}"


class ReasoningStreamWatch:
    """The reasoning channel's watches for one streamed reply.

    ``RunawayStreamWatch`` runs on the plain field and, separately, on the structured details (the
    display takes one representation per chunk, but the plain field is persisted either way and
    need not mirror the details). Restated sentences with other text between them never dominate a
    tail, so the circling watch runs beside them unless ``repetition_guard.circling`` disables it
    (``agent._circling_settings is None``)."""

    __slots__ = ("_circling", "_detail", "_plain", "chars", "detector")

    def __init__(self, agent: Any = None) -> None:
        self._plain, self._detail = RunawayStreamWatch(), RunawayStreamWatch()
        settings = getattr(agent, "_circling_settings", CirclingSettings())
        self._circling = CirclingStreamWatch(settings) if settings is not None else None
        self.chars = 0
        self.detector: str | None = None

    def feed(self, reasoning_text: str | None, detail_text: str | None = None) -> bool:
        """Add one chunk's reasoning; True once the channel is a runaway loop or circling."""
        self.chars += len(reasoning_text or detail_text or "")
        if self._plain.feed(reasoning_text or "") or self._detail.feed(detail_text or ""):
            self.detector = "exact_loop"
        elif self._circling is not None and self._circling.feed(reasoning_text or detail_text or ""):
            self.detector = "circling"
        return self.detector is not None

    def cut(self, channel: str) -> RepetitionCut:
        if channel != "reasoning":
            return RepetitionCut(channel, "exact_loop", None)
        repeated = tuple(self._circling.repeated_units()) if self._circling is not None else ()
        return RepetitionCut(channel, self.detector or "exact_loop", self.chars, repeated)

    def mark_cut(self, response: Any, channel: str) -> None:
        """Cut, not finished: the length path recovers or ends the turn on these marks."""
        response._runaway_repetition = True
        response._repetition_cut = self.cut(channel)


# ---- per-turn bookkeeping -------------------------------------------------------------------------

@dataclass
class RepetitionTurn:
    """``agent._repetition_turn``, reset at turn start; ``summary()`` is ``result["repetition"]``."""

    cuts: int = 0
    recoveries: int = 0
    cut_at_chars: list[int] = field(default_factory=list)
    repeated: list[str] = field(default_factory=list)

    def note_cut(self, cut: RepetitionCut | None) -> None:
        self.cuts += 1
        if cut is None:
            return
        if cut.chars is not None:
            self.cut_at_chars.append(cut.chars)
        units = (unit[:_FEEDBACK_UNIT_CHARS] for _, unit in cut.repeated[:2])
        self.repeated.extend(unit for unit in units if unit not in self.repeated)

    def summary(self) -> dict[str, Any]:
        return {"cuts": self.cuts, "recoveries": self.recoveries, "cut_at_chars": list(self.cut_at_chars),
                "repeated": list(self.repeated)}


def repetition_turn(agent: Any) -> RepetitionTurn:
    turn = getattr(agent, "_repetition_turn", None)
    if not isinstance(turn, RepetitionTurn):
        turn = agent._repetition_turn = RepetitionTurn()
    return turn


def repetition_result(agent: Any) -> dict[str, Any]:
    """``{"repetition": ...}`` to merge into the turn result when the turn had a repetition cut, else {}."""
    turn = getattr(agent, "_repetition_turn", None)
    return {"repetition": turn.summary()} if isinstance(turn, RepetitionTurn) else {}


_FEEDBACK_UNIT_CHARS = 160


def recovery_feedback(repeated: Sequence[tuple[int, str]] | None) -> str:
    """The message sent in place of a reply that was cut for circling.

    It says how often a statement came back and asks for one different concrete action or a blocker
    report. It quotes none of the reasoning: this row is stored with the turn (merged into the user's
    own row when it follows it), shown on resume and replayed in every later request. The cut reply
    itself is never replayed: looped text in the next request re-seeds the loop (#112764).
    """
    restated = f" (one of them {repeated[0][0]} times)" if repeated else ""
    return (
        "[System: Your previous reply was stopped because it kept restating the same statements without "
        f"acting{restated}. Do not deliberate over this again. Either make exactly one concrete tool call that "
        "is different from what you already tried, or reply with a short blocker report: what is blocked, "
        "what you tried, and what you would need to continue.]"
    )


def repeated_statements_copy(repeated: Sequence[tuple[int, str]] | None) -> str:
    """Appended to the final copy once the recoveries are spent. Counts only: reasoning is private on
    most routes, so none of its text reaches the visible reply (#111761)."""
    if not repeated:
        return ""
    return f"\n\nThe reasoning restated the same statements up to {repeated[0][0]} times."
