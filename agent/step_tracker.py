"""Step completion tracker — parses ``finish(complete|skip|fail)`` signals.

Ported from GH05TCREW/pentestagent's anti-zombie-loop pattern (2026-08-09).

The model can embed a ``finish(...)`` call anywhere in its text response to
explicitly declare what happened to the current step:

    finish(complete)  — step done, tool output verified, move on
    finish(skip)      — step skipped: env constraint, resource not available, etc.
    finish(fail)      — step attempted and tool-confirmed to have failed

This is a *cooperative* signal — the model opts in by emitting it.  Callers
that want to detect stuck loops without relying solely on the passive
``tool_guardrails`` heuristics can call ``parse_finish_signal`` on the
assistant's reply text each turn.

Design constraints (from AGENTS.md):
- No new core tools — this is a pure text-parsing helper, zero API surface.
- Cache-safe — no system-prompt mutation.
- Stateless module — the per-session accumulator lives in the caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Regex: ``finish(complete)``, ``finish( skip )``, ``FINISH(FAIL)`` etc.
# Anchored to word boundary so ``definish(...)`` doesn't match.
_FINISH_RE = re.compile(
    r"""\bfinish\s*\(\s*(complete|skip|fail)\s*\)""",
    re.IGNORECASE,
)


class StepOutcome(str, Enum):
    COMPLETE = "complete"
    SKIP = "skip"
    FAIL = "fail"


@dataclass(frozen=True)
class FinishSignal:
    """Parsed ``finish(...)`` call from a model reply."""

    outcome: StepOutcome
    raw_match: str  # verbatim matched text — kept for logging, never re-parsed


def parse_finish_signal(text: str) -> Optional[FinishSignal]:
    """Return the *first* ``finish(...)`` signal found in *text*, or ``None``.

    Only the first occurrence matters per turn — a model that emits two
    conflicting signals in the same reply is confused, and we take the earlier
    one.
    """
    if not text:
        return None
    m = _FINISH_RE.search(text)
    if m is None:
        return None
    outcome_str = m.group(1).lower()
    try:
        outcome = StepOutcome(outcome_str)
    except ValueError:
        # Regex already constrains the group — this branch is unreachable but
        # kept for safety so future regex changes don't silently break callers.
        return None
    return FinishSignal(outcome=outcome, raw_match=m.group(0))


@dataclass
class StepTracker:
    """Per-session accumulator for ``finish(...)`` outcomes.

    Usage::

        tracker = StepTracker()

        # in the conversation loop, after each assistant reply:
        signal = parse_finish_signal(reply_text)
        if signal:
            tracker.record(signal)

        # optionally check for a stalled agent:
        if tracker.consecutive_fails >= 3:
            # surface a warning to the user / guardrail layer
            ...

    The tracker is intentionally *not* a guardrail decision-maker — it only
    accumulates observations.  The caller decides what to do with them, just
    as ``ToolCallGuardrailController`` returns decisions the caller acts on.
    """

    outcomes: list[StepOutcome] = field(default_factory=list)

    def record(self, signal: FinishSignal) -> None:
        self.outcomes.append(signal.outcome)

    def reset(self) -> None:
        """Clear accumulated outcomes (e.g. at turn boundary)."""
        self.outcomes.clear()

    @property
    def consecutive_fails(self) -> int:
        """Count of trailing ``fail`` outcomes — useful for stall detection."""
        count = 0
        for outcome in reversed(self.outcomes):
            if outcome is StepOutcome.FAIL:
                count += 1
            else:
                break
        return count

    @property
    def consecutive_skips(self) -> int:
        """Count of trailing ``skip`` outcomes."""
        count = 0
        for outcome in reversed(self.outcomes):
            if outcome is StepOutcome.SKIP:
                count += 1
            else:
                break
        return count

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def is_stuck(self) -> bool:
        """Heuristic: 3+ consecutive fails or skips → agent is likely stuck."""
        return self.consecutive_fails >= 3 or self.consecutive_skips >= 3
