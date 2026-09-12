"""SkillEvalProducer — write real ``SkillEval`` records into the
ExperienceLedger from Hermes turn completions.

Phase 2 of the AIDE² self-evaluation plan (see
``docs/aide-squared-roadmap.md``). This module is the **bridge** between
the Hermes runtime and the data model shipped in Phase 1. Until Phase 2
lands, the ExperienceLedger has no producer — calls to
``ledger.record_eval(...)`` only happen in tests. After Phase 2, every
real skill invocation generates an eval record with multi-source
private signals.

Why this is a standalone module (not a hook inside ``run_agent.py``):

- **No core coupling.** The producer is pure Python; it can be invoked
  from anywhere — the turn finalizer, a gateway tool-call wrapper, a
  cron job, a test. It does not import run_agent, gateway, or any
  runtime module.
- **Testable in isolation.** All signal sources (user correction,
  rework count, reuse outcome) are pure functions; the producer is
  unit-tested against a temp ``HERMES_HOME`` with mocked inputs.
- **Pluggable signal sources.** Callers can supply their own signal
  functions or wrap the defaults. The producer does not assume a
  particular Hermes storage layout.

Usage::

    from agent.skill_eval_producer import SkillEvalProducer
    from agent.signal_sources.user_correction_detector import detect as detect_correction

    producer = SkillEvalProducer(hermes_home=Path.home() / ".hermes")

    # Called from your turn-finalizer / gateway hook:
    eval_record = producer.record_turn(
        skill_id="my-skill",
        task_family="coding",
        task_id="t-12345",
        public_signal=0.85,            # agent's self-rating
        cost_usd=0.12,
        tokens_in=1234,
        tokens_out=567,
        duration_sec=8.4,
        success=True,
        follow_up_user_messages=[msg.text for msg in recent_user_msgs],
        rework_events=[...],          # optional, for accurate rework count
        reuse_history=[...],          # optional, for reuse-success signal
    )

The producer is **async-safe** in the sense that ``record_turn`` is
synchronous and fast (signal computation + ledger write). Callers that
need to record thousands of evals per second should batch them via
``producer.record_batch([...])`` which serializes them under a single
ledger save.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from agent.experience_ledger import ExperienceLedger, SkillEval
from agent.signal_sources.rework_detector import (
    ReworkEvent,
    count_recent,
)
from agent.signal_sources.reuse_tracker import (
    ReuseEntry,
    lookup_reuse_outcome,
    mark_invocation,
)
from agent.signal_sources.user_correction_detector import detect as detect_correction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnSignals:
    """All raw signals for one turn, gathered before ledger write.

    Used as the canonical input to ``SkillEvalProducer.record_turn``.
    Frozen so callers can't accidentally mutate after construction.
    """

    skill_id: str
    task_family: str
    task_id: str
    public_signal: float  # agent self-rating (0-1)
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    duration_sec: float = 0.0
    success: bool = True
    follow_up_user_messages: Sequence[str] = field(default_factory=tuple)
    rework_events: Sequence[ReworkEvent] = field(default_factory=tuple)
    reuse_history: Sequence[ReuseEntry] = field(default_factory=tuple)
    eval_event_id: str = ""
    lineage: str = ""
    # Optional explicit overrides — if provided, skip the detector
    # and use the supplied value. Useful for tests and for callers
    # that have a more accurate signal source.
    user_corrected_override: Optional[bool] = None
    rework_count_override: Optional[int] = None
    reuse_success_override: Optional[bool] = None

    def __post_init__(self) -> None:
        if not self.skill_id:
            raise ValueError("TurnSignals.skill_id is required")
        if not self.task_id:
            raise ValueError("TurnSignals.task_id is required")
        if not self.task_family:
            raise ValueError("TurnSignals.task_family is required")
        if not (0.0 <= self.public_signal <= 1.0):
            raise ValueError(
                f"public_signal must be in [0, 1], got {self.public_signal!r}"
            )
        if self.cost_usd < 0:
            raise ValueError(f"cost_usd must be >= 0, got {self.cost_usd!r}")


class SkillEvalProducer:
    """Records ``SkillEval`` entries into the ExperienceLedger.

    The producer does **not** call any Hermes runtime. It exposes a
    single public method (``record_turn``) plus a batch helper. Callers
    wire it in wherever a turn finishes — typically in a wrapper
    around the turn finalizer or in a gateway tool-call return hook.

    Aggregation semantics
    ---------------------

    For each ``record_turn`` call, the producer computes three private
    signals and stores them on the resulting ``SkillEval``:

    - ``user_corrected``: True if any of the follow-up user messages
      matches a correction phrase in
      :mod:`agent.signal_sources.user_correction_detector`.
      Override-able via ``TurnSignals.user_corrected_override``.
    - ``rework_count``: number of times the same ``task_id`` was seen
      in the recent window (default 10 min), excluding the current
      attempt. Override-able via ``TurnSignals.rework_count_override``.
    - ``reuse_count``: incremented by one (current invocation), and
      ``reuse_success`` set from the next entry in the per-skill reuse
      history after the current timestamp (None if no further reuse).
      Override-able via ``TurnSignals.reuse_success_override``.

    Persistence
    -----------

    The producer uses the injected ``ExperienceLedger``. Every N calls
    (default 1) it persists via ``ledger.save()``. Batched callers
    should set ``auto_save=False`` and call ``producer.flush()`` at the
    end of the batch.
    """

    def __init__(
        self,
        hermes_home: Optional[Path] = None,
        ledger: Optional[ExperienceLedger] = None,
        *,
        rework_window_sec: float = 600.0,
        auto_save: bool = True,
    ) -> None:
        self.ledger = ledger or ExperienceLedger(hermes_home=hermes_home)
        self.rework_window_sec = rework_window_sec
        self.auto_save = auto_save
        # Track recent task_ids we've seen in this process so callers
        # who don't pre-supply rework_events still get a sensible
        # in-memory count.
        self._seen_task_events: List[ReworkEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_turn(
        self,
        signals: TurnSignals,
        *,
        now: Optional[float] = None,
    ) -> SkillEval:
        """Compute signals from raw inputs and record a ``SkillEval``.

        Returns the ``SkillEval`` that was added to the ledger.
        """
        if now is None:
            now = time.time()

        user_corrected = self._resolve_user_corrected(signals)
        rework_count = self._resolve_rework_count(signals, now=now)
        reuse_success = self._resolve_reuse_success(signals, now=now)

        eval_record = SkillEval(
            skill_id=signals.skill_id,
            eval_event_id=signals.eval_event_id
            or f"turn-{signals.task_id}-{int(now * 1000)}",
            task_family=signals.task_family,
            public_score=signals.public_signal,
            private_score=self._estimate_private_score(
                signals,
                user_corrected=user_corrected,
                rework_count=rework_count,
                reuse_success=reuse_success,
            ),
            cost_usd=signals.cost_usd,
            outcome=(
                "success"
                if signals.success and not user_corrected and rework_count == 0
                else "failure"
            ),
            tokens_in=signals.tokens_in,
            tokens_out=signals.tokens_out,
            duration_sec=signals.duration_sec,
            user_corrected=user_corrected,
            rework_count=rework_count,
            reuse_count=1,  # current invocation
            lineage=signals.lineage,
            created_at=now,
        )

        self.ledger.record_eval(eval_record)
        # Update in-memory trackers.
        self._seen_task_events.append(
            ReworkEvent(task_id=signals.task_id, timestamp=now)
        )
        # Trim in-memory rework history beyond the window.
        cutoff = now - self.rework_window_sec
        self._seen_task_events = [
            e for e in self._seen_task_events if e.timestamp >= cutoff
        ]

        if self.auto_save:
            self.ledger.save()

        logger.info(
            "SkillEvalProducer: recorded turn for skill=%s task=%s "
            "public=%.2f user_corrected=%s rework=%d reuse_success=%s",
            signals.skill_id,
            signals.task_id,
            signals.public_signal,
            user_corrected,
            rework_count,
            reuse_success,
        )
        return eval_record

    def record_batch(
        self,
        signals_batch: Iterable[TurnSignals],
        *,
        now: Optional[float] = None,
    ) -> List[SkillEval]:
        """Record multiple turns and save once at the end.

        Saves are deferred via ``auto_save=False`` for the duration of
        the batch, then ``ledger.save()`` is called exactly once.
        """
        if now is None:
            now = time.time()
        saved = self.auto_save
        self.auto_save = False
        try:
            out: List[SkillEval] = []
            for sig in signals_batch:
                out.append(self.record_turn(sig, now=now))
        finally:
            self.auto_save = saved
        if saved:
            self.ledger.save()
        return out

    def flush(self) -> None:
        """Persist pending ledger writes immediately."""
        self.ledger.save()

    def seen_task_events(self) -> List[ReworkEvent]:
        """Return a copy of the in-memory rework event log.

        Useful for tests and for callers who want to feed these back
        into a future call as ``signals.rework_events``.
        """
        return list(self._seen_task_events)

    # ------------------------------------------------------------------
    # Signal resolution helpers
    # ------------------------------------------------------------------

    def _resolve_user_corrected(self, signals: TurnSignals) -> bool:
        if signals.user_corrected_override is not None:
            return signals.user_corrected_override
        return detect_correction(signals.follow_up_user_messages)

    def _resolve_rework_count(
        self,
        signals: TurnSignals,
        *,
        now: float,
    ) -> int:
        if signals.rework_count_override is not None:
            return signals.rework_count_override
        # The rework_count on SkillEval is the number of times this
        # task_id appeared in the rework window *excluding* the current
        # turn (which is by definition not yet recorded). Both the
        # caller's explicit events and the in-memory tracker carry
        # past events only.
        events = list(signals.rework_events) + self._seen_task_events
        return count_recent(
            events,
            signals.task_id,
            now=now,
            window_sec=self.rework_window_sec,
        )

    def _resolve_reuse_success(
        self,
        signals: TurnSignals,
        *,
        now: float,
    ) -> Optional[bool]:
        if signals.reuse_success_override is not None:
            return signals.reuse_success_override
        return lookup_reuse_outcome(signals.reuse_history, after_timestamp=now)

    def _estimate_private_score(
        self,
        signals: TurnSignals,
        *,
        user_corrected: bool,
        rework_count: int,
        reuse_success: Optional[bool],
    ) -> float:
        """Compute an objective private score from the signal mix.

        The private score lives in [0, 1] and is a conservative
        composite of the public score degraded by the negative
        signals. It is a heuristic — the LLM judge (Phase 3) will
        replace this with a real measurement for ``EvalHarness``-
        driven evals, but for raw turn-by-turn records it is a
        reasonable starting point.

        Heuristic:

        - Start at the public score.
        - If user_corrected: subtract 0.4.
        - For each rework retry: subtract 0.15.
        - If reuse_success is False: subtract 0.2.
        - Clamp to [0, 1].
        """
        score = float(signals.public_signal)
        if user_corrected:
            score -= 0.4
        score -= 0.15 * rework_count
        if reuse_success is False:
            score -= 0.2
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        return round(score, 3)


__all__ = [
    "TurnSignals",
    "SkillEvalProducer",
]
