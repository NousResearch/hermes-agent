"""Reference integration of SkillEvalProducer into a turn-style hook.

This module is **not** auto-loaded. It exists as a worked example so
maintainers wiring the producer into ``run_agent.py`` / ``gateway/run.py``
have a concrete pattern to copy. The actual integration lives in the
caller's runtime hook.

The function ``wrap_turn`` takes the inputs a turn-finalizer would
naturally have — skill_id, task_id, the recent user messages, cost
and token metadata — and produces a ``SkillEval`` record via the
producer. It returns the recorded ``SkillEval`` so the caller can
surface it back to the user (or silently swallow it).

Pattern::

    from agent.hermes_eval_hook import wrap_turn
    from pathlib import Path

    async def my_turn_finalizer(skill_id, task_id, *, msgs, cost, ...):
        eval_record = wrap_turn(
            skill_id=skill_id,
            task_id=task_id,
            task_family="coding",
            hermes_home=Path.home() / ".hermes",
            user_messages=[m.text for m in msgs[-3:]],
            cost_usd=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_sec=duration,
            success=success_flag,
        )
        # ... continue normal turn-finalizer logic

Recommended insertion points (in order of preference):

1. ``run_agent.py`` turn-finalizer (preferred — single chokepoint)
2. ``gateway/run.py`` tool-call return wrapper (per-tool granularity)
3. A cron job that ingests an exported turn log (fallback)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence

from agent.experience_ledger import SkillEval
from agent.skill_eval_producer import SkillEvalProducer, TurnSignals

logger = logging.getLogger(__name__)


def wrap_turn(
    *,
    skill_id: str,
    task_id: str,
    task_family: str,
    hermes_home: Path,
    public_signal: float,
    cost_usd: float = 0.0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    duration_sec: float = 0.0,
    success: bool = True,
    user_messages: Optional[Sequence[str]] = None,
    eval_event_id: str = "",
    lineage: str = "",
    rework_window_sec: float = 600.0,
    producer: Optional[SkillEvalProducer] = None,
) -> SkillEval:
    """Record a single turn into the ExperienceLedger via the producer.

    Returns the recorded ``SkillEval`` so callers can surface it.
    Does not raise on producer-side errors — failures are logged at
    WARNING so a broken ledger never breaks the turn.
    """
    own_producer = producer is None
    if own_producer:
        producer = SkillEvalProducer(
            hermes_home=hermes_home,
            rework_window_sec=rework_window_sec,
        )

    signals = TurnSignals(
        skill_id=skill_id,
        task_id=task_id,
        task_family=task_family,
        public_signal=public_signal,
        cost_usd=cost_usd,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration_sec=duration_sec,
        success=success,
        follow_up_user_messages=tuple(user_messages or ()),
        eval_event_id=eval_event_id,
        lineage=lineage,
    )

    try:
        record = producer.record_turn(signals)
    except Exception as e:  # noqa: BLE001 — never break a turn on ledger error
        logger.warning(
            "hermes_eval_hook: failed to record SkillEval for skill=%s task=%s: %s",
            skill_id,
            task_id,
            e,
        )
        return None  # type: ignore[return-value]
    return record


__all__ = ["wrap_turn"]
