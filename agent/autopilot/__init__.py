"""Autopilot honesty helpers for the persisted ``/goal`` Ralph loop.

Scope note (reconciliation with review feedback on PR #51565)
-------------------------------------------------------------
The original PR shipped a *parallel* orchestration engine (``driver.py`` +
``council_gate.py``) that duplicated Hermes' already-established ``/goal`` loop
(``hermes_cli/goals.py``: completion contracts, subgoals, wait barriers, and a
bounded per-turn turn budget). The sweeper's core verdict was that this must be
*reconciled with* that path rather than added as a second engine, and that the
``--autopilot`` flag must not silently widen command authorization by enabling
YOLO.

This reconciled version drops the parallel engine and the YOLO auto-enable. It
keeps only the two genuinely-additive, self-contained pieces and offers them as
opt-in honesty signals the existing ``/goal`` loop can consult:

    * :mod:`agent.autopilot.deception` — a cheap, no-model heuristic detector for
      the known reward-seeking cheat patterns (claim-without-evidence, await-user
      handoff, scope-shrink, effort excuses, ...). It never blocks on its own; a
      flag just sharpens the ``/goal`` continuation directive and can be logged.
    * :mod:`agent.autopilot.adr` — an append-only, off-by-default, fail-soft
      decision log that records what the ``/goal`` judge returned each turn.

Both are pure additive helpers with no import into the agent hot loop, so they
carry none of the continuation-budget or authorization concerns raised in
review. The ``/goal`` loop's own bounded turn budget remains the single,
non-bypassable safety cap.
"""

from __future__ import annotations

from agent.autopilot import adr, deception
from agent.autopilot.deception import DeceptionSignal, scan as scan_deception

__all__ = [
    "adr",
    "deception",
    "DeceptionSignal",
    "scan_deception",
    "strict_completion",
]


def strict_completion(value: object) -> bool:
    """Strictly interpret a judge/reviewer completion verdict as a boolean.

    Review nit (teknium1 @ council_gate.py:281): ``bool("false")`` is ``True``,
    so a malformed-but-plausible reviewer response (a JSON string ``"false"``,
    the word ``"no"``, ``0``) could incorrectly permit completion. Accept ONLY an
    actual boolean ``True`` or an explicitly-truthy token; treat anything else —
    including any non-boolean type — as *not complete* (fail safe: keep going).
    """
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "done", "complete", "1"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    return False
