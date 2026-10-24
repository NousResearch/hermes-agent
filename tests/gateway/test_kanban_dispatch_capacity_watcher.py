"""The GATEWAY dispatcher path must tell "AT CAPACITY" apart from "spawn is BROKEN".

Field incident (2026-10-24): the embedded gateway dispatcher logged

    WARNING gateway.run: kanban dispatcher stuck: ready queue non-empty for
    6 consecutive ticks but 0 workers spawned.

That was a FALSE POSITIVE. ``config.yaml`` had ``kanban.max_spawn: 3`` and
exactly 3 tasks were running with live heartbeats and live worker PIDs. The
board was AT CAPACITY, working correctly: the ready queue stays non-empty
precisely BECAUSE the running workers are still busy, and 0 spawns is the
expected outcome of a cap, not a symptom of a broken profile.

The warning cried wolf during exactly the incidents when triage attention is
scarce.

These tests pin the GATEWAY side of that distinction. The CLI daemon side
(``hermes_cli/kanban_ops.py``) is covered in
``tests/hermes_cli/test_kanban_dispatch_capacity_saturation.py``; the two run
different code over different data shapes, and only the gateway one consumed
``(slug, result)`` PAIRS.

Regression guard: the first cut of the gateway fix read ``res.capacity_saturated``
straight off the pair instead of the result, raising ``AttributeError: 'tuple'
object has no attribute 'capacity_saturated'`` inside the watcher loop on every
tick. The exception killed the tick BEFORE the warning could be logged, so the
dispatcher silently stopped reporting anything at all.
"""

from __future__ import annotations

from gateway.kanban_watchers_dispatcher import _dispatch_saturated
from hermes_cli import kanban_db_dispatch as kbd


def _saturated_result() -> kbd.DispatchResult:
    result = kbd.DispatchResult()
    result.capacity_saturated = "max_spawn"
    return result


def test_gateway_results_are_pairs_not_bare_results():
    """The gateway consumes PAIRS; the helper must unpack them.

    ``tick_once()`` returns ``list[tuple[str, DispatchResult]]``. A helper that
    treats each element as the result object reads ``capacity_saturated`` off
    the tuple and dies.
    """
    results = [("homelab", _saturated_result())]
    assert results[0][1].capacity_saturated == "max_spawn"


def test_a_capacity_blocked_tick_reads_as_saturated():
    """The board at its cap is AT CAPACITY, so the warning must not cry wolf."""
    assert _dispatch_saturated([("homelab", _saturated_result())]) is True


def test_a_healthy_tick_reads_as_not_saturated():
    """No cap hit: the board can spawn, so 0 spawns IS a fault."""
    assert _dispatch_saturated([("homelab", kbd.DispatchResult())]) is False


def test_a_saturated_board_is_detected_across_multiple_boards():
    """One board at the cap makes the whole tick a cap-blocked tick."""
    results = [
        ("homelab", kbd.DispatchResult()),
        ("curator", _saturated_result()),
    ]
    assert _dispatch_saturated(results) is True


def test_a_board_that_produced_no_result_does_not_crash_the_check():
    """A quarantined or failed board yields ``None``; the tick must survive."""
    results = [
        ("broken", None),
        ("homelab", _saturated_result()),
    ]
    assert _dispatch_saturated(results) is True


def test_no_results_is_not_saturated():
    """An empty tick (every board quarantined) is not a capacity block."""
    assert _dispatch_saturated([]) is False
    assert _dispatch_saturated(None) is False
