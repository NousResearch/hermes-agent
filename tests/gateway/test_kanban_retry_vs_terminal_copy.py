"""A retryable crash and a terminal ``gave_up`` must read as different outcomes.

Two things go wrong today when a worker dies:

* ``crashed`` promises "it will be retried automatically", but the breaker is
  accounted AFTER the crash is recorded (``_reclaim_dead_workers`` appends the
  event inside the reclaim txn; ``_account_crashes`` decides afterwards). When the
  crash is the one that trips the limit, the promise is contradicted by a ``⛔
  blocked`` ping moments later.
* ``gave_up`` is emitted for spawn failures, crashes and timeouts alike and
  carries ``trigger_outcome`` saying which — but the TUI/desktop copy attributes
  every trip to "repeated spawn failures", sending operators to the wrong logs.

Both surfaces must agree; ``tui_gateway.session_notifications`` documents itself as
mirroring the gateway wording.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.kanban_watchers_notifier import _EVENT_FORMATTERS


def _names(task_id="T-123"):
    return SimpleNamespace(
        task_id=task_id, head=f"[board] Kanban {task_id}", title="Ship it", board_tag="[board] ")


def _event(**payload):
    return SimpleNamespace(payload=payload)


def _gateway(kind, **payload) -> str:
    msg, _wake, _reason = _EVENT_FORMATTERS[kind](_event(**payload), _names())
    return msg


def _tui(kind, **payload) -> str:
    from tui_gateway.session_notifications import _KANBAN_EVENT_FORMATTERS

    _glyph, fmt = _KANBAN_EVENT_FORMATTERS[kind]
    task = SimpleNamespace(title="Ship it", assignee="alpha", result=None)
    return fmt(task, payload, "Ship it")


TRIGGERS = [
    ("crashed", "crash"),
    ("timed_out", "timed out"),
    ("spawn_failed", "spawn"),
]


@pytest.mark.parametrize("trigger_outcome,expected", TRIGGERS)
def test_gateway_gave_up_names_the_outcome_that_tripped_it(trigger_outcome, expected):
    msg = _gateway("gave_up", failures=2, trigger_outcome=trigger_outcome).lower()
    assert expected in msg, msg


@pytest.mark.parametrize("trigger_outcome,expected", TRIGGERS)
def test_tui_gave_up_names_the_outcome_that_tripped_it(trigger_outcome, expected):
    """The TUI must not send a crash-tripped trip to the spawn logs."""
    msg = _tui("gave_up", failures=2, trigger_outcome=trigger_outcome).lower()
    assert expected in msg, msg
    if trigger_outcome != "spawn_failed":
        assert "spawn" not in msg, msg


def test_gave_up_without_a_trigger_still_reads_as_terminal():
    """Legacy events (pre-``trigger_outcome``) must degrade, not crash or lie."""
    for msg in (_gateway("gave_up", failures=2), _tui("gave_up", failures=2)):
        assert "blocked" in msg.lower()
        assert "spawn" not in msg.lower()


@pytest.mark.parametrize("render", [_gateway, _tui])
def test_crash_copy_does_not_assert_an_undecided_outcome(render):
    """The crash event is written before the breaker decides, so the copy must
    admit the terminal possibility instead of guaranteeing a retry."""
    msg = render("crashed").lower()
    assert "retr" in msg, "it is still normally a retry"
    assert "block" in msg, (
        "a ⛔ blocked ping may follow in the same tick; the crash copy must not "
        f"read as a guarantee: {msg!r}")


@pytest.mark.parametrize("render", [_gateway, _tui])
def test_terminal_copy_never_promises_another_attempt(render):
    msg = render("gave_up", failures=2, trigger_outcome="crashed").lower()
    assert "will be retried" not in msg
    assert "will retry" not in msg


def test_tui_and_gateway_agree_on_which_kinds_are_terminal():
    """Divergent formatter tables are how the TUI copy went stale in the first place."""
    from tui_gateway.session_notifications import _KANBAN_EVENT_FORMATTERS

    for kind in _KANBAN_EVENT_FORMATTERS:
        assert kind in _EVENT_FORMATTERS, f"{kind} is rendered on the TUI but not the gateway"
