"""Regression tests for #30323: leftover /steer ordering vs interrupt follow-up.

The CLI keeps a single FIFO ``_pending_input`` queue feeding ``process_loop``.
When a run ends with both:

  * leftover ``pending_steer`` (steer arrived after the final tool batch
    could absorb it, so the agent returned it in ``result["pending_steer"]``)
  * a queued normal follow-up message (via the interrupt path)

then the leftover steer MUST be applied first on the next turn, otherwise
``/steer`` guidance arrives too late and appears dropped.
"""

from __future__ import annotations

import queue

from cli import HermesCLI


def _make_cli_under_test():
    """Construct a HermesCLI without running its heavy __init__."""
    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    cli._interrupt_queue = queue.Queue()
    return cli


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_leftover_steer_enqueued_before_interrupt_followup():
    cli = _make_cli_under_test()

    cli._enqueue_post_run_followups(
        pending_message="please continue with the refactor",
        result={"pending_steer": "actually focus on the bug first"},
    )

    assert _drain(cli._pending_input) == [
        "actually focus on the bug first",
        "please continue with the refactor",
    ]


def test_steer_only_when_no_interrupt_followup():
    cli = _make_cli_under_test()

    cli._enqueue_post_run_followups(
        pending_message=None,
        result={"pending_steer": "use rg not grep"},
    )

    assert _drain(cli._pending_input) == ["use rg not grep"]


def test_interrupt_only_when_no_leftover_steer():
    cli = _make_cli_under_test()

    cli._enqueue_post_run_followups(
        pending_message="follow-up question",
        result={},
    )

    assert _drain(cli._pending_input) == ["follow-up question"]


def test_no_result_no_pending_message_is_noop():
    cli = _make_cli_under_test()

    cli._enqueue_post_run_followups(pending_message=None, result=None)

    assert _drain(cli._pending_input) == []


def test_interrupt_followup_combines_with_extra_interrupt_queue_messages():
    cli = _make_cli_under_test()
    cli._interrupt_queue.put("second interrupt")
    cli._interrupt_queue.put("third interrupt")

    cli._enqueue_post_run_followups(
        pending_message="first interrupt",
        result={"pending_steer": "steer guidance"},
    )

    drained = _drain(cli._pending_input)
    assert drained[0] == "steer guidance"
    # Combined interrupt batch preserves arrival order within the batch.
    assert drained[1] == "first interrupt\nsecond interrupt\nthird interrupt"


def test_missing_pending_input_attribute_is_safe():
    """Method short-circuits when ``_pending_input`` was never initialized."""
    cli = HermesCLI.__new__(HermesCLI)
    # Deliberately no _pending_input and no _interrupt_queue.

    cli._enqueue_post_run_followups(
        pending_message="ignored",
        result={"pending_steer": "ignored"},
    )
    # No AttributeError raised → pass.
