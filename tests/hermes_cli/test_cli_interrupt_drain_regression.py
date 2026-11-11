"""Regression test for #20271: classic-CLI hangs when messages typed during
an agent turn never leave ``_interrupt_queue``.

Background
----------
The CLI routes user input typed while ``_agent_running`` is True into
``_interrupt_queue`` (separate from ``_pending_input``) so that the explicit
interrupt path can opt to deliver them as a single combined "interrupt"
message. The explicit drain at the top of ``process_loop`` only fires when
``busy_input_mode == "interrupt"`` AND a ``pending_message`` was
acknowledged.

The original PR #17939 paired the paste-file TOCTOU fix with a separate
drain inside ``process_loop``'s ``finally`` block: any message left in
``_interrupt_queue`` after the agent's turn ends gets re-queued onto
``_pending_input``. The drain was split off in #17666 / #18760 as "worth
its own review" and never re-landed. v0.12.0 users hit a hang when typing
during a turn that completes naturally — the message sits in
``_interrupt_queue``, the next ``Enter`` re-routes input to the same
blocked queue, and the CLI looks frozen.

This test exercises the restored ``_drain_interrupt_queue_to_pending_input``
helper that ``process_loop`` now calls every turn. The integration into
``process_loop`` itself is not threaded here (it requires a real
prompt_toolkit app); the helper is unit-testable on its own and is the
load-bearing piece.
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock


def _make_cli():
    """Bare HermesCLI (no __init__) with just the two queues the drain touches."""
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._interrupt_queue = queue.Queue()
    cli._pending_input = queue.Queue()
    return cli


class TestInterruptQueueDrain:
    """``_drain_interrupt_queue_to_pending_input`` re-queues stray messages."""


    def test_preserves_order_when_draining_multiple_messages(self):
        cli = _make_cli()
        for msg in ("first", "second", "third"):
            cli._interrupt_queue.put(msg)

        cli._drain_interrupt_queue_to_pending_input()

        assert cli._interrupt_queue.empty()
        drained = []
        while not cli._pending_input.empty():
            drained.append(cli._pending_input.get_nowait())
        assert drained == ["first", "second", "third"]


    def test_skips_falsy_messages(self):
        cli = _make_cli()
        cli._interrupt_queue.put("")
        cli._interrupt_queue.put(None)
        cli._interrupt_queue.put("real")

        cli._drain_interrupt_queue_to_pending_input()

        assert cli._interrupt_queue.empty()
        assert cli._pending_input.qsize() == 1
        assert cli._pending_input.get_nowait() == "real"

    def test_swallows_exceptions_so_main_loop_never_breaks(self):
        cli = _make_cli()
        # Replace _pending_input with an object whose .put raises — simulating
        # an unexpected internal error. The drain must NOT propagate.
        broken = MagicMock(spec=queue.Queue)
        broken.put.side_effect = RuntimeError("simulated put failure")
        cli._pending_input = broken
        cli._interrupt_queue.put("anything")

        # Should not raise.
        cli._drain_interrupt_queue_to_pending_input()
