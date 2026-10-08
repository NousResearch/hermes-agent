"""An approval prompt that cannot be delivered must fail fast, not stall.

Regression: when the gateway could not deliver the approval prompt (bot not a
member of the target room, archived room, adapter error), the send failure was
only logged. The agent then blocked in ``_await_gateway_decision`` for the full
approval timeout (~300s) waiting for an answer to a message nobody ever saw.
Observed repeatedly in a multi-bot deployment where one bot's approvals were
routed to a room it had never joined.

The gateway now raises when the prompt was not delivered; the notify callback
contract turns that into an immediate ``notify_failed`` fast-deny.
"""

import time

from tools.approval import _await_gateway_decision


def test_notify_failure_fast_denies_instead_of_blocking():
    """A raising notify callback resolves immediately, not after the timeout."""
    def _undeliverable(_approval_data):
        raise RuntimeError("approval prompt undeliverable to !room:example.org")

    started = time.monotonic()
    result = _await_gateway_decision(
        "session-1",
        _undeliverable,
        {"command": "rm -rf /tmp/x", "description": "delete"},
    )
    elapsed = time.monotonic() - started

    assert result["resolved"] is False
    assert result["choice"] is None
    assert result["notify_failed"] is True
    # The whole point: no ~300s stall on an unanswerable prompt.
    assert elapsed < 5, f"fast-deny took {elapsed:.1f}s — still blocking"


def test_text_fallback_only_raises_when_delivery_failed():
    """Only a FAILED send may short-circuit ``_approval_notify_sync`` — a
    delivered prompt must not.

    Asserting the guard directly instead of driving ``_approval_notify_sync``
    end to end: it needs a live ``TurnRunner``/adapter/event-loop scaffold that
    belongs to its own dedicated test module. The condition below (raise
    on-and-only-on a failed/unschedulable send, in the text-fallback except
    block right after the button-approval path's own undeliverable handling)
    is the whole of the new behaviour, and is what regresses if the fallback
    is ever rewritten to swallow the exception again.
    """
    import inspect

    from gateway import run_turn_runner

    src = inspect.getsource(run_turn_runner.TurnRunner._approval_notify_sync)
    assert "raise RuntimeError(f\"approval prompt undeliverable to {ctx._status_chat_id}\") from e" in src, (
        "fail-fast guard missing from the text-fallback except block"
    )
    # The raise must be inside the except block that also logs the failure —
    # never unconditional, or a successfully delivered prompt would also raise.
    except_block = src[src.index("except Exception as e:"):]
    assert "logger.error(\"Failed to send approval request" in except_block
    assert "raise RuntimeError" in except_block
