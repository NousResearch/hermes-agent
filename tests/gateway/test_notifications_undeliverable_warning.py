"""A completion that cannot be routed says so once, instead of deferring in silence.

``_classify_completion_target`` returns ``retry`` when the session route cannot be resolved. The
claim is then deferred and retried on every sweep, so the durable row stays pending for good and
the parent never learns the child finished. Nothing appeared in the log, which made a lost result
indistinguishable from a child that never ran.
"""

import asyncio
import logging

from gateway.run_notifications import GatewayNotificationsMixin


def _event(delegation_id):
    return {"type": "async_delegation", "delegation_id": delegation_id, "parent_session_id": "agent-1"}


def test_a_retryable_completion_warns_once_with_the_recovery_hint(caplog):
    caplog.set_level(logging.WARNING, logger="gateway.run")

    GatewayNotificationsMixin._note_undeliverable(_event("deleg-warned"), "retry")
    GatewayNotificationsMixin._note_undeliverable(_event("deleg-warned"), "retry")

    warnings = [r for r in caplog.records if "cannot be routed" in r.getMessage()]
    assert len(warnings) == 1, "one line per delegation, not one per sweep"
    message = warnings[0].getMessage()
    assert "deleg-warned" in message and "retry" in message
    assert "pending_completions" in message, "the log has to say how to get the result back"


def test_an_event_without_a_delegation_id_does_not_warn(caplog):
    caplog.set_level(logging.WARNING, logger="gateway.run")

    GatewayNotificationsMixin._note_undeliverable({"type": "completion"}, "retry")

    assert not [r for r in caplog.records if "cannot be routed" in r.getMessage()]


class _Routing(GatewayNotificationsMixin):
    """A host with just the classifier. Neither non-deliver verdict reaches the adapter path."""

    def __init__(self, verdict):
        self.verdict = verdict

    async def _classify_completion_target(self, parent_session_id, session_key=""):
        return self.verdict


def test_a_terminal_target_is_not_described_as_retrying(caplog):
    """terminal is dropped on this same pass by the pre-flight, which logs its own line. Saying
    "deferring and retrying" there asserted the opposite of what the code does."""
    caplog.set_level(logging.WARNING, logger="gateway.run")

    assert asyncio.run(_Routing("terminal")._completion_delivery_ready(_event("deleg-terminal"))) is True
    assert not [r for r in caplog.records if "deferring and retrying" in r.getMessage()]


def test_a_retryable_target_still_defers_and_warns_once(caplog):
    caplog.set_level(logging.WARNING, logger="gateway.run")
    host = _Routing("retry")

    assert asyncio.run(host._completion_delivery_ready(_event("deleg-retry"))) is False
    assert asyncio.run(host._completion_delivery_ready(_event("deleg-retry"))) is False

    warnings = [r for r in caplog.records if "cannot be routed" in r.getMessage()]
    assert len(warnings) == 1, "one line per delegation, not one per sweep"
    assert "deferring and retrying" in warnings[0].getMessage()
