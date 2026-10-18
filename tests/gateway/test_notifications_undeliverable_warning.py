"""A completion that cannot be routed says so once, instead of deferring in silence.

``_classify_completion_target`` returns ``retry`` when the session route cannot be resolved. The
claim is then deferred and retried on every sweep, so the durable row stays pending for good and
the parent never learns the child finished. Nothing appeared in the log, which made a lost result
indistinguishable from a child that never ran.
"""

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
