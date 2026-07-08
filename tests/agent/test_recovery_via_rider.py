"""Track A — re-init recovery announce: the (new turn) / (re-init) rider on
`_emit_fallback_announce`, and the SessionEntry.last_served_identity plumbing.

Background: two paths return a session to its primary model — a same-agent
restore (restore_primary_runtime, "new turn") and a re-init where the agent
cache was evicted/rebuilt so a fresh agent inits on the config default with no
fallback state ("re-init", previously a silent snap-back). Both now surface at
ONE unified announce site (gateway/run.py _run_agent_inner), keyed on the final
served route, distinguished by the recovery_via rider.

These tests cover the rider rendering + the no-op/route-tuple guards at the
`_emit_fallback_announce` level. The end-to-end gateway wiring (persist +
compare + emit) is exercised in tests/gateway/test_reinit_recovery_announce.py.
"""

import types

from agent.chat_completion_helpers import _emit_fallback_announce


def _agent():
    a = types.SimpleNamespace()
    a._announced = []
    a._emit_status = lambda m: a._announced.append(m)
    a._last_fallback_announced = None
    return a


def test_recovery_via_re_init_rider():
    a = _agent()
    _emit_fallback_announce(
        a, "claude-opus-4-8", "claude-fable-5", "yunwu",
        old_provider="claude-api-proxy-f3",
        announce_enabled=True, record_event=False,
        kind="recovery", recovery_via="re-init",
    )
    assert len(a._announced) == 1
    assert a._announced[0] == (
        "🔄 Model recovery (re-init): "
        "claude-api-proxy-f3/claude-opus-4-8 → yunwu/claude-fable-5"
    )


def test_recovery_via_new_turn_rider():
    a = _agent()
    _emit_fallback_announce(
        a, "claude-opus-4-8", "claude-fable-5", "yunwu",
        old_provider="claude-api-proxy-f3",
        announce_enabled=True, record_event=False,
        kind="recovery", recovery_via="new turn",
    )
    assert a._announced[0].startswith("🔄 Model recovery (new turn): ")


def test_recovery_without_via_has_no_rider_backcompat():
    """recovery_via=None (default) → no rider, so pre-existing callers that pass
    kind='recovery' with no recovery_via keep the bare 'Model recovery:' text."""
    a = _agent()
    _emit_fallback_announce(
        a, "claude-opus-4-8", "claude-fable-5", "yunwu",
        old_provider="claude-api-proxy-f3",
        announce_enabled=True, record_event=False,
        kind="recovery",
    )
    assert a._announced[0].startswith("🔄 Model recovery: ")
    assert "(" not in a._announced[0].split(":")[0]  # no rider before the colon


def test_failover_rider_unaffected_by_recovery_via():
    """A failover (kind='fallback') ignores recovery_via and keeps its reason
    rider path."""
    a = _agent()
    from agent.error_classifier import FailoverReason
    _emit_fallback_announce(
        a, "claude-fable-5", "claude-opus-4-8", "claude-api-proxy-f1",
        old_provider="yunwu",
        announce_enabled=True, record_event=True,
        kind="fallback", reason=FailoverReason.content_policy_blocked,
        recovery_via="re-init",  # must be ignored on the fallback path
    )
    assert a._announced[0].startswith("🔄 Model fallback (safety refusal): ")
    assert "re-init" not in a._announced[0]


def test_recovery_same_route_is_silent():
    """INV-3: a re-init that serves on the SAME (provider, model) it last served
    is a no-op — the route-tuple guard suppresses it."""
    a = _agent()
    _emit_fallback_announce(
        a, "claude-fable-5", "claude-fable-5", "yunwu",
        old_provider="yunwu",
        announce_enabled=True, record_event=False,
        kind="recovery", recovery_via="re-init",
    )
    assert a._announced == []


def test_recovery_gate_off_suppresses_chat_emit():
    """announce_enabled=False → no chat emit (the caller resolves this from
    model.announce_recovery). The durable sink is written separately, upstream."""
    a = _agent()
    _emit_fallback_announce(
        a, "claude-opus-4-8", "claude-fable-5", "yunwu",
        old_provider="claude-api-proxy-f3",
        announce_enabled=False, record_event=False,
        kind="recovery", recovery_via="re-init",
    )
    assert a._announced == []
