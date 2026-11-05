import json
from dataclasses import replace

import pytest

from persistence import DelegationId, SessionId
from persistence.authority.coordination import DeliveryState, OwnerStamp
from tests.contracts.authority.test_bindings_api_contract import api_binding
from tests.contracts.authority.test_bindings_contract import begin, make_binding

OWNER = OwnerStamp(pid=1, started_at=1)


def _completed(coord, delegation_id: DelegationId, stamp: float) -> None:
    coord.record_delegation_dispatch(delegation_id=delegation_id, origin_session="peer", origin_ui_session_id="", parent_session_id=None,
                                     dispatched_at=stamp, owner=OWNER, task={"goal": "g"})
    coord.record_delegation_completion(delegation_id=delegation_id, state="completed", completed_at=stamp, event={"delegation_id": delegation_id}, result={})


def test_api_binding_initializes_prompt_fields(backend, scope_a):
    store_a = backend.open(scope_a)
    command = api_binding("api-prompt")
    store_a.bindings.bind_api(epoch=begin(store_a), binding=command)
    snapshot = store_a.sessions.get(command.session_id)
    assert snapshot.system_prompt is None
    assert snapshot.tool_pin is None
    store_a.sessions.update_system_prompt(command.session_id, "system")
    snapshot = store_a.sessions.get(command.session_id)
    assert snapshot.system_prompt == "system"
    assert snapshot.tool_pin is None


def test_local_binding_retains_routing_payload(backend, scope_a):
    store_a = backend.open(scope_a)
    command = make_binding("local-rich", route="rich")
    entry = {**command.entry, "display_name": "Named", "platform": "local", "chat_type": "group", "state": {"keep": 1}, "extra": {"nested": True}}
    command = replace(command, entry=entry)
    store_a.bindings.reserve(epoch=begin(store_a), binding=command, owner=None)
    saved = json.loads(store_a.routing.load_routing_entries()["rich"])
    # The entry text is the caller's entry as given, keys untouched.
    assert saved == entry
    entry["extra"]["nested"] = False
    again = json.loads(store_a.routing.load_routing_entries()["rich"])
    assert again["extra"] == {"nested": True}


def test_delivery_claim_reclaims_after_legacy_timeout(backend, scope_a, monkeypatch):
    now = [1700002000.0]
    monkeypatch.setattr("time.time", lambda: now[0])
    coord = backend.open(scope_a).coordination
    delegation_id = DelegationId("stale")
    _completed(coord, delegation_id, now[0])
    assert coord.claim_delivery(delegation_id=delegation_id, claim_id="one") is True
    now[0] += 300
    assert coord.claim_delivery(delegation_id=delegation_id, claim_id="two") is False
    now[0] += 1
    assert coord.claim_delivery(delegation_id=delegation_id, claim_id="two") is True
    assert not coord.release_delivery(delegation_id=delegation_id, claim_id="one")
    assert coord.complete_delivery(delegation_id=delegation_id, claim_id="two")
    saved = coord.get_delegation(delegation_id=delegation_id)
    assert saved.delivery_attempts == 2
    assert saved.delivery_state is DeliveryState.DELIVERED


def test_delivery_release_stops_at_legacy_attempt_limit(backend, scope_a):
    coord = backend.open(scope_a).coordination
    delegation_id = DelegationId("cap")
    _completed(coord, delegation_id, 1767225600.0)
    for attempt in range(1, 9):
        assert coord.claim_delivery(delegation_id=delegation_id, claim_id=f"c{attempt}") is True
        assert not coord.release_delivery(delegation_id=delegation_id, claim_id="wrong")
        assert coord.release_delivery(delegation_id=delegation_id, claim_id=f"c{attempt}")
        saved = coord.get_delegation(delegation_id=delegation_id)
        assert saved.delivery_attempts == attempt
        assert saved.delivery_state is (DeliveryState.DROPPED if attempt == 8 else DeliveryState.PENDING)
    assert coord.claim_delivery(delegation_id=delegation_id, claim_id="late") is False


@pytest.mark.parametrize("error", ["short", "e" * 600])
def test_handoff_errors_match_legacy_limit(backend, scope_a, error):
    store_a = backend.open(scope_a)
    from persistence.conversation.sessions import NewSession

    session_id = SessionId("handoff")
    store_a.sessions.create(NewSession(session_id=session_id, source="cli"))
    coord = store_a.coordination
    assert coord.request_handoff(session_id=session_id, platform="local")
    assert coord.fail_handoff(session_id=session_id, error=error)
    assert (coord.get_handoff(session_id=session_id)).error == error[:500]
    assert coord.request_handoff(session_id=session_id, platform="local")
    assert coord.claim_handoff(session_id=session_id)
    assert coord.reclaim_stale_running_handoffs(error=error) == (session_id,)
    assert (coord.get_handoff(session_id=session_id)).error == error[:500]
