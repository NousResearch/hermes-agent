"""A conversation its chat moved past keeps its own process / subagent projections."""
import pytest

from tests.gateway.test_authority_cold_route_recovery import _cold, _home, _source


async def _detached(tmp_path):
    cold = await _cold(tmp_path)
    old = cold.authority.register(_source())
    cold.store.get_or_create_session(_source(), force_new=True)  # the chat names a new conversation
    live = cold.authority.sessions[old.session_id]
    assert live.detached
    return cold, old, live


@pytest.mark.asyncio
async def test_a_detached_conversation_passes_the_current_route_check(tmp_path):
    from gateway.session_ancillary import _require_current_route
    cold, old, live = await _detached(tmp_path)
    with cold.store._lock:
        assert cold.store._entries.get(live.route) is None  # a detached key is never indexed
    _require_current_route(cold.authority, old)


@pytest.mark.asyncio
async def test_a_detached_conversation_on_another_conversations_key_is_still_stale(tmp_path):
    from gateway.session_ancillary import _require_current_route
    from gateway.session_detached import detached_key
    from hermes_state_runtime import RuntimeStoreError
    cold, old, live = await _detached(tmp_path)
    live.route = detached_key(live.route, 'some-other-conversation')
    with pytest.raises(RuntimeStoreError, match='stale_generation'):
        _require_current_route(cold.authority, old)


@pytest.mark.asyncio
async def test_a_detached_conversation_back_on_the_chat_route_is_still_stale(tmp_path):
    from gateway.session_ancillary import _require_current_route
    from hermes_state_runtime import RuntimeStoreError
    cold, old, live = await _detached(tmp_path)
    live.route = cold.store._generate_session_key(_source())  # now names the successor
    with pytest.raises(RuntimeStoreError, match='stale_generation'):
        _require_current_route(cold.authority, old)
