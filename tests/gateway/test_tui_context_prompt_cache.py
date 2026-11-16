"""An agentless ``session.context_breakdown`` (gateway restarted, agent evicted) still reports the
prompt-cache window from the session's saved clock, without a model call."""
import pytest

from tests.gateway.test_tui_session_reads import owner as reads_owner

tui_owner = reads_owner  # the TUI-reads owner fixture, under a name the gateway conftest's `owner` does not shadow


@pytest.mark.asyncio
async def test_agentless_context_breakdown_reads_the_saved_cache_clock(tui_owner):
    owner = tui_owner
    from types import SimpleNamespace

    from agent.prompt_cache_clock import CacheEvent, event_to_json
    from gateway.session_tui_reads import _stored_id
    sid = (await owner.call('session.create', request_id='p', source='cli', cwd=str(owner.workdir)))['session_id']
    stored = _stored_id(owner.authority, SimpleNamespace(session_id=sid))
    db = owner.authority.db
    route = ('custom', 'http://127.0.0.1:9/v1', 'fake-model')  # the route the session row persists
    event = CacheEvent(route=route, destination=route, route_class='openrouter_claude', requested_ttl='1h',
                       sent_at=1_000.0, read_tokens=0, write_tokens=4800, tier='1h', confidence='declared',
                       reason='openrouter_honours_ttl', coverage={'1h': 4_600.0}, guaranteed_until=4_600.0,
                       estimated_until=4_600.0, session_id=stored, generation=db.prompt_cache_generation(stored))
    db.update_token_counts(stored, cache_write_tokens=4800, api_call_count=1, prompt_cache_clock=event_to_json(event))
    result = await owner.call('session.context_breakdown', session_id=sid)
    cache = result['prompt_cache']
    assert (cache['tier'], cache['confidence'], cache['guaranteed_until']) == ('1h', 'declared', 4_600.0)



@pytest.mark.asyncio
async def test_agentless_context_reports_unknown_after_a_model_switch(tui_owner, monkeypatch):
    """session.mutate switches the model with no agent resident: the saved clock was recorded on the old
    model's cache, so the agentless breakdown must report unknown (and not serve its cached payload)."""
    from types import SimpleNamespace

    from agent.prompt_cache_clock import CacheEvent, event_to_json
    from gateway.session_tui_reads import _stored_id
    from hermes_cli import model_switch
    owner = tui_owner
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model='switched', target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    owner.authority.runner._resolve_session_agent_runtime = lambda **k: ('fake-model', {})
    owner.authority.runner._evict_cached_agent = lambda route: None
    sid = (await owner.call('session.create', request_id='p', source='cli', cwd=str(owner.workdir)))['session_id']
    stored = _stored_id(owner.authority, SimpleNamespace(session_id=sid))
    db = owner.authority.db
    route = ('custom', 'http://127.0.0.1:9/v1', 'fake-model')
    event = CacheEvent(route=route, destination=route, route_class='anthropic_gateway', requested_ttl='5m',
                       sent_at=1_000.0, read_tokens=0, write_tokens=4800, tier='5m', confidence='declared',
                       reason='gateway_requested_5m', coverage={'5m': 1_300.0}, guaranteed_until=1_300.0,
                       estimated_until=1_300.0, session_id=stored, generation=db.prompt_cache_generation(stored))
    db.update_token_counts(stored, cache_write_tokens=4800, api_call_count=1, prompt_cache_clock=event_to_json(event))
    before = (await owner.call('session.context_breakdown', session_id=sid))['prompt_cache']
    assert before['confidence'] == 'declared'
    snap = db.get_session(stored)
    await owner.call('session.mutate', session_id=sid, request_id='switch', expected_revision=snap['runtime_revision'],
                     expected_generation=snap['runtime_generation'], operation='model', payload={'model': 'switched'})
    after = (await owner.call('session.context_breakdown', session_id=sid))['prompt_cache']
    assert (after['confidence'], after['reason']) == ('unknown', 'route_changed')
