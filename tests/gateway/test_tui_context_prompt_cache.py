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
    route = ('openrouter', 'https://openrouter.ai/api/v1', 'anthropic/claude-sonnet-4.6')
    event = CacheEvent(route=route, destination=route, route_class='openrouter_claude', requested_ttl='1h',
                       sent_at=1_000.0, read_tokens=0, write_tokens=4800, tier='1h', confidence='declared',
                       reason='openrouter_honours_ttl', coverage={'1h': 4_600.0}, guaranteed_until=4_600.0,
                       estimated_until=4_600.0, session_id=stored, generation=db.prompt_cache_generation(stored))
    db.update_token_counts(stored, cache_write_tokens=4800, api_call_count=1, prompt_cache_clock=event_to_json(event))
    result = await owner.call('session.context_breakdown', session_id=sid)
    cache = result['prompt_cache']
    assert (cache['tier'], cache['confidence'], cache['guaranteed_until']) == ('1h', 'declared', 4_600.0)
