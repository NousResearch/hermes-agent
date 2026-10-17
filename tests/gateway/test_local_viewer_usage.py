"""``session.usage`` / ``session.stats``: the session-cumulative counters main's HermesCLI read off its
in-process agent (status bar, /usage, exit summary), published from the gateway (wave A2, CALLMAP W3)."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from gateway.config import Platform

FIELDS = {'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens', 'prompt_tokens',
          'completion_tokens', 'total_tokens', 'reasoning_tokens', 'api_calls', 'last_prompt_tokens',
          'context_tokens', 'context_length', 'config_context_length', 'context_source', 'compression_count',
          'avg_latency_s', 'tokens_per_s', 'last_turn_tokens', 'rate_limit', 'active_subagents', 'model',
          'provider', 'fallback_model'}


def _agent():
    compressor = SimpleNamespace(last_prompt_tokens=900, context_length=128000, _config_context_length=128000,
                                 compression_count=1)
    return SimpleNamespace(
        model='fake-model', provider='custom', session_id='sid',
        session_input_tokens=100, session_output_tokens=40, session_cache_read_tokens=10,
        session_cache_write_tokens=5, session_prompt_tokens=115, session_completion_tokens=40,
        session_total_tokens=155, session_reasoning_tokens=3, session_api_calls=2,
        context_compressor=compressor, _api_latency_history=deque([1.0, 3.0]),
        _api_output_history=deque([10, 30]), get_rate_limit_state=lambda: None,
        _session_messages=[], _turn_base_usage_anchor=None)


def test_usage_stats_is_exactly_the_frozen_field_list():
    from gateway.session_usage import usage_stats
    stats = usage_stats(_agent())
    assert set(stats) == FIELDS
    assert stats['input_tokens'] == 100 and stats['api_calls'] == 2 and stats['total_tokens'] == 155
    assert stats['avg_latency_s'] == 2.0 and stats['tokens_per_s'] == 10.0
    assert stats['context_length'] == 128000 and stats['compression_count'] == 1
    assert stats['rate_limit'] is None and stats['fallback_model'] is None
    from tui_gateway.contracts.registry import EVENTS
    EVENTS['session.usage'].payload.model_validate(stats)


def _turn(platform, published, agent):
    from gateway.run_turn_runner import TurnRunner
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(source=SimpleNamespace(platform=platform, chat_id='chat'), _step_callback_sync=None,
                                _run_still_current=lambda: True, agent_holder=[agent],
                                _hooks_ref=SimpleNamespace(loaded_hooks=[]))
    turn._runner = SimpleNamespace()
    turn._approval_owner = (SimpleNamespace(sessions={'s': SimpleNamespace()}), 's', 1)
    turn._publish_execution = lambda kind, payload: published.append((kind, payload)) or True
    return turn


@pytest.mark.parametrize('platform', [Platform.LOCAL, Platform.TELEGRAM])
def test_each_api_step_publishes_usage_only_for_local_viewers(platform):
    published = []
    turn = _turn(platform, published, _agent())
    turn._step_callback_sync(2, [])
    kinds = [kind for kind, _ in published]
    if platform == Platform.LOCAL:
        usage = next(payload for kind, payload in published if kind == 'session.usage')
        assert set(usage) == FIELDS | {'usage'} and usage['api_calls'] == 2
        assert usage['usage']['cache_hit_pct'] == 9 and usage['usage']['avg_tps'] == 10.0
    else:
        assert 'session.usage' not in kinds


@pytest.mark.asyncio
async def test_session_stats_returns_the_same_dict(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_surface_admission import _authority
    from tui_gateway.contracts.registry import CANONICAL_METHODS

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    agent = _agent()
    authority.runner._cached_agent_for = lambda route: agent
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        reply = await connection.dispatch({'id': 2, 'method': 'session.stats', 'params': {'session_id': 's'}})
        assert 'result' in reply, reply
        from gateway.session_usage import usage_stats
        assert reply['result'] == usage_stats(agent)
        CANONICAL_METHODS['session.stats'].result.model_validate(reply['result'])
    finally:
        await connection.close()


def test_completion_carries_main_tui_usage_from_the_turn_end_stats():
    """``message.complete.usage`` on a local-viewer turn is main's TUI ``Usage`` (``_get_usage``):
    context used/max/percent, ◷ latency, t/s, compressions — what the status bar paints."""
    from gateway.session_settlement_recovery import completion_payload
    from gateway.session_usage import usage_stats
    from tui_gateway.contracts.registry import EVENTS
    row = {'admission_id': 'a1', 'request_id': 'r1'}
    captured = {'result': {'final_response': 'hi'}, 'session_usage': usage_stats(_agent())}
    complete = completion_payload(row, {'outcome': 'completed'}, 'hi', captured)
    usage = complete['usage']
    assert usage['context_used'] == 900 and usage['context_max'] == 128000 and usage['context_percent'] == 1
    assert usage['avg_latency_s'] == 2.0 and usage['avg_tps'] == 10.0 and usage['compressions'] == 1
    assert (usage['input'], usage['output'], usage['total'], usage['calls']) == (100, 40, 155, 2)
    assert usage['cache_hit_pct'] == 9 and usage['model'] == 'fake-model'
    EVENTS['message.complete'].payload.model_validate({'text': 'hi', 'usage': usage})
    # No turn-end stats (messaging turns, recovered results): no usage key, as before.
    assert 'usage' not in completion_payload(row, {'outcome': 'completed'}, 'hi', {'result': {}})


def test_local_viewer_turn_end_records_stats_for_the_completion():
    from gateway.session_results import execution_result
    from gateway.session_usage import publish_usage
    published = []
    turn = _turn(Platform.LOCAL, published, _agent())
    captured = {}
    token = execution_result.set(captured)
    try:
        publish_usage(turn, _agent(), record=True)
    finally:
        execution_result.reset(token)
    assert captured['session_usage'] == {k: v for k, v in published[-1][1].items() if k != 'usage'}


def test_session_usage_frame_carries_the_projected_usage_object():
    """``session.usage`` publishes both shapes ``SessionUsagePayload`` declares: the frozen counters
    at the top level (the Ink TUI's ``usageFromTick``) *and* ``tui_usage(stats)`` under ``usage``,
    which the desktop merges into its status-bar ``UsageStats``. A flat-only frame left
    ``payload.usage`` undefined, so the desktop dropped every mid-turn tick and its cache-hit-rate /
    tokens-per-second chips sat on the "no figure" placeholder until a turn ended."""
    from gateway.session_usage import publish_usage, tui_usage, usage_stats
    from tui_gateway.contracts.registry import EVENTS
    published = []
    agent = _agent()
    publish_usage(_turn(Platform.LOCAL, published, agent), agent)
    kind, payload = published[-1]
    stats = usage_stats(agent)
    assert kind == 'session.usage'
    assert {k: v for k, v in payload.items() if k != 'usage'} == stats
    assert payload['usage'] == tui_usage(stats)
    assert payload['usage']['cache_hit_pct'] == 9 and payload['usage']['avg_tps'] == 10.0
    EVENTS['session.usage'].payload.model_validate(payload)


def test_turn_end_publishes_the_agents_banner_info():
    """Main re-emitted ``session.info`` at every turn end; the first turn's carries the built agent's
    ``system_prompt`` (the banner's ``▸ System Prompt — N chars`` row, FINDINGS row 2) and tools."""
    from gateway.session_local_inventory import publish_settled_info
    published = []
    agent = _agent()
    agent._cached_system_prompt = 'You are Hermes.' * 10
    agent.tools = [{'function': {'name': 'terminal'}}]
    turn = _turn(Platform.LOCAL, published, agent)
    turn._approval_owner = (SimpleNamespace(profile_id='p'), 's', 1)
    from gateway import session_local
    original = session_local.local_session_info
    session_local.local_session_info = lambda authority, ref: {'model': 'fake-model', 'lazy': False}
    try:
        publish_settled_info(turn, agent)
    finally:
        session_local.local_session_info = original
    kind, info = published[-1]
    assert kind == 'session.info'
    assert info['system_prompt'] == agent._cached_system_prompt
    assert info['tools'] == {'terminal': ['terminal']} and info['context_length'] == 128000
    assert info['model'] == 'fake-model'


@pytest.mark.asyncio
async def test_local_viewer_admission_settles_with_tui_usage_on_message_complete(tmp_path, monkeypatch):
    """End-to-end through ``session_ingress``: ``publish_usage(record=True)`` writes ``captured['session_usage']`` during the turn, and the admission packing carries it into ``pending_results`` so ``completion_payload`` attaches ``usage`` to ``message.complete``."""
    from gateway.session_controls import AuthorityConnection
    from gateway.session_usage import publish_usage
    from tests.gateway.test_surface_admission import _authority

    published = []
    agent = _agent()

    async def answer(event):
        turn = _turn(Platform.LOCAL, published, agent)
        publish_usage(turn, agent, record=True)
        return 'done'

    authority = await _authority(tmp_path, monkeypatch, answer)
    frames = []
    authority.sessions['s'].event_stream.observers.add(frames.append)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        reply = await connection.dispatch({'id': 2, 'method': 'prompt.submit', 'params': {
            'session_id': 's', 'submission_id': 'sub-1', 'text': 'hello'}})
        assert reply['result']['status'] == 'queued', reply
        await authority.sessions['s'].task
    finally:
        await connection.close()

    complete = next(f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete')
    usage = complete.get('usage')
    assert isinstance(usage, dict), complete
    assert usage['cache_hit_pct'] == 9 and usage['avg_tps'] == 10.0
    assert usage['context_used'] == 900 and usage['context_max'] == 128000 and usage['context_percent'] == 1
