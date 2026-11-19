"""One GatewayRunner serves every concurrent turn (all profiles under ``multiplex_profiles``), so a
turn's service tier must reach its own agent even when another turn resolves its tier meanwhile."""
import threading
from types import SimpleNamespace

import pytest

LAUNCH = {'model': 'm', 'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1'}


class _Built(Exception):
    pass


class _ReusedAgent:
    """A cached agent the turn reuses: records the tier its rewire sets, then stops the turn."""

    def __init__(self, record):
        object.__setattr__(self, '_record', record)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == 'service_tier':
            self._record(value)
            raise _Built


def _turn(runner, chat_id, built):
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    turn = TurnRunner.__new__(TurnRunner)
    turn._runner = runner
    source = SimpleNamespace(platform=Platform.LOCAL, chat_id=chat_id, user_id='u', user_id_alt=None,
                             user_name=None, chat_name=None, chat_type='dm', thread_id=None)
    turn._ctx = SimpleNamespace(source=source, session_key=chat_id, session_id=chat_id, user_config={},
                                message='hi', inbound_message_id=None, enabled_toolsets=[], disabled_toolsets=None,
                                AIAgent=lambda **kw: built.update({chat_id: kw}) or SimpleNamespace(),
                                progress_callback=None, _voice_ack_guild=[None], _native_slack_task_cards=False,
                                _hooks_ref=SimpleNamespace(loaded_hooks=[]), _status_callback_sync=None,
                                _event_callback_sync=None)
    turn._combined_ephemeral_prompt = lambda: None
    turn._setup_stream_consumer = lambda platform_key: (None, None, None, False)
    return turn


@pytest.mark.parametrize('reused', [False, True], ids=['fresh-agent', 'reused-agent'])
def test_overlapping_turns_each_build_their_agent_with_their_own_service_tier(tmp_path, monkeypatch, reused):
    from gateway import session_policy
    from gateway.run import GatewayRunner
    from gateway.session_policy import build_policy
    monkeypatch.setattr('hermes_cli.models.resolve_fast_mode_overrides', lambda *a, **k: {})
    policies = {
        'fast': build_policy({'cwd': str(tmp_path), **LAUNCH}, {'agent': {'service_tier': 'fast'}}),
        'normal': build_policy({'cwd': str(tmp_path), **LAUNCH}, {}),
    }
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policies[source.chat_id])
    runner = SimpleNamespace(
        _provider_routing={}, _peek_session_state=lambda key: None, _prefill_messages=None, _session_db=None,
        _refresh_fallback_model=lambda: None, session_authority=SimpleNamespace(profile_id=str(tmp_path)),
        _resolve_session_agent_runtime=lambda **kw: ('m', {'provider': 'custom'}))
    runner._resolve_turn_agent_config = lambda *a, **k: GatewayRunner._resolve_turn_agent_config(runner, *a, **k)
    built, resolved, release = {}, threading.Event(), threading.Event()

    def make(chat_id, gate):
        turn = _turn(runner, chat_id, built)

        def resolve(route, platform_key, ephemeral, max_iterations, reasoning, pr):
            if gate:
                resolved.set()
                assert release.wait(10)
            if reused:  # the rewire (_wire_turn_agent_callbacks) sets the tier on a cached agent
                return _ReusedAgent(lambda tier: built.update({chat_id: {'service_tier': tier}})), True
            turn._build_fresh_agent(route, platform_key, ephemeral, max_iterations, reasoning, pr, False)
            raise _Built
        turn._resolve_turn_agent = resolve
        return turn

    errors = []

    def run_fast():
        try:
            make('fast', True)._run_sync_scoped()
        except _Built:
            pass
        except BaseException as exc:  # surfaced below
            errors.append(exc)
    worker = threading.Thread(target=run_fast)
    worker.start()
    assert resolved.wait(10)
    with pytest.raises(_Built):
        make('normal', False)._run_sync_scoped()
    release.set()
    worker.join(10)
    assert not errors
    assert built['normal']['service_tier'] is None
    assert built['fast']['service_tier'] == 'priority'


@pytest.mark.asyncio
async def test_background_task_builds_its_agent_with_its_own_sessions_service_tier():
    """/bg reads its session's tier, never the runner slot another session's /fast last wrote."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from gateway.config import Platform
    from gateway.hooks import HookRegistry
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    runner = object.__new__(GatewayRunner)
    runner.adapters, runner._voice_mode, runner._session_db, runner._reasoning_config = {}, {}, None, None
    runner._provider_routing, runner._fallback_model, runner._running_agents = {}, None, {}
    runner._background_tasks, runner.hooks = set(), HookRegistry()
    runner.session_store = MagicMock(**{'get_model_override.return_value': None})
    runner._service_tier = 'priority'  # another session's /fast left the shared slot here
    runner._resolve_session_service_tier = lambda **kw: None
    adapter = AsyncMock(extract_media=lambda text: ([], text), extract_images=lambda text: ([], text),
                        toolsets_for_source=MagicMock(return_value=None))
    runner.adapters[Platform.TELEGRAM] = adapter
    source = SessionSource(platform=Platform.TELEGRAM, user_id='u', chat_id='c', user_name='n')
    with patch('gateway.run._resolve_runtime_agent_kwargs', return_value={'api_key': 'k'}), \
         patch('gateway.run._load_gateway_config', return_value={}), \
         patch('run_agent.AIAgent') as agent_cls:
        agent_cls.return_value.run_conversation.return_value = {'final_response': 'ok', 'messages': []}
        await runner._run_background_task('hi', source, 'bg_tier')
    assert agent_cls.call_args.kwargs['service_tier'] is None
