"""A hosted room member switched by a profile edit keeps its cached agent after a successful turn.

The member resolves its route from the owning profile only inside the turn's policy scope; the
post-turn fallback check runs after that scope exits. Reading the frozen model there made the new
route look like a provider fallback, so every reply after a switch rebuilt the agent.
"""
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock


def _write_profile(home, model):
    (home / 'config.yaml').write_text(json.dumps({'model': {
        'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1', 'default': model}}), encoding='utf-8')


def test_switched_room_member_is_not_evicted_as_a_fallback(tmp_path, monkeypatch):
    from gateway import session_policy
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext

    frozen = {'model': {'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1', 'default': 'old-model'}}
    policy = replace(session_policy.build_policy({'source': 'gui', 'cwd': str(tmp_path), 'model': 'old-model'}, frozen),
                     source='bot_room', platform='bot_room')
    _write_profile(tmp_path, 'new-model')
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._session_model_overrides = {}
    runner.session_authority = SimpleNamespace(db=SimpleNamespace(db_path=tmp_path / 'state.db'))  # type: ignore[attr-defined]
    runner._evict_cached_agent = MagicMock()
    source = SessionSource(platform=Platform.LOCAL, user_id='u', chat_id='member', chat_type='dm')
    ctx = TurnContext(source=source, session_key='agent:main:local:dm:member')
    ctx.agent_holder[0] = SimpleNamespace(model='new-model', provider='custom')
    ctx.result_holder[0] = {'final_response': 'ok'}

    runner._run_agent_evict_on_fallback(ctx)
    assert not runner._evict_cached_agent.called

    # A real fallback off the profile's current model still evicts.
    ctx.agent_holder[0] = SimpleNamespace(model='fallback-model', provider='custom')
    runner._run_agent_evict_on_fallback(ctx)
    assert runner._evict_cached_agent.called
