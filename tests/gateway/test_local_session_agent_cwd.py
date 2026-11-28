"""A local (desktop/ACP) session's agent is built with the session's workspace as its cwd, so memory
providers, subdirectory hints and Codex see the same folder the session's terminal runs in."""
from types import SimpleNamespace

import pytest

LAUNCH = {'model': 'm', 'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1'}


class _Built(Exception):
    pass


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

    def resolve(route, platform_key, ephemeral, max_iterations, reasoning, pr):
        turn._build_fresh_agent(route, platform_key, ephemeral, max_iterations, reasoning, pr, False)
        raise _Built
    turn._resolve_turn_agent = resolve
    return turn


def _runner(tmp_path):
    from gateway.run import GatewayRunner
    runner = SimpleNamespace(
        _provider_routing={}, _peek_session_state=lambda key: None, _prefill_messages=None, _session_db=None,
        _refresh_fallback_model=lambda: None, session_authority=SimpleNamespace(profile_id=str(tmp_path)),
        _resolve_session_agent_runtime=lambda **kw: ('m', {'provider': 'custom'}))
    runner._resolve_turn_agent_config = lambda *a, **k: GatewayRunner._resolve_turn_agent_config(runner, *a, **k)
    return runner


def test_local_sessions_build_their_agent_in_their_own_workspace(tmp_path, monkeypatch):
    from gateway import session_policy
    from gateway.session_policy import build_policy
    monkeypatch.setattr('hermes_cli.models.resolve_fast_mode_overrides', lambda *a, **k: {})
    (tmp_path / 'repo-a').mkdir()
    (tmp_path / 'repo-b').mkdir()
    policies = {name: build_policy({'cwd': str(tmp_path / name), **LAUNCH}, {}) for name in ('repo-a', 'repo-b')}
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policies[source.chat_id])
    runner, built = _runner(tmp_path), {}
    for chat_id in policies:
        with pytest.raises(_Built):
            _turn(runner, chat_id, built)._run_sync_scoped()
    assert built['repo-a']['cwd'] == str(tmp_path / 'repo-a')
    assert built['repo-b']['cwd'] == str(tmp_path / 'repo-b')

