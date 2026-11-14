"""One GatewayRunner serves every concurrent turn (all profiles under ``multiplex_profiles``), so a
turn's service tier must reach its own agent even when another turn resolves its tier meanwhile."""
import threading
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
                                AIAgent=lambda **kw: built.update({chat_id: kw}) or SimpleNamespace())
    turn._combined_ephemeral_prompt = lambda: None
    turn._setup_stream_consumer = lambda platform_key: (None, None, None, False)
    return turn


def test_overlapping_turns_each_build_their_agent_with_their_own_service_tier(tmp_path, monkeypatch):
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
