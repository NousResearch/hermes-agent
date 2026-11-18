"""#51797: local side-agent construction keeps automatic memory isolation separate from tools."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("isolated", [False, True])
def test_side_agent_inherits_memory_prompt_isolation(tmp_path, monkeypatch, isolated):
    from gateway import session_tui_side
    from gateway.session_policy import build_policy
    import run_agent

    policy = build_policy({'cwd': str(tmp_path), 'model': 'fixture', 'provider': 'custom',
                           'base_url': 'http://127.0.0.1:9/v1', 'ignore_rules': isolated,
                           'toolsets': ['memory', 'file']}, {})
    runner = SimpleNamespace(_peek_session_state=lambda route: None)
    authority = SimpleNamespace(profile_id=str(tmp_path), runner=runner, db=None)
    live = SimpleNamespace(route='side-fixture')
    monkeypatch.setattr(session_tui_side, '_resolve_runtime', lambda *a: ('fixture', {}))
    monkeypatch.setattr('gateway.session_worker_construct.session_routing', lambda *a: {})
    captured = {}
    monkeypatch.setattr(run_agent, 'AIAgent', lambda **kw: captured.update(kw) or SimpleNamespace())
    session_tui_side._side_agent(authority, policy, live, 'side-test')
    assert captured['skip_memory_prompt'] is isolated
    assert captured['skip_memory'] is isolated
    assert 'memory' in captured['enabled_toolsets']
