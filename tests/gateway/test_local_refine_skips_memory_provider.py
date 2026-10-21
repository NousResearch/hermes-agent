"""A cold ``/refine`` on a local session never builds or starts an external memory provider.

The review side agent borrows the foreground session's id and is released with
``release_clients()``, which does not shut a memory provider down. ``_side_agent`` defaults to
``skip_memory=policy.ignore_rules`` (normally False), so the refine must ask for
``skip_memory=True``, as the background review fork does. The real refine body and the real
``_init_memory`` run; agent construction and review inference are stubbed.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_cold_refine_builds_no_memory_provider(tmp_path, monkeypatch):
    refine = pytest.importorskip('gateway.session_refine')
    from gateway import session_tui_side, session_policy, session_local_recovery
    from agent import agent_init, background_review, safe_worker_policy
    import plugins.memory
    from run_agent import AIAgent
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(safe_worker_policy, 'safe_worker_enabled', lambda: False)
    loaded = []

    class _Provider:
        name = 'fake'

        def is_available(self):
            return True

    def load_provider(name):
        loaded.append(name)
        return _Provider()
    monkeypatch.setattr(plugins.memory, 'load_memory_provider', load_provider)
    monkeypatch.setattr(agent_init, 'is_core_memory_provider', lambda name: False)
    sid = 'refine-memory-parent'
    source = SimpleNamespace(chat_id=sid)
    live = SimpleNamespace(source=source, review=SimpleNamespace())
    authority = SimpleNamespace(profile_id=str(tmp_path), sessions={sid: live},
        db=SimpleNamespace(get_messages_as_conversation=lambda *a, **k: [{'role': 'user', 'content': 'h'}]))
    policy = SimpleNamespace(ignore_rules=False)
    monkeypatch.setattr(refine, '_local_session', lambda r, s: (authority, live))
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda r, s: policy)
    monkeypatch.setattr(session_local_recovery, 'transcript_target', lambda a, r: sid)
    monkeypatch.setattr(session_tui_side, '_side_scope', lambda *a: nullcontext())
    built = []

    def side_agent(authority, policy, live, task_id, **overrides):
        agent = object.__new__(AIAgent)
        agent.session_id = overrides.get('session_id', task_id)
        agent.valid_tool_names = set()
        agent.enabled_toolsets = agent.disabled_toolsets = None
        # _side_agent's own default, then the real memory init AIAgent runs with it.
        skip = overrides.get('skip_memory', policy.ignore_rules)
        agent_init._init_memory(agent, {'memory': {'provider': 'fake'}}, skip, 'local')
        built.append(agent)
        return agent
    monkeypatch.setattr(session_tui_side, '_side_agent', side_agent)
    monkeypatch.setattr(background_review, 'spawn_background_review_thread', lambda *a, **k: (lambda: None, ''))
    monkeypatch.setattr(session_tui_side, '_spawn', lambda a, r, t, e, body, **kw: body())
    await refine.refine_local_session(object(), source, None)
    assert built, 'the refine body never built its side agent'
    assert loaded == [], f'the review side agent loaded a memory provider: {loaded}'
    assert built[0]._memory_manager is None
