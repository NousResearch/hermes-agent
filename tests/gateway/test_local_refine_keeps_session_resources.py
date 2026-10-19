"""A cold ``/refine`` on a local session leaves the session's tool resources alone (T7).

The review side agent borrows the session's id so its usage is attributed to the session. Closing it
would run the session-id-keyed task cleanup (terminal sandbox, browser) of the foreground session, so
it releases only its own clients. Real refine body, AIAgent cleanup, terminal registry and
LocalEnvironment; review inference and agent construction are stubbed.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_cold_refine_keeps_the_sessions_terminal(tmp_path, monkeypatch):
    refine = pytest.importorskip('gateway.session_refine')
    from gateway import session_tui_side, session_policy, session_local_recovery
    from agent import background_review
    from run_agent import AIAgent
    from tools.environments.local import LocalEnvironment
    from tools import terminal_tool
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setenv('TERMINAL_TEMP_DIR', str(tmp_path))
    sid = 'independent-refine-parent'
    env = LocalEnvironment(cwd=str(tmp_path), persistent=True)
    first = env.execute('export INDEPENDENT_REFINE_SENTINEL=kept; printf ready')
    assert first['returncode'] == 0, first
    before = env.execute('printf %s "$INDEPENDENT_REFINE_SENTINEL"')['output']
    assert 'kept' in before, before
    # Per-task isolation uses the session id (as nonpersistent Docker also does),
    # unlike ordinary local environments whose key is session:<route>.
    terminal_tool.register_task_env_overrides(sid, {'env_type': 'local'})
    key = terminal_tool._resolve_container_task_id(sid)
    assert key == sid, key
    terminal_tool._active_environments[key] = env
    source = SimpleNamespace(chat_id=sid)
    live = SimpleNamespace(source=source, review=SimpleNamespace())
    authority = SimpleNamespace(profile_id=str(tmp_path), sessions={sid: live},
        db=SimpleNamespace(get_messages_as_conversation=lambda *a, **k: [{'role': 'user', 'content': 'history'}]))
    policy = object()
    # Isolate inference/authority construction only. The production refine body,
    # AIAgent.close, task-resource cleanup, registry and LocalEnvironment are real.
    monkeypatch.setattr(refine, '_local_session', lambda r, s: (authority, live))
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda r, s: policy)
    monkeypatch.setattr(session_local_recovery, 'transcript_target', lambda a, r: sid)
    monkeypatch.setattr(session_tui_side, '_side_scope', lambda *a: nullcontext())
    def side_agent(*args, **kwargs):
        agent = object.__new__(AIAgent)
        agent.session_id = kwargs.get('session_id', args[3])
        agent.valid_tool_names = set()
        return agent
    monkeypatch.setattr(session_tui_side, '_side_agent', side_agent)
    monkeypatch.setattr(background_review, 'spawn_background_review_thread', lambda *a, **k: (lambda: None, ''))
    monkeypatch.setattr(session_tui_side, '_spawn', lambda a, r, t, e, body, **kw: body())
    try:
        reply = await refine.refine_local_session(object(), source, 'preserve my terminal')
        retained = sid in terminal_tool._active_environments
        after = env.execute('printf %s "${INDEPENDENT_REFINE_SENTINEL-missing}"')['output']
        print('REFINE_EVIDENCE', {'reply': reply, 'before': before, 'after': after, 'registered': retained})
        assert retained and 'kept' in after, 'Refine closed the foreground terminal and deleted its shell snapshot'
    finally:
        terminal_tool._active_environments.pop(sid, None)
        terminal_tool.clear_task_env_overrides(sid)
        env.cleanup()
