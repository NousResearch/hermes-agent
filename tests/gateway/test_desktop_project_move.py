"""``desktop_project`` create during a Desktop chat served by the session authority moves the chat like
the user's Move to project: the next ``pwd`` in the same turn runs in the project folder, and the turn
boundary commits the session's ``workspace`` mutation (its policy cwd). Before the gateway wired
``set_project_workspace_callback`` the tool reported success while the chat stayed put."""
import json
from types import SimpleNamespace

import pytest

from gateway import local_desktop_ui
from tests.gateway.test_session_resume_config_refresh import (
    _Backend, _call, _connect, _home, _owner_authority, _policy)


@pytest.mark.asyncio
async def test_desktop_project_create_in_a_local_viewer_turn_moves_the_session(tmp_path, monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars
    from gateway.session_contract import SessionRef
    from gateway.session_mutation_policy import apply_pending_workspace
    from gateway.session_policy import policy_scope, restore_policy
    from hermes_cli import projects_db as pdb
    from tools import project_tools, terminal_tool
    built = []
    monkeypatch.setattr(terminal_tool, '_active_environments', {})
    monkeypatch.setattr(terminal_tool, '_session_cwd', {})
    monkeypatch.setattr(terminal_tool, '_pre_exec_block', lambda *a, **k: None)
    monkeypatch.setattr(terminal_tool, '_create_configured_env',
                        lambda config, env_type, **k: built.append(_Backend(env_type, k['cwd'])) or built[-1])
    monkeypatch.setattr(local_desktop_ui, '_turns', {})
    monkeypatch.setattr(local_desktop_ui, '_workspace_routes', {})
    monkeypatch.setattr(local_desktop_ui, '_process_routes', {})
    monkeypatch.setattr(project_tools, '_workspace_callback', None)
    db_path, connect_closing = tmp_path / 'projects.db', pdb.connect_closing
    monkeypatch.setattr(pdb, 'connect_closing', lambda: connect_closing(db_path=db_path))
    home = _home(monkeypatch)
    start, worktree = tmp_path / 'start', tmp_path / 'worktree'
    start.mkdir(), worktree.mkdir()
    store, authority = await _owner_authority(home)
    owner = _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='t', source='gui', cwd=str(start)))['session_id']
        route = authority.sessions[sid].route
        turn = SimpleNamespace(_approval_owner=(authority, sid, 1), _publish_execution=lambda *a: True,
                               _ctx=SimpleNamespace(session_id=sid, session_key=route))
        token = local_desktop_ui.wire(turn, SimpleNamespace())
        tokens = set_session_vars(platform='local', session_key=route, session_id=sid)
        try:
            with policy_scope(_policy(authority, sid), authority=authority):
                created = json.loads(project_tools.project_create('Worktree', path=str(worktree), task_id=sid))
                json.loads(terminal_tool.terminal_tool('pwd', task_id=sid))
        finally:
            clear_session_vars(tokens)
            local_desktop_ui.release(token)
        assert created['success'] is True and created['moved'] is True, created
        assert built[-1].ran[-1] == str(worktree), [(b.cwd, b.ran) for b in built]
        with connect_closing(db_path=db_path) as conn:
            assert pdb.get_active_id(conn) is None  # a moved chat leaves the profile-global pointer alone
        assert authority.sessions[sid].pending_workspace == str(worktree)

        await apply_pending_workspace(authority, SessionRef(authority.profile_id, sid), authority.sessions[sid])
        stored = restore_policy(authority.persistence.bindings.receipt(sid).policy)
        assert stored.cwd == str(worktree)
        assert json.loads(stored.terminal_json)['TERMINAL_CWD'] == str(worktree)
        assert _policy(authority, sid).cwd == str(worktree)
        assert authority.sessions[sid].pending_workspace is None
    finally:
        await owner.close()
        store.close_all_db_handles()


def test_a_task_no_turn_of_ours_owns_goes_to_the_wrapped_callback(monkeypatch):
    from tools import project_tools
    monkeypatch.setattr(local_desktop_ui, '_workspace_routes', {})
    seen = []
    monkeypatch.setattr(project_tools, '_workspace_callback', lambda *args: seen.append(args) or True)
    local_desktop_ui._install_workspace_hook()
    local_desktop_ui._install_workspace_hook()  # idempotent: never wraps itself
    assert project_tools.get_project_workspace_callback()('tui-sid', '/p', 'P') is True
    assert seen == [('tui-sid', '/p', 'P')]
