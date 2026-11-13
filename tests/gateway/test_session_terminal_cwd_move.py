"""A workspace move clears the saved terminal cwd too (#41128 follow-up): the next command after a
move runs in the new folder even when the session had ``cd``-ed before it."""
import pytest

from hermes_agent.gateway.session_terminal_cwd import persist_turn_cwd, restore_turn_cwd
from tests.gateway.test_session_resume_config_refresh import (
    _Backend, _call, _connect, _home, _mutate, _owner_authority, _policy, _terminal)


@pytest.mark.asyncio
async def test_a_workspace_move_after_a_cd_runs_the_next_command_in_the_new_folder(tmp_path, monkeypatch):
    from hermes_agent.tools import terminal_tool
    built = []
    monkeypatch.setattr(terminal_tool, '_active_environments', {})
    monkeypatch.setattr(terminal_tool, '_session_cwd', {})
    monkeypatch.setattr(terminal_tool, '_pre_exec_block', lambda *a, **k: None)
    monkeypatch.setattr(terminal_tool, '_create_configured_env',
                        lambda config, env_type, **k: built.append(_Backend(env_type, k['cwd'])) or built[-1])
    home = _home(monkeypatch)
    sub, moved = tmp_path / 'sub', tmp_path / 'tree'
    sub.mkdir(), moved.mkdir()
    store, authority = await _owner_authority(home)
    owner = _connect(authority)

    def turn(sid, command):  # TurnRunner.run_sync's order around the turn
        route, policy = authority.sessions[sid].route, _policy(authority, sid)
        start = restore_turn_cwd(authority.runner, route, policy)
        try:
            _terminal(authority, sid, command)
        finally:
            persist_turn_cwd(authority.runner, route, start)
    try:
        sid = (await _call(owner, 'session.create', request_id='t', source='cli', cwd=str(tmp_path)))['session_id']
        turn(sid, f'cd {sub}')
        await _mutate(owner, store, sid, 'move', 'workspace', {'cwd': str(moved)})
        turn(sid, 'pwd')
        assert built[-1].ran[-1] == str(moved), [(backend.cwd, backend.ran) for backend in built]
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_refresh_whose_cwd_save_fails_still_publishes_its_committed_policy(tmp_path, monkeypatch):
    """The refresh commits the new policy, then drops the session's saved cwd. A routing store that
    refuses that save (here: state.db rejects it and the sessions.json fallback is a directory) must
    not leave the live policy behind the committed one: restore refuses that pair, and a retry finds
    nothing moved and never publishes."""
    import json
    from hermes_agent.gateway.session_contract import SessionRef
    from hermes_agent.gateway.session_local_recovery import restore_local_session
    from hermes_agent.gateway.session_policy import restore_policy
    from hermes_agent.gateway.session_policy_refresh import commit_refresh
    from hermes_agent.hermes_state_local import local_receipt
    from tests.gateway.test_session_resume_config_refresh import AFTER
    home = _home(monkeypatch)
    store, authority = await _owner_authority(home)
    owner = _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='t', source='cli', cwd=str(tmp_path)))['session_id']
        assert store.set_session_metadata(authority.sessions[sid].route, 'terminal_cwd', str(tmp_path))
        mirror = store.sessions_dir / 'sessions.json'
        mirror.unlink(missing_ok=True)
        mirror.mkdir()
        store._db._write_sql("CREATE TRIGGER refuse_routes BEFORE DELETE ON gateway_routing "
                             "BEGIN SELECT RAISE(ABORT, 'routing disk unavailable'); END")
        (home / 'config.yaml').write_text(AFTER)
        assert await commit_refresh(authority, SessionRef(authority.profile_id, sid))
        stored = restore_policy(local_receipt(authority.db, sid)['policy'])
        backends = [json.loads(p.terminal_json)['TERMINAL_ENV'] for p in (stored, _policy(authority, sid))]
        assert backends == ['docker', 'docker']
        store._db._write_sql('DROP TRIGGER refuse_routes')
        mirror.rmdir()
        restore_local_session(authority, sid)
    finally:
        await owner.close()
        store.close_all_db_handles()
