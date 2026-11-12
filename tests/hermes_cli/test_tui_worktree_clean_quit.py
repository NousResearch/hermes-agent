"""N8: quitting ``hermes --tui -w`` removes a clean worktree (and its branch), as ``main`` did
before #106742; a dirty one, or one the gateway still has queued/started work in, is kept."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

GIT = ['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid']


def _repo(tmp_path):
    repo = tmp_path / 'repo'
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(repo)], check=True)
    (repo / '.gitignore').write_text('.worktrees/\n')
    subprocess.run([*GIT, '-C', str(repo), 'add', '.gitignore'], check=True)
    subprocess.run([*GIT, '-C', str(repo), 'commit', '-qm', 'fixture'], check=True)
    return repo


def _branches(repo):
    return subprocess.run(['git', '-C', str(repo), 'branch', '--format=%(refname:short)'],
                          capture_output=True, text=True, check=True).stdout.split()


@pytest.mark.parametrize('case', ['clean', 'clean_with_history', 'dirty', 'pending_turn'])
def test_tui_worktree_quit_removes_only_a_clean_idle_tree(tmp_path, monkeypatch, case):
    from hermes_cli import main_tui_launch
    from hermes_cli.worktree_ops import _setup_worktree
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setenv('HERMES_PYTHON', sys.executable)
    wt_info = _setup_worktree(str(repo), sync_base=False, retain_db=get_hermes_home() / 'state.db')
    monkeypatch.setattr(main_tui_launch, '_setup_tui_worktree', lambda: wt_info)
    monkeypatch.setattr(main_tui_launch, '_make_tui_argv', lambda *_: (['node'], Path('.')))
    monkeypatch.setattr(main_tui_launch, '_print_tui_exit_summary', lambda *_: None)

    def ink(_argv, cwd=None, env=None):
        db_path = get_hermes_home() / 'state.db'
        db = SessionDB(db_path=db_path)
        db.create_session('tui-wt', 'tui', cwd=env['HERMES_TUI_CWD'])
        if case != 'clean':
            db.append_message('tui-wt', 'user', 'hello')
        if case != 'pending_turn':
            # Ink's exit detaches with ``tui_shutdown``; the owner ends the idle row once no other
            # viewer holds it (gateway/session_viewer_close.py).
            db.end_session('tui-wt', 'tui_shutdown')
        db.close()
        if case == 'pending_turn':  # an accepted turn the gateway has not finished yet
            conn = sqlite3.connect(db_path)
            conn.execute(
                "INSERT INTO session_admissions(admission_id,request_id,principal_id,target_session_id,"
                "lineage_json,payload_json,payload_digest,intent,status,outcome,owner_epoch,generation) "
                "VALUES('adm-1','req-1','p','tui-wt','[]','{}','d','queue','queued',NULL,1,1)")
            conn.commit()
            conn.close()
        if case == 'dirty':
            (Path(env['HERMES_TUI_CWD']) / 'work.txt').write_text('uncommitted')
        Path(env['HERMES_TUI_ACTIVE_SESSION_FILE']).write_text(json.dumps({'session_id': 'tui-wt'}))
        return 0

    monkeypatch.setattr(main_tui_launch.subprocess, 'call', ink)
    with pytest.raises(SystemExit) as exited:
        main_tui_launch._launch_tui(worktree=True)
    assert exited.value.code == 0
    tree = Path(wt_info['path'])
    if case in ('clean', 'clean_with_history'):
        assert not tree.exists()
        assert wt_info['branch'] not in _branches(repo)
    else:
        assert tree.is_dir()
        assert wt_info['branch'] in _branches(repo)
        if case == 'dirty':
            assert (tree / 'work.txt').read_text() == 'uncommitted'


def _quit_with(tmp_path, monkeypatch, prepare):
    """Run ``hermes --tui -w`` whose Ink session ``prepare(db, cwd)`` shapes, then quit; the tree."""
    from hermes_cli import main_tui_launch
    from hermes_cli.worktree_ops import _setup_worktree
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setenv('HERMES_PYTHON', sys.executable)
    wt_info = _setup_worktree(str(repo), sync_base=False, retain_db=get_hermes_home() / 'state.db')
    monkeypatch.setattr(main_tui_launch, '_setup_tui_worktree', lambda: wt_info)
    monkeypatch.setattr(main_tui_launch, '_make_tui_argv', lambda *_: (['node'], Path('.')))
    monkeypatch.setattr(main_tui_launch, '_print_tui_exit_summary', lambda *_: None)

    def ink(_argv, cwd=None, env=None):
        db = SessionDB(db_path=get_hermes_home() / 'state.db')
        db.create_session('tui-wt', 'tui', cwd=env['HERMES_TUI_CWD'])
        db.append_message('tui-wt', 'user', 'every 10m: run the tests')
        prepare(db, env['HERMES_TUI_CWD'])
        db.close()
        Path(env['HERMES_TUI_ACTIVE_SESSION_FILE']).write_text(json.dumps({'session_id': 'tui-wt'}))
        return 0

    monkeypatch.setattr(main_tui_launch.subprocess, 'call', ink)
    with pytest.raises(SystemExit):
        main_tui_launch._launch_tui(worktree=True)
    return Path(wt_info['path']), repo, wt_info


def _scheduled(kind, status='active'):
    """The gateway's own per-session automation row, as /loop, /heartbeat and /goal store it."""
    import time
    if kind == 'loop':
        from hermes_cli.loops import LoopState
        return LoopState(prompt='run the tests', interval_seconds=600, next_due_at=time.time() + 600,
                         status=status, route={'platform': 'local', 'chat_id': 'tui-wt'}).to_json()
    if kind == 'heartbeat':
        from hermes_cli.heartbeat import HeartbeatState
        return HeartbeatState(prompt='check in', interval_seconds=600, status=status).to_json()
    from hermes_cli.goals import GoalState
    return GoalState(goal='ship it', status=status).to_json()


@pytest.mark.parametrize('kind', ['loop', 'heartbeat', 'goal'])
def test_quit_keeps_a_clean_tree_while_the_gateway_has_scheduled_work_for_its_session(tmp_path, monkeypatch, kind):
    """N8 review F4: ``/loop`` / ``/heartbeat`` / an active ``/goal`` on the ``--tui -w`` session
    keeps running on the gateway after the viewer quits; between ticks no admission is pending, and
    removing the tree made the next tick run in a missing cwd (``workspace_unavailable``)."""
    def prepare(db, _cwd):
        db.set_meta(f'{kind}:tui-wt', _scheduled(kind))
        db.end_session('tui-wt', 'tui_shutdown')
    tree, repo, wt_info = _quit_with(tmp_path, monkeypatch, prepare)
    assert tree.is_dir(), f'{kind}: tree removed while the gateway still schedules work in it'
    assert wt_info['branch'] in _branches(repo)


@pytest.mark.parametrize('kind', ['loop', 'heartbeat', 'goal'])
def test_finished_scheduled_work_does_not_keep_the_tree(tmp_path, monkeypatch, kind):
    def prepare(db, _cwd):
        db.set_meta(f'{kind}:tui-wt', _scheduled(kind, status='cleared'))
        db.end_session('tui-wt', 'tui_shutdown')
    tree, repo, wt_info = _quit_with(tmp_path, monkeypatch, prepare)
    assert not tree.exists() and wt_info['branch'] not in _branches(repo)


def test_quit_keeps_the_tree_while_another_viewer_still_holds_the_session(tmp_path, monkeypatch):
    """Ink's exit detaches with ``tui_shutdown``; the owner leaves the row open while another viewer
    (a Desktop window on the same session) still holds it, and that viewer's next turn needs the cwd."""
    tree, repo, wt_info = _quit_with(tmp_path, monkeypatch, lambda db, _cwd: None)
    assert tree.is_dir()
    assert wt_info['branch'] in _branches(repo)


@pytest.mark.parametrize('bound', ['workdir', 'origin'])
def test_quit_keeps_the_tree_while_a_cron_job_is_bound_to_it(tmp_path, monkeypatch, bound):
    from cron import jobs

    def prepare(db, cwd):
        db.end_session('tui-wt', 'tui_shutdown')
        db._execute_write(lambda conn: conn.execute("UPDATE sessions SET chat_id='local-chat' WHERE id='tui-wt'"))
        if bound == 'workdir':
            jobs.create_job('run the tests', 'every 10m', workdir=cwd)
        else:
            jobs.create_job('run the tests', 'every 10m', origin={'platform': 'local', 'chat_id': 'local-chat'})
    tree, repo, wt_info = _quit_with(tmp_path, monkeypatch, prepare)
    assert tree.is_dir(), f'cron {bound}: tree removed while a job still runs there'
    assert wt_info['branch'] in _branches(repo)
