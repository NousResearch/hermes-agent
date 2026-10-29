"""N8 review 2.2: quitting a clean ``hermes --tui -w`` session that had a conversation removes its
tree and branch, but the session's policy froze that cwd, so a later resume had every turn refused
``workspace_unavailable``. A resume (``--tui --resume``, ``-c``, classic CLI; all go through
``_resolve_chat_session_args``) now recreates a missing tree the ``--tui -w`` launcher created, at
the same path, before attaching. Other missing directories are never created."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _quit_clean_session_with_history(tmp_path, monkeypatch):
    """``hermes --tui -w``, one user + assistant exchange, ``/quit`` (row ended ``tui_shutdown``)."""
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
        db.append_message('tui-wt', 'user', 'hello')
        db.append_message('tui-wt', 'assistant', 'hi')
        db.end_session('tui-wt', 'tui_shutdown')
        db.close()
        Path(env['HERMES_TUI_ACTIVE_SESSION_FILE']).write_text(json.dumps({'session_id': 'tui-wt'}))
        return 0

    monkeypatch.setattr(main_tui_launch.subprocess, 'call', ink)
    with pytest.raises(SystemExit):
        main_tui_launch._launch_tui(worktree=True)
    tree = Path(wt_info['path'])
    assert not tree.exists(), 'precondition: the clean idle tree is removed on quit (N8)'
    return repo, tree, wt_info


def _resume(session_id, **extra):
    from hermes_cli import main as m
    args = SimpleNamespace(resume=session_id, continue_last=None, in_dir=None, worktree=False,
                           no_restore_cwd=False, create_if_missing=False, **extra)
    m._resolve_chat_session_args(args, use_tui=True)
    return args


def _turn_runs_in(cwd):
    """The owner's readiness gate a resumed turn passes through (``policy_scope``)."""
    from gateway.session_policy import policy_scope
    from hermes_state_runtime import RuntimeStoreError
    try:
        with policy_scope(SimpleNamespace(cwd=str(cwd))):
            pass
    except RuntimeStoreError as exc:
        return str(exc) != 'workspace_unavailable'
    except Exception:  # health: allow BLE001 -- past the readiness gate; the stub policy is incomplete
        return True
    return True


@pytest.mark.parametrize('no_restore_cwd', [False, True])
def test_resume_recreates_the_removed_tui_worktree_so_turns_run(tmp_path, monkeypatch, no_restore_cwd):
    repo, tree, wt_info = _quit_clean_session_with_history(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    from hermes_cli import main as m
    args = SimpleNamespace(resume='tui-wt', continue_last=None, in_dir=None, worktree=False,
                           no_restore_cwd=no_restore_cwd, create_if_missing=False)
    m._resolve_chat_session_args(args, use_tui=True)
    assert tree.is_dir(), ('quit removed the resumable session\'s worktree; the resumed session\'s '
                           'turns are refused workspace_unavailable')
    assert _turn_runs_in(tree)
    assert wt_info['branch'] in _branches(repo)
    head = subprocess.run(['git', '-C', str(tree), 'rev-parse', '--abbrev-ref', 'HEAD'],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == wt_info['branch']
    if not no_restore_cwd:
        assert Path.cwd().resolve() == tree.resolve()


def test_resume_reuses_the_surviving_branch(tmp_path, monkeypatch):
    """A tree removed while its branch survived comes back on that branch, at its commit."""
    from hermes_cli.worktree_ops import _setup_worktree
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    wt_info = _setup_worktree(str(repo), sync_base=False, retain_db=get_hermes_home() / 'state.db')
    tree = Path(wt_info['path'])
    (tree / 'f.txt').write_text('work')
    subprocess.run([*GIT, '-C', str(tree), 'add', 'f.txt'], check=True)
    subprocess.run([*GIT, '-C', str(tree), 'commit', '-qm', 'work'], check=True)
    tip = subprocess.run(['git', '-C', str(tree), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                         check=True).stdout.strip()
    db = SessionDB(db_path=get_hermes_home() / 'state.db')
    db.create_session('tui-wt', 'tui', cwd=str(tree))
    db.append_message('tui-wt', 'user', 'hello')
    db.close()
    subprocess.run(['git', '-C', str(repo), 'worktree', 'remove', '--force', '--force', str(tree)], check=True)
    assert not tree.exists() and wt_info['branch'] in _branches(repo)

    _resume('tui-wt')
    assert (tree / 'f.txt').read_text() == 'work'
    assert subprocess.run(['git', '-C', str(tree), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip() == tip


def test_resume_never_creates_a_missing_dir_the_launcher_did_not_make(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    repo = _repo(tmp_path)
    foreign = repo / '.worktrees' / 'hermes-deadbeef'  # looks like ours; no launcher record
    db = SessionDB(db_path=get_hermes_home() / 'state.db')
    db.create_session('other', 'tui', cwd=str(foreign))
    db.append_message('other', 'user', 'hello')
    db.create_session('plain', 'tui', cwd=str(tmp_path / 'gone'))
    db.append_message('plain', 'user', 'hello')
    db.close()
    monkeypatch.chdir(tmp_path)
    _resume('other')
    _resume('plain')
    assert not foreign.exists()
    assert not (tmp_path / 'gone').exists()


def test_a_store_error_reading_the_policy_falls_back_to_the_row_cwd(tmp_path, monkeypatch):
    """The frozen-policy read failing on the store (e.g. no ``state_meta`` table) still recreates
    the tree from the row's cwd instead of skipping the resume step."""
    import sqlite3

    import hermes_state_local
    from hermes_cli.worktree_ops import _setup_worktree
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    tree = Path(_setup_worktree(str(repo), sync_base=False, retain_db=get_hermes_home() / 'state.db')['path'])
    db = SessionDB(db_path=get_hermes_home() / 'state.db')
    db.create_session('tui-wt', 'tui', cwd=str(tree))
    db.append_message('tui-wt', 'user', 'hello')
    db.close()
    subprocess.run(['git', '-C', str(repo), 'worktree', 'remove', '--force', '--force', str(tree)], check=True)

    def broken(*_a, **_k):
        raise sqlite3.OperationalError('no such table: state_meta')

    monkeypatch.setattr(hermes_state_local, 'local_receipt', broken)
    monkeypatch.chdir(tmp_path)
    _resume('tui-wt')
    assert tree.is_dir()



@pytest.mark.asyncio
async def test_resume_recreates_the_removed_classic_worktree(tmp_path, monkeypatch):
    """Classic ``hermes -w`` runs its session's tools in the tree (frozen policy cwd) while the row
    keeps main's stamp, the repo root. Quit removes the clean tree; ``-c`` now gets it back."""
    import cli
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_cli import main as m

    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setitem(cli.CLI_CONFIG, 'worktree_sync', False)
    monkeypatch.setitem(sys.modules, 'model_tools', SimpleNamespace(get_tool_definitions=lambda **_: []))
    monkeypatch.setattr(cli.atexit, 'register', lambda *_a, **_k: None)
    monkeypatch.setattr(cli, '_prune_stale_worktrees', lambda *_: None)
    monkeypatch.setattr(cli, '_maintain_pack_health', lambda *_: None)
    monkeypatch.setattr(cli, '_active_worktree', None)
    monkeypatch.setenv('TERMINAL_CWD', str(tmp_path))
    wt_info = cli._start_worktree_setup(False, False, True, False)()  # the -w launcher
    tree = Path(wt_info['path'])

    from gateway import run
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    try:
        authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='owner')
        actor = AuthorityConnection(authority, object(), {'user_id': 'human', 'profile_id': 'fixture'}).actor
        # What GatewaySessionStore.create sends for -w: TERMINAL_CWD (the tree), row stamp os.getcwd().
        ref = create_local_session(authority, actor, {'request_id': 'w', 'source': 'cli', 'cwd': str(tree),
                                                      'row_cwd': str(repo), 'model': 'm', 'toolsets': []})
        assert authority.db.get_session(ref.session_id)['cwd'] == str(repo)
        cli._cleanup_worktree(wt_info)  # quit
        assert not tree.exists(), 'precondition: quit removes the clean tree'

        monkeypatch.chdir(tmp_path)
        args = SimpleNamespace(resume=ref.session_id, continue_last=None, in_dir=None, worktree=False,
                               no_restore_cwd=False, create_if_missing=False)
        m._resolve_chat_session_args(args, use_tui=False)
        assert tree.is_dir(), 'the resumed -w session\'s turns are refused workspace_unavailable'
        assert _turn_runs_in(tree)
    finally:
        store.close_all_db_handles()
