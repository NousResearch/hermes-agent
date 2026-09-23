"""Mount-loss admission at the real workspace/dispatcher boundary."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli import kanban_db as kb


@pytest.fixture
def home(tmp_path, monkeypatch):
    for key in tuple(__import__('os').environ):
        if key.startswith('HERMES_KANBAN_'):
            monkeypatch.delenv(key)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    kb.init_db()
    return tmp_path


def configure(home, root, require_mount=True):
    (home / 'config.yaml').write_text(
        yaml.safe_dump({
            'kanban': {
                'workspaces_root': str(root),
                'workspaces_root_require_mount': require_mount,
            },
        })
    )


def scratch(path=None):
    return SimpleNamespace(id='t_probe', workspace_kind='scratch', workspace_path=path)


def test_config_preserves_board_isolation(home):
    root = home / 'scratch'
    configure(home, root, False)
    assert kb.workspaces_root('default') == root / 'default'
    assert kb.workspaces_root('other') == root / 'other'


@pytest.mark.parametrize('precreate', [False, True])
def test_missing_mount_never_creates_workspace(home, precreate):
    root = home / 'absent-volume' / 'kanban-workspaces'
    if precreate:
        root.mkdir(parents=True)
    configure(home, root)
    with pytest.raises(ValueError, match='workspaces_root_unmounted'):
        kb.resolve_workspace(scratch(), board='default')
    assert not (root / 'default').exists()
    assert root.exists() == precreate


def test_filesystem_root_never_satisfies_mount_guard(monkeypatch):
    from hermes_cli import kanban_workspace_policy as policy

    monkeypatch.setattr('os.path.ismount', lambda p: Path(p) == Path('/'))
    with pytest.raises(ValueError, match='workspaces_root_unmounted'):
        policy.validate_mount(Path('/tmp'))
    with pytest.raises(ValueError, match='workspaces_root_invalid'):
        policy.validate_mount(Path('/'))


def test_recorded_mount_anchor_cannot_fall_back_to_mounted_parent(home, monkeypatch):
    from hermes_cli import kanban_workspace_policy as policy

    root = home / 'mounted-root'
    root.mkdir()
    monkeypatch.setattr('os.path.ismount', lambda p: Path(p) in {root, root.parent})
    assert policy.validate_mount(root) == root
    monkeypatch.setattr('os.path.ismount', lambda p: Path(p) == root.parent)
    with pytest.raises(ValueError, match='workspaces_root_unmounted'):
        policy.validate_mount(root, expected_mount=root)


def test_create_scratch_keeps_admitted_mount_anchor(home, monkeypatch):
    from hermes_cli import kanban_workspace_policy as policy

    root = home / 'mounted-root'
    root.mkdir()
    configure(home, root)
    mount_present = True
    monkeypatch.setattr(
        policy.os.path,
        'ismount',
        lambda p: Path(p) == (root if mount_present else root.parent),
    )
    real_validate_target = policy.validate_target

    def vanish_after_admission(admitted_root, target):
        nonlocal mount_present
        real_validate_target(admitted_root, target)
        mount_present = False

    monkeypatch.setattr(policy, 'validate_target', vanish_after_admission)
    target = root / 'default' / 't_probe'
    with pytest.raises(ValueError, match='workspaces_root_unmounted'):
        kb.resolve_workspace(scratch(), board='default')
    assert not target.exists()


def test_persisted_path_missing_after_config_rollback_is_not_recreated(home, monkeypatch):
    root = home / 'volume' / 'kanban-workspaces'
    root.mkdir(parents=True)
    configure(home, root)
    monkeypatch.setattr('os.path.ismount', lambda p: Path(p) == root.parent)
    path = kb.resolve_workspace(scratch(), board='default')
    path.rmdir()
    (home / 'config.yaml').write_text('kanban: {}\n')
    monkeypatch.setattr('os.path.ismount', lambda p: False)
    with pytest.raises(ValueError, match='stranded_by_mount_loss|workspaces_root_unmounted'):
        kb.resolve_workspace(scratch(str(path)), board='default')
    assert not path.exists()


def test_durable_explicit_scratch_keeps_creation_contract(home):
    path = home / 'durable' / 'new-workspace'
    assert kb.resolve_workspace(scratch(str(path)), board='default') == path
    assert path.is_dir()


def test_persisted_dir_cannot_bypass_registered_mount(home, monkeypatch):
    root = home / 'volume' / 'kanban-workspaces'
    root.mkdir(parents=True)
    configure(home, root)
    monkeypatch.setattr('os.path.ismount', lambda p: Path(p) == root.parent)
    path = kb.resolve_workspace(scratch(), board='default')
    (home / 'config.yaml').write_text('kanban: {}\n')
    monkeypatch.setattr('os.path.ismount', lambda p: False)
    task = SimpleNamespace(id='t_probe', workspace_kind='dir', workspace_path=str(path))
    with pytest.raises(ValueError, match='workspaces_root_unmounted'):
        kb.resolve_workspace(task, board='default')


@pytest.mark.parametrize('dry_run', [False, True])
def test_dispatch_refuses_before_spawn_or_failure_count(home, dry_run):
    root = home / 'absent-volume' / 'kanban-workspaces'
    configure(home, root)
    with kb.connect() as conn:
        task = kb.create_task(conn, title='mount probe', assignee='default')
        calls = []
        result = kb.dispatch_once(conn, spawn_fn=lambda *args, **kw: calls.append(args), dry_run=dry_run)
        assert not calls
        assert not result.spawned
        assert not result.auto_blocked
        assert task in [item[0] for item in result.workspace_refused]
        persisted = kb.get_task(conn, task)
        assert persisted is not None
        assert persisted.status == 'ready'
        row = conn.execute('SELECT consecutive_failures FROM tasks WHERE id=?', (task,)).fetchone()
        assert row[0] == 0
        assert not root.exists()


@pytest.mark.parametrize('kind', ['scratch', 'dir', 'worktree'])
def test_restart_marks_missing_persisted_workspace_without_recreating(home, monkeypatch, kind):
    root = home / 'volume' / 'kanban-workspaces'
    root.mkdir(parents=True)
    configure(home, root)
    monkeypatch.setattr('os.path.ismount', lambda p: Path(p) == root.parent)
    path = kb.resolve_workspace(scratch(), board='default')
    path.rmdir()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title='lost workspace', assignee='default',
                                 workspace_kind=kind, workspace_path=str(path))
        calls = []
        result = kb.dispatch_once(conn, spawn_fn=lambda *args, **kw: calls.append(args))
        assert not calls
        assert task_id in result.stranded_by_mount_loss
        assert not path.exists()
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='stranded_by_mount_loss'",
                            (task_id,)).fetchone()[0] == 1
        retry = kb.dispatch_once(conn, spawn_fn=lambda *args, **kw: calls.append(args))
        assert task_id in retry.stranded_by_mount_loss
        assert not calls
        assert not path.exists()
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='stranded_by_mount_loss'",
                            (task_id,)).fetchone()[0] == 1


def test_board_symlink_cannot_redirect_creation(home, monkeypatch):
    root = home / 'volume' / 'kanban-workspaces'
    root.mkdir(parents=True)
    outside = home / 'outside'
    outside.mkdir()
    (root / 'default').symlink_to(outside, target_is_directory=True)
    configure(home, root)
    monkeypatch.setattr('os.path.ismount', lambda p: Path(p) == root.parent)
    with pytest.raises((OSError, ValueError)):
        kb.resolve_workspace(scratch(), board='default')
    assert not (outside / 't_probe').exists()

    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title='symlink target', assignee='default')
        result = kb.dispatch_once(conn, spawn_fn=lambda *_args, **_kw: None)
        assert task_id in [item[0] for item in result.workspace_refused]


def test_persisted_workspace_race_is_never_recreated(home, monkeypatch):
    root = home / 'volume' / 'kanban-workspaces'
    root.mkdir(parents=True)
    configure(home, root)
    monkeypatch.setattr('os.path.ismount', lambda p: Path(p) == root.parent)
    path = kb.resolve_workspace(scratch(), board='default')
    from hermes_cli import kanban_workspace_policy as policy
    real_validate = policy.validate_persisted

    def delete_after_validation(candidate):
        real_validate(candidate)
        candidate.rmdir()

    monkeypatch.setattr(policy, 'validate_persisted', delete_after_validation)
    assert kb.resolve_workspace(scratch(str(path)), board='default') == path
    assert not path.exists()


def test_post_claim_mount_race_requeues_without_failure_charge(home, monkeypatch):
    root = home / 'volume' / 'kanban-workspaces'
    root.mkdir(parents=True)
    configure(home, root)
    monkeypatch.setattr('os.path.ismount', lambda p: Path(p) == root.parent)
    from hermes_cli import kanban_workspace_policy as policy

    def vanish_during_create(_root, _path, **_kwargs):
        raise policy.WorkspaceUnavailable(f'workspaces_root_unmounted: {root}')

    monkeypatch.setattr(policy, 'create_scratch', vanish_during_create)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title='mount race', assignee='default')
        calls = []
        result = kb.dispatch_once(conn, spawn_fn=lambda *args, **kw: calls.append(args))
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == 'ready'
        assert task.consecutive_failures == 0
        assert not calls
        assert task_id in [item[0] for item in result.workspace_refused]


def test_review_post_claim_mount_race_returns_to_review(home, monkeypatch):
    root = home / 'volume' / 'kanban-workspaces'
    root.mkdir(parents=True)
    configure(home, root)
    monkeypatch.setattr('os.path.ismount', lambda p: Path(p) == root.parent)
    from hermes_cli import kanban_workspace_policy as policy

    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title='review mount race', assignee='default')
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.request_review(
            conn, task_id, reviewer='default',
            expected_run_id=claimed.current_run_id,
        )

        def vanish_during_create(_root, _path, **_kwargs):
            raise policy.WorkspaceUnavailable(f'workspaces_root_unmounted: {root}')

        monkeypatch.setattr(policy, 'create_scratch', vanish_during_create)
        result = kb.dispatch_once(conn, spawn_fn=lambda *_args, **_kw: None)
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == 'review'
        assert task.current_run_id is None
        assert task.consecutive_failures == 0
        assert task_id in [item[0] for item in result.workspace_refused]


def test_most_specific_historical_root_wins(home, monkeypatch):
    broad = home / 'volume'
    specific = broad / 'kanban-workspaces'
    path = specific / 'default' / 't_probe'
    path.mkdir(parents=True)
    (home / 'config.yaml').write_text('kanban: {}\n')
    with kb.connect_closing() as conn:
        conn.execute(
            "INSERT INTO workspace_mount_roots(root, mount_path) VALUES (?, ?), (?, ?)",
            (str(broad), str(broad), str(specific), str(broad)),
        )
        from hermes_cli import kanban_workspace_policy as policy

        def validate(root, expected_mount=None):
            if root == specific:
                raise policy.WorkspaceUnavailable(f'workspaces_root_unmounted: {root}')
            return expected_mount or root

        monkeypatch.setattr(policy, 'validate_mount', validate)
        task_id = kb.create_task(
            conn, title='nested fence', assignee='default',
            workspace_kind='scratch', workspace_path=str(path),
        )
        result = kb.dispatch_once(conn, spawn_fn=lambda *_args, **_kw: None)
        assert task_id in result.stranded_by_mount_loss


def test_unwritable_mount_refuses_before_claim(home, monkeypatch):
    root = home / 'volume' / 'kanban-workspaces'
    root.mkdir(parents=True)
    configure(home, root)
    monkeypatch.setattr('os.path.ismount', lambda p: Path(p) == root.parent)

    from hermes_cli import kanban_workspace_policy as policy

    real_open = policy.os.open

    def refuse_probe(path, flags, *args, **kwargs):
        if Path(path).name.startswith('.hermes-write-probe-') and flags & policy.os.O_CREAT:
            raise PermissionError('read-only mount')
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(policy.os, 'open', refuse_probe)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title='read-only mount', assignee='default')
        calls = []
        result = kb.dispatch_once(conn, spawn_fn=lambda *args, **kw: calls.append(args))
        assert not calls
        assert task_id in [item[0] for item in result.workspace_refused]
        assert 'workspaces_root_unwritable' in dict(result.workspace_refused)[task_id]
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == 'ready'
