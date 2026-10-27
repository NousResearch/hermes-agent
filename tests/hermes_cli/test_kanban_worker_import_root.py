"""Kanban module workers must not import a task worktree's Hermes checkout."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

@pytest.mark.parametrize('resolved', [None, ['/opt/hermes/bin/hermes']])
def test_module_worker_ignores_workspace_checkout(tmp_path, monkeypatch, resolved):
    import tools.environments.local  # warm import-time git probe before patching Popen
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    home = tmp_path / '.hermes'
    home.mkdir()
    workspace = tmp_path / 'worktree'
    shadow = workspace / 'hermes_cli'
    shadow.mkdir(parents=True)
    (shadow / '__init__.py').write_text("raise RuntimeError('workspace shadow imported')\n")
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    for key in ('HERMES_KANBAN_HOME', 'PYTHONPATH', 'PYTHONSAFEPATH', 'HERMES_BIN'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(kbd, '_resolve_hermes_argv', lambda: resolved or [sys.executable, '-m', 'hermes_cli.main'])
    monkeypatch.setattr(kbd, '_live_worker_procs', {})
    captured = {}
    real_popen = subprocess.Popen

    class CapturePopen:
        def __init__(self, cmd, **kwargs):
            captured.update(cmd=cmd, **kwargs)
            self.pid = 4242

    monkeypatch.setattr(subprocess, 'Popen', CapturePopen)
    task = kb.Task(
        id='t_safe_path', title='x', body=None, assignee='coder', status='ready',
        priority=0, created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind='worktree', workspace_path=str(workspace), claim_lock=None,
        claim_expires=None, tenant=None, branch_name=None,
    )
    kbd._default_spawn(task, str(workspace))
    monkeypatch.setattr(subprocess, 'Popen', real_popen)
    command = captured['cmd']
    assert command[:4] == [sys.executable, '-P', '-m', 'hermes_cli.kanban_worker_client']
    assert command.count('-P') == 1
    assert 'PYTHONSAFEPATH' not in captured['env']
    root = Path(kbd.__file__).resolve().parents[1]
    assert captured['env']['PYTHONPATH'].split(os.pathsep)[0] == str(root)
    # Resolve the real package in a bare child without starting an owner or agent.
    probe = subprocess.run(
        command[:2] + ['-c', 'import importlib.util,json; print(json.dumps(importlib.util.find_spec("hermes_cli").origin))'],
        cwd=captured['cwd'], env=captured['env'], capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, probe.stderr
    assert Path(json.loads(probe.stdout)).resolve() == root / 'hermes_cli' / '__init__.py'
