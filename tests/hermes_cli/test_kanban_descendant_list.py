"""Read-only CLI descendants must not perform dependency transitions."""
import json
import os
from pathlib import Path
import subprocess
import sys

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def test_descendant_list_reads_without_promoting_or_granting_writes(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD_CONTEXT', raising=False)
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title='Read-only fixture')
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (tid,))
        conn.commit()
    env = {**os.environ, 'HERMES_DELEGATED_CHILD_CONTEXT': '1'}
    root = Path(__file__).resolve().parents[2]
    def cli(*args, child=True):
        return subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'kanban', *args],
            cwd=root, env=env if child else dict(os.environ), stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=45)
    for alias in ('list', 'ls'):
        result = cli(alias, '--json')
        assert result.returncode == 0, result.stderr
        assert next(t for t in json.loads(result.stdout) if t['id'] == tid)['status'] == 'todo'
    assert cli('complete', tid, '--result', 'forbidden').returncode != 0
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == 'todo'
    result = cli('list', '--json', child=False)
    assert result.returncode == 0, result.stderr
    assert next(t for t in json.loads(result.stdout) if t['id'] == tid)['status'] == 'ready'
