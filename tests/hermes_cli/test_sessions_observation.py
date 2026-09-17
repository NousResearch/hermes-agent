"""Session listings must not initialize or migrate canonical storage."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('populated', [False, True])
def test_session_observation_preserves_storage(tmp_path, populated):
    repo = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), HERMES_HOME=str(tmp_path / 'profile'), PYTHONPATH=str(repo))
    (tmp_path / 'profile').mkdir()
    code = '''
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
import io, os
from hermes_state import SessionDB
from hermes_cli.sessions_cmd import cmd_sessions
path = Path(os.environ['HERMES_HOME']) / 'state.db'
if POPULATED:
    with SessionDB(db_path=path) as db:
        db.create_session('observation-fixture', source='cli')
        db.append_message('observation-fixture', role='user', content='fixture')
before = path.read_bytes() if path.exists() else None
for action in ('list', 'stats', 'pinned', 'export'):
    output = io.StringIO()
    with redirect_stdout(output):
        result = cmd_sessions(SimpleNamespace(sessions_action=action, source=None, limit=20, json=True,
            session_id=None, redact=False, format='jsonl', output='-', dry_run=False))
    text = output.getvalue()
    assert result in (None, 0), text
    assert 'Error:' not in text, text
    if action == 'list':
        assert ('observation-fixture' if POPULATED else 'No sessions found.') in text, text
    if action == 'stats':
        assert ('Total sessions: 1' if POPULATED else 'Total sessions: 0') in text, text
    if action == 'pinned':
        assert text.strip() == '[]', text
    if action == 'export':
        assert ('observation-fixture' if POPULATED else 'No sessions found.') in text, text
    after = path.read_bytes() if path.exists() else None
    assert after == before, action + ' created or mutated state.db'
'''.replace('POPULATED', repr(populated))
    result = subprocess.run([sys.executable, '-c', code], cwd=repo, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
