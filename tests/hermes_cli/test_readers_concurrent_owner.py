"""Real CLI readers stay usable while the canonical store has a writer."""
import os
from pathlib import Path
import subprocess
import sys


def test_cli_readers_observe_committed_state_during_writer_transaction(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    home = tmp_path / 'profile'
    home.mkdir()
    env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(repo))
    code = '''
import os, subprocess, sys
from pathlib import Path
from hermes_state import SessionDB
path = Path(os.environ['HERMES_HOME']) / 'state.db'
with SessionDB(db_path=path) as owner:
    owner.create_session('committed-fixture', source='cli')
    owner.append_message('committed-fixture', role='user', content='committed text')
    owner.set_session_title('committed-fixture', 'Committed title')
    owner.set_session_pinned('committed-fixture', True)
    owner.flush_token_counts()
    with owner._lock:
        owner._conn.execute('BEGIN IMMEDIATE')
        owner._conn.execute("UPDATE sessions SET title='UNCOMMITTED_SENTINEL' WHERE id='committed-fixture'")
        try:
            for command in (['sessions','list'], ['sessions','stats'], ['sessions','pinned','--json'], ['insights']):
                result = subprocess.run([sys.executable, '-m', 'hermes_cli.main', *command],
                                        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
                assert result.returncode == 0, result.stdout + result.stderr
                assert 'Error:' not in result.stdout and 'Error generating' not in result.stdout, result.stdout
                assert 'UNCOMMITTED_SENTINEL' not in result.stdout, result.stdout
                if command[0] == 'sessions' and command[1] in ('list','pinned'):
                    assert 'Committed title' in result.stdout, result.stdout
                elif command[0] == 'sessions':
                    assert 'Total sessions: 1' in result.stdout, result.stdout
                else:
                    assert 'Hermes Insights' in result.stdout, result.stdout
                assert owner._conn.in_transaction, 'reader disturbed owner transaction'
        finally:
            owner._conn.rollback()
    assert owner.get_session_title('committed-fixture') == 'Committed title'
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=repo, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
