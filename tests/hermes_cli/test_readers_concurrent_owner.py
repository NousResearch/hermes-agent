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
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session


class RollBack(Exception):
    pass


def readers(conn):
    conn.execute("UPDATE sessions SET title='UNCOMMITTED_SENTINEL' WHERE id='committed-fixture'")
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
        assert conn.in_transaction, 'reader disturbed owner transaction'
    raise RollBack  # roll the write transaction back


with open_persistence(profile_scope(home=Path(os.environ['HERMES_HOME']))) as owner:
    seed_session(owner, 'committed-fixture', 'cli')
    seed_messages(owner, 'committed-fixture', {'role': 'user', 'content': 'committed text'})
    owner.sessions.set_title('committed-fixture', 'Committed title')
    owner.sessions.set_pinned('committed-fixture', True)
    owner.usage.flush_usage()
    try:
        raw_holder(owner)._execute_write(readers)
    except RollBack:
        pass
    assert owner.sessions.get_title('committed-fixture') == 'Committed title'
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=repo, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
