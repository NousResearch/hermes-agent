"""Real CLI + zstd + SQLite path against a tiny synthetic home, never real services."""
import json
from contextlib import closing
import shutil
import sqlite3
import subprocess
import sys

import pytest

pytestmark = pytest.mark.platforms('linux')


def test_real_offline_worker_verifies_actual_optimizer(tmp_path, monkeypatch):
    if shutil.which('zstd') is None:
        pytest.skip('zstd is required for the real compressed-backup integration')

    import offline_rebuild as worker
    from bounded_verification import UIDBudget
    from service_runner import verify_receipt

    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.delenv('HERMES_PROFILE', raising=False)
    monkeypatch.delenv('HERMES_GATEWAY_OWNS_PROFILE', raising=False)
    database = home / 'state.db'
    # SessionDB intentionally pins a raw header fd for its process lifetime. Seed in a
    # separate process and let it exit: never bypass the real foreign-holder preflight.
    subprocess.run([sys.executable, '-c', '''
import sys
from pathlib import Path
from hermes_state import SessionDB
with SessionDB(db_path=Path(sys.argv[1])) as db:
    db.create_session('synthetic', source='cli')
    db.append_message('synthetic', 'user', 'alpha beta preserved synthetic content')
    from hermes_state_common import LEGACY_FTS_SQL
    c = db._conn
    names = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'messages_fts%'")]
    for name in names:
        c.execute('DROP TRIGGER "' + name + '"')
    for name in ('messages_fts_cjk', 'messages_fts_trigram', 'messages_fts'):
        c.execute('DROP TABLE IF EXISTS ' + name)
    c.executescript(LEGACY_FTS_SQL)
    c.execute("INSERT INTO messages_fts(rowid,content) SELECT id,content FROM messages")
    c.execute("UPDATE state_meta SET value='0' WHERE key='fts_storage_version'")
    c.commit()
''', str(database)], cwd=worker.REPO, check=True)
    with closing(sqlite3.connect(database)) as conn:
        original = conn.execute('SELECT id,content,message_uid FROM messages').fetchall()
        cursor = conn.execute("SELECT value FROM state_meta WHERE key='message_uid_backfill_id'").fetchone()
    budget = UIDBudget(int(cursor[0]) if cursor else 0, 2000, 1)
    output = tmp_path / 'evidence'
    for name, value in [('HOME', home), ('DB', database), ('ROOT', output), ('SEARCH_TERMS', ('alpha', 'beta'))]:
        monkeypatch.setattr(worker, name, value)

    try:
        assert worker.main(execute_offline=True, budget=budget) == 0
    except Exception:
        for log in output.glob('offline-*/*.log'):
            print(log.name, log.read_text())
        raise
    assert verify_receipt(output, budget)
    receipt = json.loads(next(output.glob('offline-*/receipt.json')).read_text())
    assert receipt['quick_check'] == ['ok']
    assert receipt['search_before'] == receipt['search_after']
    assert receipt['verification']['uid_backfill']['rows'] == 0
    with closing(sqlite3.connect(database)) as conn:
        assert conn.execute('SELECT id,content,message_uid FROM messages').fetchall() == original
