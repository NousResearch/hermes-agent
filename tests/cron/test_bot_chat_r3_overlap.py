"""Upgrade overlap uses the predecessor's stripe before reserving SQLite."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys

import pytest


OLD = Path(__file__).parent / "fixtures" / "approved_bot_chat_delivery.py"


def _line(proc):
    assert select.select([proc.stdout], [], [], 15)[0], "child failed to reach barrier"
    return proc.stdout.readline().strip()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipe barriers")
@pytest.mark.parametrize("producer", ["legacy", "sql"])
def test_predecessor_check_replace_serializes_new_producer(tmp_path, producer):
    env = dict(os.environ, HERMES_HOME=str(tmp_path))
    old_code = '''
import importlib.util, sys
from pathlib import Path
from cron import delivery_queue as q
spec = importlib.util.spec_from_file_location('predecessor', sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
home = Path(sys.argv[2]); key = 'a' * 64
original = m._next_sequence_unlocked

def pause(root):
    print('checked-absent', flush=True)
    assert sys.stdin.readline().strip() == 'publish'
    # The successor must NOT hold SQLite while waiting for our stripe.
    assert q.get_status(key) is None
    return original(root)

m._next_sequence_unlocked = pause
with m.ownership_arbitration(key):
    assert q.get_status(key) is None
    record, created = m.claim_direct(key, {'id': 'job', 'name': 'Daily',
        'execution_id': 'run'}, 'body', '', home)
    assert created
print('old-published', flush=True)
'''
    new_code = '''
import json, sys
from contextlib import contextmanager
from pathlib import Path
from cron import bot_chat_legacy as m, delivery_queue as q
from hermes_cli.active_sessions import _FileLock
home = Path(sys.argv[1]); key = 'a' * 64
original_enter = _FileLock.__enter__
original_transaction = q._transaction

def enter(lock):
    print('stripe-attempt', flush=True)
    return original_enter(lock)

@contextmanager
def transaction(*a, **kw):
    print('sql-attempt', flush=True)
    with original_transaction(*a, **kw) as conn:
        yield conn

_FileLock.__enter__ = enter
q._transaction = transaction
job = {'id': 'job', 'name': 'Daily', 'execution_id': 'run'}
if sys.argv[2] == 'legacy':
    record, created = m.publish(key, job, 'body', home, for_failure=False, claimed=True)
    if created:
        with m.ownership_arbitration(key):
            record['status'] = 'settled'
            m._terminalize_unlocked(m._receipt_path(m._root(), key), record)
else:
    job['_bot_chat_pending'] = str(home)
    record = q.enqueue(key, job, 'body')
    created = record.get('ownership_store') != 'legacy'
print(json.dumps({'created': created, 'status': record['status']}), flush=True)
'''
    old = subprocess.Popen([sys.executable, "-u", "-c", old_code, str(OLD), str(tmp_path)],
                           env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, bufsize=1)
    new = None
    try:
        assert _line(old) == "checked-absent"
        new = subprocess.Popen([sys.executable, "-u", "-c", new_code, str(tmp_path), producer],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, bufsize=1)
        first_lock = _line(new)
        # While the predecessor is paused, the successor must not enter SQL
        # or publish anything. The pipe is signalled immediately before SQL.
        crossed_while_paused = bool(select.select([new.stdout], [], [], 0.25)[0])
        # Always release the child, including on RED; no hung test processes.
        old.stdin.write("publish\n")
        old.stdin.flush()
        old_out, old_err = old.communicate(timeout=20)
        new_out, new_err = new.communicate(timeout=20)
        assert first_lock == "stripe-attempt", "successor reserved SQLite before predecessor stripe"
        assert not crossed_while_paused, "successor crossed the paused predecessor's ownership stripe"
        assert old.returncode == 0, (old_out, old_err)
        assert new.returncode == 0, (new_out, new_err)
        result = json.loads(new_out.splitlines()[-1])
        assert result == {"created": False, "status": "claimed"}
        receipt = json.loads((tmp_path / "cron/bot_chat_pending" / ("a" * 64 + ".json")).read_text())
        assert receipt["status"] == "claimed" and receipt["direct"] is True
    finally:
        for proc in (old, new):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.communicate()
