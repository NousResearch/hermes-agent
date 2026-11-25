"""Local HTTP/lifecycle probe; no external GitHub writes or inference.
Run: python evals/kanban_pr_acceptance_live.py ROOT
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
os.environ.pop('HERMES_DELEGATED_CHILD_CONTEXT', None)
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
import inspect

def create(conn, title):
    if 'completion_contract' in inspect.signature(kb.create_task).parameters:
        return kb.create_task(conn, title=title, completion_contract='acme/repo')
    tid = kb.create_task(conn, title=title)
    if 'completion_contract' not in {r[1] for r in conn.execute('PRAGMA table_info(tasks)')}:
        conn.execute('ALTER TABLE tasks ADD COLUMN completion_contract TEXT')
    conn.execute('UPDATE tasks SET completion_contract=? WHERE id=?', ('acme/repo', tid))
    conn.commit()
    return tid

fixture_root = Path(sys.argv[2]) if len(sys.argv) > 2 else root
spec = importlib.util.spec_from_file_location('http_fixture', fixture_root / 'tests/hermes_cli/test_kanban_pr_acceptance.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
patch = pytest.MonkeyPatch()
with tempfile.TemporaryDirectory(prefix='hermes-pr-live-') as home:
    setup = fixture.github.__wrapped__(Path(home), patch)
    state = next(setup)
    reports = []
    try:
        with connect() as conn:
            for outcome in ('failure', 'cancelled', 'timed_out', 'success'):
                state.update(conclusion=outcome)
                tid = create(conn, 'PR acceptance live')
                accepted = kb.complete_task(conn, tid, metadata={'published_pr': 'https://github.com/acme/repo/pull/7'})
                row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC", (tid,)).fetchone()
                receipt = json.loads(row[0]) if row else None
                reports.append({'case': outcome, 'accepted': accepted, 'status': kb.get_task(conn, tid).status, 'receipt': receipt})
            for outcome in (('success', 'failure') if 'completion_contract' in inspect.signature(kb.create_task).parameters else ()):
                tid = kb.create_task(conn, title='Concurrent ownership', completion_contract='acme/repo')
                owner = kb.claim_task(conn, tid)
                run_id = owner.current_run_id
                def reclaim():
                    with connect() as other:
                        kb.block_task(other, tid, reason='Reassigned during acceptance')
                        kb.unblock_task(other, tid)
                        state['replacement'] = kb.claim_task(other, tid).current_run_id
                state.update(conclusion=outcome, race=reclaim)
                accepted = kb.complete_task(conn, tid, expected_run_id=run_id, metadata={'published_pr': 'https://github.com/acme/repo/pull/7'})
                reports.append({'case': 'CAS-'+outcome, 'accepted': accepted, 'run_id': kb.get_task(conn, tid).current_run_id,
                    'receipts': conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0]})
                del state['race']
        print(json.dumps({'reports': reports, 'requests': state['requests']}, indent=2))
    finally:
        try:
            next(setup)
        except StopIteration:
            pass
        patch.undo()
