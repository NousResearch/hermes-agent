"""Kanban route wakes are one pending, live-state digest, not frozen turns."""
import json

import pytest

from gateway.session_automation import admit_route_wake
from gateway.session_contract import SessionActor, Submission
from gateway.session_local import create_local_session
from gateway.session_authority import admission_row
from gateway.session_persistence import authority_persistence
from tests.gateway.test_automation_notification_category import _authority


async def owner(tmp_path, monkeypatch):
    runner, authority = await _authority(tmp_path, monkeypatch)
    actor = SessionActor('test-owner', 'default', frozenset({'session:create', 'session:submit', 'session:read'}), 'fixture')
    ref = create_local_session(authority, actor, dict(request_id='digest', source='gui', model='fixture', toolsets=[]))
    live = authority.sessions[ref.session_id]
    return runner, authority, actor, ref, live, runner._adapter_for_source(live.source)


def queued(authority, ref):
    return [admission_row(r) for r in authority_persistence(authority).admissions.list(session_id=ref.session_id)
            if r.status == 'queued']


@pytest.mark.asyncio
async def test_pending_cards_fold_across_human_without_folding_other_producers(tmp_path, monkeypatch):
    runner, authority, actor, ref, live, adapter = await owner(tmp_path, monkeypatch)
    try:
        await admit_route_wake(runner, adapter, live.source, 'card A', 'kanban-wake')
        await authority.submit(actor, Submission('human', ref, {'text': 'owner first'}, 'queue'))
        await admit_route_wake(runner, adapter, live.source, 'loop', 'loop-tick')
        await admit_route_wake(runner, adapter, live.source, 'card B', 'kanban-wake')
        rows = queued(authority, ref)
        assert len(rows) == 3
        assert rows[0]['payload']['text'] == 'owner first'
        assert rows[1]['payload']['text'] == 'loop'
        assert 'card A' in rows[2]['payload']['text'] and 'card B' in rows[2]['payload']['text']
        descriptor = rows[2]['payload']['local_automation_v1']
        assert len(descriptor['identities']) == 2
    finally:
        authority.runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_owner_queued_after_digest_is_claimed_first(tmp_path, monkeypatch):
    runner, authority, actor, ref, live, adapter = await owner(tmp_path, monkeypatch)
    try:
        await admit_route_wake(runner, adapter, live.source, 'card A', 'kanban-wake')
        await authority.submit(actor, Submission('human', ref, {'text': 'owner first'}, 'queue'))
        row, _ = await authority._claim_next(ref, live)
        assert row['payload']['text'] == 'owner first'
    finally:
        authority.runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_diagnostics_are_not_promoted_into_result_digest(tmp_path, monkeypatch):
    runner, authority, _actor, ref, live, adapter = await owner(tmp_path, monkeypatch)
    try:
        for text, category in [('result A', 'result'), ('warning', 'diagnostic'), ('result B', 'result')]:
            await admit_route_wake(runner, adapter, live.source, text, 'kanban-wake', {'notification_category': category})
        rows = queued(authority, ref)
        assert len(rows) == 2
        assert rows[0]['payload']['local_automation_v1']['notification_category'] == 'diagnostic'
        assert 'warning' not in rows[1]['payload']['text']
        assert 'result A' in rows[1]['payload']['text'] and 'result B' in rows[1]['payload']['text']
    finally:
        authority.runner.session_store.close_all_db_handles()


@pytest.mark.parametrize('status', ['done', 'running'])
def test_execution_reads_current_state_and_never_replays_failure(tmp_path, status):
    from gateway.session_kanban_digest import refresh_text
    path = tmp_path / 'board.db'
    import sqlite3
    with sqlite3.connect(path) as conn:
        conn.executescript('CREATE TABLE tasks(id TEXT,status TEXT,title TEXT,assignee TEXT); CREATE TABLE kanban_notify_subs(task_id TEXT,platform TEXT,chat_id TEXT,thread_id TEXT);')
        conn.execute('INSERT INTO tasks VALUES(?,?,?,?)', ('t_a', 'blocked', 'Live title', 'worker'))
        conn.execute('INSERT INTO kanban_notify_subs VALUES(?,?,?,?)', ('t_a', 'tui', 'session', ''))
    notice = {'db': str(path), 'board': 'test', 'task_id': 't_a', 'platform': 'tui', 'chat_id': 'session', 'thread_id': ''}
    with sqlite3.connect(path) as conn:
        conn.execute('UPDATE tasks SET status=?', (status,))
    text = refresh_text({'kanban_notices': [notice]}, 'OLD blocked reason')
    assert f'Current status: {status}' in text
    assert 'Live title' in text
    assert 'OLD blocked reason' not in text


def test_unsubscribe_and_archive_drop_queued_notice_without_writable_open(tmp_path):
    from gateway.session_kanban_digest import refresh_text
    import sqlite3
    path = tmp_path / 'board.db'
    with sqlite3.connect(path) as conn:
        conn.executescript("CREATE TABLE tasks(id TEXT,status TEXT,title TEXT,assignee TEXT); CREATE TABLE kanban_notify_subs(task_id TEXT,platform TEXT,chat_id TEXT,thread_id TEXT); INSERT INTO tasks VALUES('t_a','archived','title','worker');")
    notice = {'db': str(path), 'board': 'test', 'task_id': 't_a', 'platform': 'tui', 'chat_id': 'session', 'thread_id': ''}
    assert refresh_text({'kanban_notices': [notice]}, 'old') is None
    assert refresh_text({'kanban_notices': [{**notice, 'db': str(tmp_path / 'missing.db')}]}, 'old') == '[kanban] Current state unavailable; historical notification withheld.'
    assert not (tmp_path / 'missing.db').exists()


@pytest.mark.asyncio
async def test_concurrent_notices_fold_once_without_losing_identities(tmp_path, monkeypatch):
    import asyncio
    runner, authority, _actor, ref, live, adapter = await owner(tmp_path, monkeypatch)
    try:
        await asyncio.gather(*(admit_route_wake(runner, adapter, live.source, f'card {i}', 'kanban-wake')
                               for i in range(4)))
        rows = queued(authority, ref)
        assert len(rows) == 1
        desc = rows[0]['payload']['local_automation_v1']
        assert len(desc['identities']) == 4
        for i in range(4):
            assert rows[0]['payload']['text'].count(f'card {i}') == 1
        all_rows = authority_persistence(authority).admissions.list(session_id=ref.session_id, pending_only=False)
        assert len([r for r in all_rows if r.status == 'terminal']) == 3
    finally:
        authority.runner.session_store.close_all_db_handles()


def test_only_kanban_yields_to_human_and_preflight_head_matches_runtime():
    from persistence.authority.admission_order import preferred_session_input
    def row(seq, principal, payload, producer=None):
        return dict(seq=seq, status='queued', principal_id=principal, payload=payload,
                    request_id=json.dumps([producer, 'route', str(seq)]))
    kanban = row(1, 'automation:route', {'local_automation_v1': {}}, 'kanban-wake')
    loop = row(2, 'automation:route', {'local_automation_v1': {}}, 'loop-tick')
    worker = row(3, 'worker:one', {'text': 'work'})
    human = row(4, 'owner', {'text': 'owner first'})
    assert preferred_session_input([kanban, worker]) is kanban
    assert preferred_session_input([kanban, loop, human]) is loop
    assert preferred_session_input([kanban, human]) is human
    assert preferred_session_input([kanban]) is kanban
