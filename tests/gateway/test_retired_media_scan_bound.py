"""Retired-media collection: one bounded messages pass per collection, never on the event loop."""
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from tests.fixtures.storage_holder import raw_holder, retired_media_key
from tests.fixtures.storage_seed import seed_messages, seed_session


def _trace_messages_scans(holder):
    """Record every statement that searches messages.content, on read and writer connections."""
    seen = []
    def trace(sql):
        if 'messages' in sql and 'instr(' in sql:
            seen.append(sql)
    original = holder._read_ctx
    @contextmanager
    def traced():
        with original() as conn:
            conn.set_trace_callback(trace)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)
    holder._read_ctx = traced
    holder._conn.set_trace_callback(trace)
    return seen


def test_candidates_share_one_messages_pass(owner, tmp_path, monkeypatch):
    from gateway.session_ingress_media import _media_root
    from session_retention import collect_retired_media
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store, root = owner.persistence, _media_root()
    seed_session(store, 'keep', 'api_server')
    references = []
    for index in range(6):
        sha = f'{index:064x}'
        path = root / sha / f'api_{sha[:32]}.png'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'x')
        references.append({'path': str(path), 'sha256': sha, 'size': 1})
    for reference in references[:3]:
        seed_messages(store, 'keep', {'role': 'user', 'content': '[Image attached at: {}]'.format(reference['path'])})
    seed_messages(store, 'keep', {'role': 'user', 'content': 'unrelated history'})
    raw_holder(store)._execute_write(lambda conn: conn.executemany('INSERT INTO state_meta(key,value) VALUES(?,?)',
        [(retired_media_key(str(index)), json.dumps(reference)) for index, reference in enumerate(references)]))
    scans = _trace_messages_scans(raw_holder(store))
    collect_retired_media(store, store.scope.profile_dir)
    full = [sql for sql in scans if 'id>' not in sql.replace(' ', '')]
    assert len(full) == 1, f'{len(references)} candidates must share one messages scan, saw {len(full)}'
    assert [Path(reference['path']).exists() for reference in references] == [True] * 3 + [False] * 3
    left = store.admissions.retired_candidates('media')
    assert sorted(key for key, _reference in left) == [retired_media_key(str(index)) for index in range(3)]


@pytest.mark.asyncio
async def test_gateway_delete_collects_retired_media_off_the_event_loop(tmp_path, monkeypatch):
    import session_retention
    from gateway.session_mutations import mutate_session
    from tests.gateway.test_local_authority_transitions import local_session
    owner, connection, ref = await local_session(tmp_path, monkeypatch)
    threads = []
    monkeypatch.setattr(session_retention, 'collect_retired_media', lambda store, home: threads.append(threading.get_ident()))
    handle = owner._handle(ref)
    try:
        await mutate_session(owner, connection.actor, ref, dict(session_id=ref.session_id, request_id='delete',
            operation='delete', payload={}, expected_revision=handle.revision,
            expected_generation=handle.execution_generation))
        assert threads and threading.get_ident() not in threads
    finally:
        await connection.close()
        owner.runner.session_store.close_all_db_handles()
