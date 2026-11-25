"""The stored picker must not list zero-message husk rows (#160128).

A launch that never receives a prompt, a startup orphan reap and a resume
takeover each leave a session row with no messages behind; the row stays in
the store (resume, routing and prune still see it) but it is not a
conversation the picker can open, so the stored listing skips it. Ended-ness
is ``ended_at``, not ``end_reason``: a row the reaper closed carries a reason,
but an ended row may carry none, and a live row carries neither.
"""
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session


@asynccontextmanager
async def _picker(tmp_path, monkeypatch, seed):
    """A native reader over the rows ``seed(db)`` leaves behind."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store.persistence
    seed(db)
    runner = SimpleNamespace(session_store=store, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': str(tmp_path), 'instance_id': 'owner'})

    async def listing(**params):
        return await native.dispatch({'id': 1, 'method': 'session.list', 'params': params})
    try:
        yield listing, db
    finally:
        await native.close()
        store.close_all_db_handles()


async def _stored_ids(listing, **params):
    reply = await listing(scope='stored', **params)
    assert 'result' in reply, reply
    return [row['id'] for row in reply['result']['sessions']]


def _end_husk_without_reason(db, sid):
    """An ended row whose close left no reason: the wild shape behind the five
    ``end_reason``-NULL rows, which the store's own ghost count treats as ended
    because ``ended_at`` is set."""
    raw_holder(db)._write_sql('UPDATE sessions SET ended_at = ? WHERE id = ?', (time.time(), sid))


@pytest.mark.asyncio
async def test_zero_message_husks_are_hidden_but_conversations_stay_listed(tmp_path, monkeypatch):
    def seed(db):
        seed_session(db, 'real', 'cli', title='Real chat', model='m', cwd=str(tmp_path))
        seed_messages(db, 'real', {'role': 'user', 'content': 'real question'})
        db.sessions.end('real', 'cli_exit')
        seed_session(db, 'fresh', 'cli', title='Fresh chat', model='m', cwd=str(tmp_path))
        seed_messages(db, 'fresh', {'role': 'user', 'content': 'fresh question'})
        seed_session(db, 'pinned-real', 'cli', title='Pinned chat', model='m',
                     cwd=str(tmp_path), pinned=True)
        seed_messages(db, 'pinned-real', {'role': 'user', 'content': 'pinned question'})
        db.sessions.end('pinned-real', 'cli_exit')
        for sid, reason in [('husk-reaped', 'startup_orphan_reap'),
                            ('husk-resumed', 'resumed_other'),
                            ('husk-cron', 'cron_suppressed')]:
            seed_session(db, sid, 'cli', model='m', cwd=str(tmp_path), end_reason=reason)
        seed_session(db, 'husk-null-reason', 'cli', model='m', cwd=str(tmp_path))
        _end_husk_without_reason(db, 'husk-null-reason')
        seed_session(db, 'launch-stub', 'cli', model='m', cwd=str(tmp_path))

    async with _picker(tmp_path, monkeypatch, seed) as (listing, db):
        ids = await _stored_ids(listing)
        assert ids == ['pinned-real', 'fresh', 'real']
        # Hidden from the picker, not removed: the rows stay in the store.
        for sid in ('husk-reaped', 'husk-resumed', 'husk-cron', 'husk-null-reason', 'launch-stub'):
            assert db.sessions.get(sid) is not None


@pytest.mark.asyncio
async def test_compression_tip_with_messages_still_lists_under_its_tip(tmp_path, monkeypatch):
    from persistence.conversation.lineage import CompressionChild
    from persistence.conversation.messages import NewMessage

    def seed(db):
        seed_session(db, 'chain-root', 'cli', title='Chain chat', model='m', cwd=str(tmp_path))
        seed_messages(db, 'chain-root', {'role': 'user', 'content': 'root question'})
        assert db.coordination.try_acquire_compression_lock(session_id='chain-root', holder='fixture')
        try:
            db.lineage.publish_compression_child(
                'chain-root',
                CompressionChild(session_id='chain-tip', source='cli', model='m'),
                [NewMessage.from_live(m) for m in ({'role': 'user', 'content': 'compressed question'},)],
                compression_lock_holder='fixture')
        finally:
            db.coordination.release_compression_lock(session_id='chain-root', holder='fixture')
        seed_session(db, 'husk', 'cli', model='m', cwd=str(tmp_path), end_reason='startup_orphan_reap')

    async with _picker(tmp_path, monkeypatch, seed) as (listing, db):
        assert await _stored_ids(listing) == ['chain-tip']
        assert db.sessions.get('chain-root') is not None


@pytest.mark.asyncio
async def test_paging_over_interleaved_husks_lists_every_conversation_once(tmp_path, monkeypatch):
    def seed(db):
        for n in range(6):
            seed_session(db, f'real{n}', 'cli', title=f'Chat {n}', model='m', cwd=str(tmp_path))
            seed_messages(db, f'real{n}', {'role': 'user', 'content': f'question {n}'})
            db.sessions.end(f'real{n}', 'cli_exit')
            seed_session(db, f'husk{n}', 'cli', model='m', cwd=str(tmp_path),
                         end_reason='startup_orphan_reap')

    async with _picker(tmp_path, monkeypatch, seed) as (listing, db):
        unpaged = await _stored_ids(listing)
        assert len(unpaged) == 6
        pages, cursor = [], None
        while True:
            reply = await listing(scope='stored', limit=2,
                                  **({'cursor': cursor} if cursor else {}))
            assert 'result' in reply, reply
            pages.append([row['id'] for row in reply['result']['sessions']])
            cursor = reply['result']['next_cursor']
            if cursor is None:
                break
            assert len(pages) < 100
        assert all(len(page) == 2 for page in pages)
        assert [sid for page in pages for sid in page] == unpaged
        assert db.sessions.get('husk0') is not None
