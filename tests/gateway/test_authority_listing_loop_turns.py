"""The live listings walk every live session; their store reads must not run on the gateway's loop.

A gateway can hold thousands of live session rows (a restart refills the registry from receipts);
``session.list`` / ``session.active_list`` walked them one at a time on the loop thread, with three
full hydrations and a receipt JSON parse per session and no suspension point. Measured on a
4,049-row store the loop stalled past 10 s, the Telegram adapter's ``updater.stop()`` missed its
deadline and the adapter was declared fatal (#141236).

Pinned here: the loop gets a turn while a listing is in flight, no store read of the walk runs on the
loop thread, the reads are batched (not one per session), and the closed-and-unviewed rule is kept.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest
import pytest_asyncio

SESSION_COUNT = 300  # more than one LISTING_BATCH


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                             _handle_message=None, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id='live', instance_id='owner')
    sids = []
    for index in range(SESSION_COUNT):
        sid = f'sess-{index:03d}'
        store._db.create_session(sid, source='telegram')
        authority.sessions[sid] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id=sid), sid)
        sids.append(sid)
    connection = AuthorityConnection(authority, object(), {'user_id': 'viewer'})

    async def listing(method, **params):
        reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
        assert 'result' in reply, reply
        return reply['result']['sessions']

    yield SimpleNamespace(authority=authority, db=store._db, sids=sids, listing=listing)


def _count_reads_by_thread(monkeypatch, db):
    """Every read connection checkout the walk makes, keyed by whether it ran on the loop thread."""
    loop_thread = threading.get_ident()
    reads = {'loop': 0, 'worker': 0}
    original = type(db)._read_ctx

    def counting(self, *args, **kwargs):
        reads['loop' if threading.get_ident() == loop_thread else 'worker'] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(db), '_read_ctx', counting)
    return reads


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['session.list', 'session.active_list'])
async def test_a_live_listing_reads_the_store_off_the_loop_in_batches(owner, monkeypatch, method):
    reads = _count_reads_by_thread(monkeypatch, owner.db)
    rows = await owner.listing(method)
    assert len(rows) == (200 if method == 'session.list' else SESSION_COUNT)
    assert reads['loop'] == 0, reads
    # One worker read per batch, never one per session.
    assert 0 < reads['worker'] < SESSION_COUNT // 10, reads


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['session.list', 'session.active_list'])
async def test_a_live_listing_gives_the_loop_a_turn_before_it_returns(owner, method):
    order = []

    async def sweep():
        await owner.listing(method)
        order.append('listing')

    async def competing():
        order.append('competing')

    # Scheduled in this order, so an unyielding sweep appends first.
    await asyncio.gather(sweep(), competing())
    assert order == ['competing', 'listing'], order


@pytest.mark.asyncio
async def test_a_closed_unviewed_local_session_still_leaves_the_live_feed(owner):
    """The batched read keeps ``closed_and_unviewed``: a local session whose receipt target is ended
    and that nobody views is listed as resumable (stored feed), not live; an open one stays live."""
    import json
    from hermes_state_local import POLICY_PREFIX
    closed, open_ = owner.sids[0], owner.sids[1]
    owner.db.create_session('target-closed', source='tui')
    owner.db.create_session('target-open', source='tui')
    owner.db.end_session('target-closed', 'tui_close')
    for sid, target in ((closed, 'target-closed'), (open_, 'target-open')):
        owner.db.set_meta(POLICY_PREFIX + sid, json.dumps({'entry': {'session_id': target}}))
    owner.db.set_meta(POLICY_PREFIX + owner.sids[2], '{not json')
    with owner.db._lock:  # newest, so session.list's 200-row window holds all three
        owner.db._conn.execute('UPDATE sessions SET started_at = 9e9 WHERE id IN (?,?,?)', owner.sids[:3])
    for method in ('session.list', 'session.active_list'):
        ids = {row['id'] for row in await owner.listing(method)}
        assert closed not in ids and open_ in ids and owner.sids[2] in ids, method


@pytest.mark.asyncio
async def test_a_session_retired_during_the_read_is_not_listed_or_restored(owner, monkeypatch):
    from gateway import session_live_listing
    original = session_live_listing._store_facts
    gone = owner.sids[5]

    def retire_midway(db, ids):
        owner.authority.sessions.pop(gone, None)
        return original(db, ids)

    monkeypatch.setattr(session_live_listing, '_store_facts', retire_midway)
    ids = {row['id'] for row in await owner.listing('session.active_list')}
    assert gone not in ids and gone not in owner.authority.sessions


@pytest.mark.asyncio
async def test_a_profile_parked_during_the_off_loop_read_lists_nothing(owner, monkeypatch):
    """The store read is an await: a park (unserve) can retire the authority while it runs, and the
    batch read for it belongs to the home's next owner, not to this listing."""
    from gateway import session_live_listing
    real = session_live_listing._store_facts

    def parked_mid_read(db, batch):
        owner.authority.retiring = True
        return real(db, batch)

    monkeypatch.setattr(session_live_listing, '_store_facts', parked_mid_read)
    actor = SimpleNamespace(profile_id='live', subject='viewer', capabilities=frozenset({'session:read'}))
    monkeypatch.setattr(owner.authority, 'authorize', lambda *a: None)
    assert await session_live_listing.live_entries(owner.authority, actor) == []
