"""The live listings walk every live session, on the loop the gateway serves I/O from.

A gateway can hold thousands of live session rows; ``session.list`` / ``session.active_list`` walk
the authority's sessions in one request, and every step of that walk was synchronous DB work with
no suspension point, so the whole listing ran as one uninterruptible block on the loop thread.
Anything that needed the loop meanwhile — a platform adapter's stop path, a WS write, delivery —
waited for it; measured on a 4,049-row store the loop stalled past 10 s, the Telegram adapter's
``updater.stop()`` missed its deadline and the adapter was declared fatal (#141236).

These tests pin the two properties the sweep needs: it gives the loop a turn every batch, and it
reads the picker's own columns instead of hydrating a full row (prompt/tool JOIN + token flush)
per session.
"""
import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.session_contract import SessionRef

SESSION_COUNT = 130  # more than one LISTING_YIELD_EVERY batch


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

    yield SimpleNamespace(authority=authority, db=store._db, sids=sids, listing=listing,
                          connection=connection)


@pytest.mark.asyncio
async def test_the_live_listing_gives_the_loop_a_turn_before_it_returns(owner):
    """A listing must not hold the loop for the whole walk: another task runs while it is in flight."""
    order = []

    async def sweep():
        await owner.listing('session.list', limit=200)
        order.append('listing')

    async def competing():
        order.append('competing')

    # Scheduled in this order, so an unyielding sweep appends first and the loop only gets the
    # competing task after the whole walk.
    await asyncio.gather(sweep(), competing())
    assert order == ['competing', 'listing'], order


@pytest.mark.asyncio
async def test_the_active_listing_gives_the_loop_a_turn_before_it_returns(owner):
    order = []

    async def sweep():
        await owner.listing('session.active_list')
        order.append('listing')

    async def competing():
        order.append('competing')

    await asyncio.gather(sweep(), competing())
    assert order == ['competing', 'listing'], order


@pytest.mark.asyncio
async def test_the_live_listing_does_not_hydrate_a_row_per_session(owner, monkeypatch):
    """``get_session`` is a full hydration; the walk needs columns, not the row."""
    hydrated = []
    original = type(owner.db).get_session

    def counting_get_session(db, session_id):
        hydrated.append(session_id)
        return original(db, session_id)

    monkeypatch.setattr(type(owner.db), 'get_session', counting_get_session)
    rows = await owner.listing('session.list', limit=200)
    assert len(rows) == SESSION_COUNT
    # One hydration per session is ``resolve``/``_handle``'s own read (it needs the runtime
    # revision/generation). The listing's own reads must not add any.
    assert len(hydrated) <= SESSION_COUNT, (len(hydrated), SESSION_COUNT)


def test_closed_and_unviewed_reads_the_row_without_hydrating(monkeypatch):
    """The per-candidate close test reads ``ended_at``, and must not hydrate the whole row."""
    from gateway import session_local_close

    monkeypatch.setattr(session_local_close, 'local_receipt',
                        lambda db, session_id: {'entry': {'session_id': 'target'}})

    class Db:
        def __init__(self):
            self.hydrated = []

        def get_session(self, session_id):
            self.hydrated.append(session_id)
            return {'id': session_id, 'ended_at': 1.0}

        def get_session_list_row(self, session_id):
            return {'id': session_id, 'ended_at': 1.0}

    db = Db()
    authority = SimpleNamespace(sessions={'live-1': SimpleNamespace(subscribers={})}, db=db)
    handle = SimpleNamespace(execution_state='idle')
    assert session_local_close.closed_and_unviewed(authority, 'live-1', handle) is True
    assert db.hydrated == []


def test_closed_and_unviewed_keeps_an_open_row_listed(monkeypatch):
    """The light read keeps the rule: an open target is listed, a closed one is not."""
    from gateway import session_local_close

    monkeypatch.setattr(session_local_close, 'local_receipt',
                        lambda db, session_id: {'entry': {'session_id': 'target'}})
    authority = SimpleNamespace(
        sessions={'live-1': SimpleNamespace(subscribers={})},
        db=SimpleNamespace(get_session_list_row=lambda session_id: {'id': session_id, 'ended_at': None},
                           get_session=lambda session_id: {'id': session_id, 'ended_at': None}))
    assert session_local_close.closed_and_unviewed(
        authority, 'live-1', SimpleNamespace(execution_state='idle')) is False
