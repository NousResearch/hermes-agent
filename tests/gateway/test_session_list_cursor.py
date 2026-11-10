"""``session.list``'s stored scope pages with ``cursor`` / ``next_cursor`` and filters by ``query`` (#141982)."""
import base64
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest


@asynccontextmanager
async def _stored(tmp_path, monkeypatch, rows, cwd=None):
    """A native reader over old local rows *rows*: ``(id, title, offered)``, oldest first. A row that is not
    offered belongs to another profile, so the stored listing reads it from storage and skips it.
    *cwd*: the folder the rows ran in (default: *tmp_path*)."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    for sid, title, offered in rows:
        db.create_session(sid, source='cli', model='m', cwd=cwd or str(tmp_path),
                          profile_name=None if offered else 'other')
        if title:
            db.set_session_title(sid, title)
        db.append_message(sid, 'user', f'{sid} question')
        db.end_session(sid, 'cli_exit')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': str(tmp_path), 'instance_id': 'owner'})

    async def listing(**params):
        return await native.dispatch({'id': 1, 'method': 'session.list', 'params': params})
    try:
        yield listing
    finally:
        await native.close()
        store.close_all_db_handles()


async def _follow(listing, **params):
    """Every page of a stored listing, following ``next_cursor`` until it is null."""
    pages, cursor = [], None
    while True:
        reply = await listing(scope='stored', **params, **({'cursor': cursor} if cursor else {}))
        assert 'result' in reply, reply
        pages.append([row['id'] for row in reply['result']['sessions']])
        cursor = reply['result']['next_cursor']
        if cursor is None:
            return pages
        assert len(pages) < 100


@pytest.mark.asyncio
async def test_following_next_cursor_lists_every_conversation_once_in_order(tmp_path, monkeypatch):
    # Every third storage row is skipped, and the rows span more than one 200-row storage batch.
    rows = [(f's{n:03}', None, n % 3 != 1) for n in range(240)]
    async with _stored(tmp_path, monkeypatch, rows) as listing:
        whole = (await listing(scope='stored'))['result']
        assert whole['next_cursor'] is None
        unpaged = [row['id'] for row in whole['sessions']]
        assert len(unpaged) == 160
        pages = await _follow(listing, limit=7)
        assert all(len(page) == 7 for page in pages[:-1])
        assert [sid for page in pages for sid in page] == unpaged


@pytest.mark.asyncio
async def test_query_lists_only_matching_titles_and_pages_with_its_own_cursor(tmp_path, monkeypatch):
    rows = [(f'old{n}', f'needle notes {n}', True) for n in range(3)]
    rows += [(f'mid{n}', f'skip me needle {n}', False) for n in range(2)]
    rows += [(f'new{n:02}', f'chat {n}', True) for n in range(12)]
    async with _stored(tmp_path, monkeypatch, rows) as listing:
        first = (await listing(scope='stored', limit=5))['result']
        assert not {'old0', 'old1', 'old2'} & {row['id'] for row in first['sessions']}
        pages = await _follow(listing, query='needle', limit=2)
        assert sorted(sid for page in pages for sid in page) == ['old0', 'old1', 'old2']
        assert len(pages) == 2
        assert (await listing(scope='stored', query='no such title'))['result'] == {
            'sessions': [], 'scope': 'stored', 'next_cursor': None}


@pytest.mark.asyncio
async def test_a_cursor_continues_a_listing_with_long_non_ascii_filters(tmp_path, monkeypatch):
    # The longest title a session can have, in a script JSON escapes, in a deeply nested folder.
    title = '\u4f1a\u8bdd\u6807\u9898' * 24
    folder = str(tmp_path.joinpath(*['\u9879\u76ee' * 20] * 12))
    rows = [(f'deep{n}', f'{title} {n}', True) for n in range(3)]
    async with _stored(tmp_path, monkeypatch, rows, cwd=folder) as listing:
        pages = await _follow(listing, query=title, workspace=folder, limit=1)
        assert sorted(sid for page in pages for sid in page) == ['deep0', 'deep1', 'deep2']


@pytest.mark.asyncio
async def test_cursor_and_query_refusals(tmp_path, monkeypatch):
    rows = [(f'needle{n}', f'needle {n}', True) for n in range(4)]
    async with _stored(tmp_path, monkeypatch, rows) as listing:
        page = (await listing(scope='stored', query='needle', limit=1))['result']
        cursor = page['next_cursor']
        assert isinstance(cursor, str)
        assert 'result' in await listing(scope='stored', query='needle', limit=3, cursor=cursor)
        # The same cursor with an offset no store can hold: refused like any cursor the method did not issue.
        held = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
        held[1] = 2**63
        too_large = base64.urlsafe_b64encode(json.dumps(held).encode()).decode().rstrip('=')
        refused = [
            {'scope': 'stored', 'query': 'other', 'cursor': cursor},
            {'scope': 'stored', 'cursor': cursor},
            {'scope': 'stored', 'query': 'needle', 'workspace': str(tmp_path / 'x'), 'cursor': cursor},
            {'scope': 'stored', 'query': 'needle', 'include_hidden': True, 'cursor': cursor},
            {'cursor': cursor},
            {'scope': 'live', 'query': 'needle'},
            {'query': 'needle'},
            {'title': 'needle 1', 'query': 'needle'},
            {'scope': 'stored', 'query': '   '},
            {'scope': 'stored', 'query': 'x' * 201},
            {'scope': 'stored', 'cursor': 'not a cursor'},
            {'scope': 'stored', 'cursor': cursor[:-4]},
            {'scope': 'stored', 'query': 'needle', 'cursor': too_large},
        ]
        for params in refused:
            reply = await listing(**params)
            assert reply['error']['data']['reason'] == 'invalid_params', (params, reply)
