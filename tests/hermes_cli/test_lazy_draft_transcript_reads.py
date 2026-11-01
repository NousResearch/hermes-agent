"""Lazy-draft transcript reads answer 200-empty instead of 404 (#135065).

A ``session.create`` for an empty draft intentionally has no DB row yet (anti
"Untitled" litter); the renderer's immediate ``/messages`` + ``/timeline`` reads
used to 404 and fail-latch the transcript bind on the client. These tests pin
the three-way contract at the REST boundary:

- a live in-memory draft            -> 200, empty transcript payloads
- a persisted session               -> normal DB-backed read, untouched
- a truly unknown session id        -> 404 preserved (only drafts divert)
"""
import httpx
import pytest
from fastapi import FastAPI

import tui_gateway.server as tui_server
from hermes_cli.web_routers.sessions import manage_router
from hermes_state import SessionDB


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setattr('hermes_state._default_db_path', lambda: tmp_path / 'state.db')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    application = FastAPI()
    application.include_router(manage_router)
    return application


@pytest.fixture
def live_draft():
    """Register a fake in-memory lazy draft in the gateway's session registry.

    Mirrors the record shape ``session.create`` mints: ``session_key`` plus the
    ``_db_row_pending`` stamp the registry helper filters on."""
    tui_server._sessions['draft-1'] = {'session_key': 'draft-1', '_db_row_pending': True}
    yield 'draft-1'
    tui_server._sessions.pop('draft-1', None)


@pytest.mark.asyncio
async def test_lazy_draft_messages_answer_200_empty(app, live_draft):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
        r = await client.get(f'/api/sessions/{live_draft}/messages')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['messages'] == []
    assert body['session_id'] == live_draft
    assert body['pagination']['returned'] == 0


@pytest.mark.asyncio
async def test_lazy_draft_timeline_answers_200_empty(app, live_draft):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
        r = await client.get(f'/api/sessions/{live_draft}/timeline')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['entries'] == []
    assert body['pagination']['returned'] == 0
    assert body['pagination']['has_more'] is False


@pytest.mark.asyncio
async def test_persisted_session_messages_still_read_the_db(app, tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('persisted-1', source='cli')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
            r = await client.get('/api/sessions/persisted-1/messages')
    assert r.status_code == 200, r.text
    assert r.json()['session_id'] == 'persisted-1'


@pytest.mark.asyncio
async def test_unknown_session_id_still_404s(app, tmp_path):
    # Production always serves against an initialized store; mirror that so the
    # assertion exercises route logic, not the uninitialized-store 503 guard.
    with SessionDB(db_path=tmp_path / 'state.db'):
        pass
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
        r = await client.get('/api/sessions/never-existed/messages')
    assert r.status_code == 404, r.text


@pytest.fixture
def other_profile(app, monkeypatch, tmp_path):
    """A named profile ``other`` with its own home (and store dir), resolved by the router.

    Timestamp-based durable keys are only unique per profile (#100029), so a live draft
    and a persisted session CAN share one — these tests pin that the draft window stays
    inside its own profile."""
    home = tmp_path / 'profiles' / 'other'
    home.mkdir(parents=True)
    monkeypatch.setattr('hermes_cli.web_routers.sessions._cron_profile_home',
                        lambda name=None: (name or 'other', home))
    return home


def _register_live(key, profile_home=None):
    record = {'session_key': key, '_db_row_pending': True}
    if profile_home is not None:
        record['profile_home'] = profile_home
    tui_server._sessions[key] = record
    return key


@pytest.mark.asyncio
async def test_launch_draft_does_not_mask_other_profiles_persisted_session(app, other_profile, tmp_path):
    """The masking bug this scoping exists for: profile A's live draft must not 200-empty
    profile B's persisted session that carries the same durable key."""
    with SessionDB(db_path=other_profile / 'state.db') as db:
        db.create_session('shared-key', source='cli')
        db.append_message('shared-key', 'user', 'persisted body')
    _register_live('shared-key')  # launch-profile draft (records stamp no profile_home)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
            r = await client.get('/api/sessions/shared-key/messages', params={'profile': 'other'})
    finally:
        tui_server._sessions.pop('shared-key', None)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['session_id'] == 'shared-key'
    assert body['pagination']['returned'] == 1
    assert body['messages'][0]['content'] == 'persisted body'


@pytest.mark.asyncio
async def test_named_profile_draft_answers_200_empty_for_its_own_profile(app, other_profile):
    _register_live('draft-other', profile_home=str(other_profile))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
            r = await client.get('/api/sessions/draft-other/messages', params={'profile': 'other'})
    finally:
        tui_server._sessions.pop('draft-other', None)
    assert r.status_code == 200, r.text
    assert r.json()['pagination']['returned'] == 0


@pytest.mark.asyncio
async def test_named_profile_draft_does_not_leak_into_launch_scope(app, other_profile, tmp_path):
    # The launch store has no such row; the draft belongs to `other`, so the unscoped
    # request must fall through to the store and 404 — not 200-empty off another
    # profile's live session.
    with SessionDB(db_path=tmp_path / 'state.db'):
        pass
    _register_live('draft-other', profile_home=str(other_profile))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
            r = await client.get('/api/sessions/draft-other/messages')
    finally:
        tui_server._sessions.pop('draft-other', None)
    assert r.status_code == 404, r.text
