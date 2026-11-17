"""The durable session event log: gapless seqs that survive a restart, served as an HTTP long-poll."""
import asyncio
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


def _bind(owner, home, session_id):
    from gateway.session_authority import LiveSession
    from gateway.session_event_log import EventLog
    owner.sessions.log = EventLog(home)
    owner.sessions[session_id] = live = LiveSession(None, f'route:{session_id}')
    return live.event_stream


@pytest.mark.asyncio
async def test_long_poll_waits_for_records_skips_deltas_and_keeps_seq_across_restart(api, owner, tmp_path):
    from gateway.platforms import api_server_session_events
    owner.db.create_session('s1', source='api_server')
    stream = _bind(owner, tmp_path, 's1')
    app = web.Application()
    for method, path, handler in api_server_session_events.http_routes(api):
        app.router.add_route(method, path, handler)
    async with TestClient(TestServer(app)) as client:
        started = time.monotonic()
        poll = asyncio.ensure_future(client.get('/api/sessions/s1/events?after=0&wait_ms=5000'))
        await asyncio.sleep(0.4)  # health: allow HX013 -- the poll must still be blocked after this delay
        stream.publish('s1', {'text': 'Hel'}, event_type='message.delta')
        stream.publish('s1', {'text': 'Hello'}, event_type='message.complete')
        body = await (await poll).json()
        assert time.monotonic() - started >= 0.4  # it blocked until the record landed
        assert [(e['seq'], e['type'], e['payload']['text']) for e in body['events']] == [(1, 'message.complete', 'Hello')]

        # A restart: a fresh log over the same file continues the numbering, and the old events stay.
        stream = _bind(owner, tmp_path, 's1')
        stream.publish('s1', {'name': 'terminal'}, event_type='tool.start')
        stream.publish('s1', {'text': 'done'}, event_type='message.complete')
        owner.sessions.log.flush()
        body = await (await client.get('/api/sessions/s1/events?after=0&types=tool.*')).json()
        assert [(e['seq'], e['type']) for e in body['events']] == [(2, 'tool.start')]
        assert body['last_seq'] == 3
        idle = time.monotonic()
        body = await (await client.get('/api/sessions/s1/events?after=3&wait_ms=300')).json()
        assert body['events'] == [] and time.monotonic() - idle >= 0.3


def test_event_log_is_opt_in_per_profile_config(tmp_path):
    from gateway.session_event_log import enabled_in_home, log_for_home, release_home
    on, off = tmp_path / 'on', tmp_path / 'off'
    on.mkdir()
    off.mkdir()
    (on / 'config.yaml').write_text('sessions:\n  event_log: true\n', encoding='utf-8')
    (off / 'config.yaml').write_text('sessions:\n  auto_prune: false\n', encoding='utf-8')
    assert enabled_in_home(on) is True and enabled_in_home(off) is False
    assert log_for_home(off, enabled_in_home(off)) is None
    log = log_for_home(on, enabled_in_home(on))
    assert log is not None and log.path == on.resolve() / 'session_events.db'
    release_home(on)
