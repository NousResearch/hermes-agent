"""Desktop plugin WebSockets on a local gateway ride the private native-HTTP grant (PLG-1).

A local Desktop has no static dashboard token (``authMode: native``), so the legacy ``?token=``
upgrade always failed and Kanban fell back to polling. The plugin socket presents the same
one-use ``native-http`` grant its REST twin sends, in the same header, under the same peer/Origin
rules; anything else stays refused.
"""
import asyncio
import json

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from tests.gateway.test_native_http_auth import daemon, headers, ticket


async def _upgrade(origin, path, extra=None):
    url = origin.replace('http:', 'ws:') + path
    try:
        async with connect(url, additional_headers=extra or {}, open_timeout=10):
            return 'open'
    except InvalidStatus as exc:
        return exc.response.status_code


@pytest.mark.platforms("linux")
def test_plugin_socket_accepts_native_grant_only(tmp_path):
    with daemon(tmp_path) as (home, descriptor):
        origin = descriptor['api_origin']
        events = '/api/plugins/kanban/events'

        def up(path=events, extra=None):
            return asyncio.run(_upgrade(origin, path, extra))

        assert up() != 'open'
        assert up(events + '?token=') != 'open'
        grant = ticket(home, descriptor)
        assert up(extra=headers(grant)) == 'open'
        assert up(extra=headers(grant)) != 'open'  # one use
        assert up(extra=headers('invalid')) != 'open'
        for purpose in ('interactive', 'worker-adoption'):
            assert up(extra=headers(ticket(home, descriptor, purpose))) != 'open', purpose
        for browser_origin in ('https://evil.example', 'null'):
            assert up(extra={**headers(ticket(home, descriptor)), 'Origin': browser_origin}) != 'open'
        assert up(events + '?profile=other', headers(ticket(home, descriptor))) != 'open'
        assert up(events + '?profile=default', headers(ticket(home, descriptor))) == 'open'
        # The grant is not a query credential, and opens plugin sockets only.
        assert up(events + '?ticket=' + ticket(home, descriptor)) != 'open'
        for core in ('/api/ws', '/api/pty', '/api/audio/speak-stream'):
            assert up(core, headers(ticket(home, descriptor))) != 'open', core
    print(json.dumps({'plugin_ws_native_grant': 'accepted once; replay/purpose/origin/profile refused'}))
