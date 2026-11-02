"""API admission never blocks the shared owner loop on a held SQLite writer (every HTTP door)."""
import asyncio
import sqlite3
import threading
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

KEY = 'off-loop-admission-key-long-enough'
HOLD_S = 1.5
AUTH = {'Authorization': 'Bearer ' + KEY}

DOORS = [
    ('chat', '/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'hi'}]}, {}, 200),
    ('responses', '/v1/responses', {'input': 'hi'}, {'Idempotency-Key': 'resp-1'}, 200),
    ('runs', '/v1/runs', {'input': 'hi'}, {}, 202),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('name,path,body,headers,expected', DOORS, ids=[d[0] for d in DOORS])
async def test_admission_waits_for_the_writer_off_the_owner_loop(api, owner, monkeypatch, name, path, body,
                                                                headers, expected):
    from gateway.platforms import api_server_runs

    async def handle(event):
        from gateway.session_results import execution_result
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    async def no_execution(*args, **kwargs):
        pass
    owner.runner._handle_message = handle
    monkeypatch.setattr(api_server_runs, '_execute_run', no_execution)
    api._api_key = KEY
    app = web.Application()
    app.router.add_post(path, getattr(api, {'chat': '_handle_chat_completions', 'responses': '_handle_responses',
                                            'runs': '_handle_runs'}[name]))
    async with TestClient(TestServer(app)) as client:
        # Another process holds the profile's state.db writer for HOLD_S (a long transaction).
        holder = sqlite3.connect(owner.db.db_path, timeout=0, isolation_level=None, check_same_thread=False)
        holder.execute('BEGIN IMMEDIATE')
        threading.Timer(HOLD_S, lambda: holder.execute('COMMIT')).start()
        request = asyncio.create_task(client.post(path, json=body, headers={**AUTH, **headers}))
        gaps, last = [], time.monotonic()
        while not request.done():
            await asyncio.sleep(0.02)
            now = time.monotonic()
            gaps.append(now - last)
            last = now
        response = await request
        holder.close()
        assert response.status == expected, await response.text()
    # Other sessions, timers and sockets kept running while the admission waited on the writer.
    assert max(gaps) < HOLD_S / 3, f'owner loop stalled {max(gaps):.2f}s behind the admission write'
