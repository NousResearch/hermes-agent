"""A hosted-room peer whose member gateway restarted mid-turn reports the run ``unknown``: the
room's poll resolves that exact admission and settles the run interrupted, and the member's
queued follower then runs once (the room no longer wedges on "mismatched run status")."""
import asyncio

import pytest

from tests.gateway.test_api_unknown_resolution import (
    _AUTH_HEADERS, _close_fixture, _restarted_api_runs)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


@pytest.mark.asyncio
async def test_peer_poll_resolves_unknown_member_run(tmp_path, monkeypatch):
    client, adapters, store, owner, unknown, _follower, executed, tasks = (
        await _restarted_api_runs(tmp_path, monkeypatch))
    peer, loop = PeerRunsHTTPClient(base_url="http://127.0.0.1:9", api_key="k" * 32), asyncio.get_running_loop()

    def forward(path, *, method="GET", body=None, headers=None, room_grant=None):
        async def call():
            response = await client.request(method, path, json=body, headers=_AUTH_HEADERS)
            payload = await response.json()
            if response.status >= 400:
                raise PeerRunsHTTPError(str(payload), status_code=response.status)
            return payload
        return asyncio.run_coroutine_threadsafe(call(), loop).result(10)

    monkeypatch.setattr(peer, "_request", forward)
    record = {"run_id": unknown["request_id"], "task_id": "room-task", "execution_generation": 1}
    try:
        status = await asyncio.to_thread(peer._poll_receipt, record, grant="room-grant-token")
        assert status["status"] == "interrupted" and status["run_id"] == unknown["request_id"]
        assert ("room-task", 1) in peer._terminal_receipts
        await asyncio.wait_for(owner.sessions["api-session"].task, 10)
        assert executed == ["FOLLOWER"]
    finally:
        await _close_fixture(client, adapters, store, tasks)
