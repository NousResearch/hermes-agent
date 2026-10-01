"""The owner can find the exact participant identity a Stop needs, from the participant."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server_run_scope import room_run_scope_key
from tests.gateway.test_api_group_owner_stop import (  # noqa: F401 - adapter is a fixture
    OWNER, PARTICIPANTS, STOP, adapter, app_for, command, dispatch, identity, invite, participation, submit,
)
from tests.gateway.test_api_server_runs import _make_slow_agent


def record(adapter, participant, run_id):
    adapter._run_idempotency_store.reserve(
        room_run_scope_key(participant), f"key-{run_id}", "fingerprint", run_id,
        {"run_id": run_id, "status": "running", "output": "PRIVATE_OUTPUT_MUST_NOT_LEAK"}, identity=participant)


@pytest.mark.asyncio
async def test_owner_lists_a_real_participant_and_stops_it_from_the_list(adapter, monkeypatch):
    agent, ready, interrupted = _make_slow_agent()
    monkeypatch.setattr(adapter, "_create_agent", MagicMock(return_value=agent))
    async with TestClient(TestServer(app_for(adapter))) as cli:
        empty = await cli.get(PARTICIPANTS, headers=OWNER)
        assert empty.status == 200 and (await empty.json())["data"] == []
        invited = await invite(cli)
        payload = dispatch(invited)
        accepted = await submit(cli, invited, payload)
        assert accepted.status == 202, await accepted.json()
        assert await asyncio.to_thread(ready.wait, 3)
        listed = await cli.get(PARTICIPANTS, headers=OWNER)
        body = await listed.json()
        assert listed.status == 200, body
        [item] = body["data"]
        assert item["participant"] == participation(payload)
        assert item["admissions_frozen"] is False and item["frozen_at"] is None
        assert item["counts"] == {"total": 1, "terminal": 0, "nonterminal": 1, "unknown": 0}
        assert body["truncated"] is False and "PRIVATE_" not in json.dumps(body)
        # The listed identity is exactly what Stop needs.
        stopped = await cli.post(STOP, headers=OWNER, json=command(item["participant"]))
        reply = await stopped.json()
        assert stopped.status == 200, reply
        assert await asyncio.to_thread(interrupted.wait, 1)
        await asyncio.wait_for(asyncio.gather(*adapter._active_run_tasks.values()), timeout=3)
        [after] = (await (await cli.get(PARTICIPANTS, headers=OWNER)).json())["data"]
        assert after["admissions_frozen"] is True and after["frozen_at"] == reply["frozen_at"]
        assert after["counts"]["nonterminal"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "HermesRoom scoped"])
async def test_list_requires_the_installation_owner(adapter, authorization):
    record(adapter, identity(), "run-known")
    headers = {} if authorization is None else {"Authorization": authorization}
    async with TestClient(TestServer(app_for(adapter))) as cli:
        assert (await cli.get(PARTICIPANTS, headers=headers)).status == 401
        prefixed = await cli.get("/p/default" + PARTICIPANTS, headers=OWNER)
        assert prefixed.status == 403, await prefixed.json()


@pytest.mark.asyncio
async def test_list_shows_only_this_gateways_default_profile_participants(adapter):
    known = identity()
    record(adapter, known, "run-default")
    record(adapter, {**known, "target_profile": "private-profile"}, "run-named")
    record(adapter, {**known, "target_install_id": "install:another-gateway"}, "run-elsewhere")
    async with TestClient(TestServer(app_for(adapter))) as cli:
        body = await (await cli.get(PARTICIPANTS, headers=OWNER)).json()
    assert [item["participant"] for item in body["data"]] == [known]
    assert body["coverage"] == "default_profile_participants_in_this_runs_store"
