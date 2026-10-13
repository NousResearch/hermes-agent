"""Successor grants can read and stop the run they replay, including after restart."""
import asyncio
import json
import threading
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server_run_authority import room_authority, room_run_scope
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tests.gateway.test_api_server_room_cancellation import _adapter, _headers, _invitation
from tests.gateway.test_room_authority_lineage import _member_claims, invite
from tests.gateway.test_room_cancellation_retirement_http import app


def _blocking_agent(started: threading.Event, release: threading.Event):
    agent = MagicMock()

    def run_conversation(*_args, **_kwargs):
        started.set()
        assert release.wait(20)
        return {"final_response": "once", "messages": []}

    agent.run_conversation.side_effect = run_conversation
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    return agent


def _with_stop_by_id(adapter):
    application = app(adapter)
    application.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return application


async def _epoch2(cli):
    invited = await invite(cli, {
        "room_id": "room-1", "home_install_id": "home", "authority_gateway_id": "home",
        "authority_epoch": 2, "member_id": "member-1"})
    assert invited.status == 201, await invited.text()
    return (await invited.json())["grant"]


def _replay_body(body):
    return {"input": body["input"], "hosted_room_dispatch": {
        **body["hosted_room_dispatch"], "authority_epoch": 2}}


@pytest.mark.asyncio
async def test_successor_grant_reads_and_stops_the_live_run(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path / "runs.db")
    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(adapter, "_create_agent", MagicMock(return_value=_blocking_agent(started, release)))
    try:
        async with TestClient(TestServer(_with_stop_by_id(adapter))) as cli:
            grant, body = await _invitation(cli)
            first = await cli.post("/v1/runs", headers=_headers(grant), json=body)
            assert first.status == 202, await first.text()
            run_id = (await first.json())["run_id"]
            assert started.wait(10)
            store = adapter._run_idempotency_store
            store._conn.execute("UPDATE run_idempotency SET retention_until=1")
            store._conn.commit()
            grant2 = await _epoch2(cli)
            headers2 = {**_headers(grant2), "Idempotency-Key": "room:task-1:1"}
            replay = await cli.post("/v1/runs", headers=headers2, json=_replay_body(body))
            assert replay.status == 202, await replay.text()
            assert (await replay.json())["run_id"] == run_id
            assert adapter._create_agent.call_count == 1

            status = await cli.get(f"/v1/runs/{run_id}", headers=_headers(grant2))
            payload = await status.json()
            assert status.status == 200, payload
            assert payload["run_id"] == run_id
            assert payload["status"] in {"queued", "running"}
            assert store._conn.execute(
                "SELECT retention_until FROM run_idempotency").fetchone()[0] > 1
            original = await cli.get(f"/v1/runs/{run_id}", headers=_headers(grant))
            assert original.status == 403

            stopped = await cli.post(f"/v1/runs/{run_id}/stop", headers=_headers(grant2), json={})
            body_stop = await stopped.json()
            assert stopped.status == 200, body_stop
            assert body_stop["status"] == "stopping"
            assert store.stop_requested(run_id) is True
            assert adapter._active_run_tasks[run_id].done() is False
    finally:
        release.set()
        tasks = tuple(adapter._active_run_tasks.values())
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_restarted_successor_stop_interrupts_a_dead_owner(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path / "runs.db")
    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(adapter, "_create_agent", MagicMock(return_value=_blocking_agent(started, release)))
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            grant, body = await _invitation(cli)
            first = await cli.post("/v1/runs", headers=_headers(grant), json=body)
            assert first.status == 202, await first.text()
            run_id = (await first.json())["run_id"]
            assert started.wait(10)
            grant2 = await _epoch2(cli)
            adapter._run_idempotency_store._conn.execute(
                "UPDATE run_idempotency SET owner_pid=99999999, owner_started=0 WHERE run_id=?",
                (run_id,))
            adapter._run_idempotency_store._conn.commit()
            for table in (
                adapter._run_owners, adapter._run_statuses, adapter._run_idempotency_ids,
                adapter._active_run_agents, adapter._active_run_tasks, adapter._stopping_run_ids,
            ):
                table.clear()
            stopped = await cli.post(
                "/v1/runs/stop", headers={**_headers(grant2), "Idempotency-Key": "room:task-1:1"},
                json=_replay_body(body))
            payload = await stopped.json()
            assert stopped.status == 200, payload
            assert payload["status"] == "interrupted"
            row = adapter._run_idempotency_store._conn.execute(
                "SELECT status_json, stop_requested FROM run_idempotency WHERE run_id=?",
                (run_id,)).fetchone()
            assert row[1] == 1
            assert '"status":"interrupted"' in row[0]
            follow = await cli.get(f"/v1/runs/{run_id}", headers=_headers(grant2))
            assert follow.status == 200, await follow.text()
            assert (await follow.json())["status"] == "interrupted"
    finally:
        release.set()
        adapter._run_idempotency_store.close()


def test_expired_completed_generation_stays_fenced(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    try:
        claims = _member_claims()
        scope, authority = room_run_scope(claims), room_authority(claims)
        assert store.observe_room_authority(scope, authority, claims=claims)
        assert store.reserve(
            scope, "room:task-1:1", "fp", "run-done", {"status": "completed", "output": "once"},
            retention_until=1, room_authority=authority)[0] == "created"
        assert store.reserve(
            scope, "room:task-2:1", "fp", "run-stopped", {"status": "completed", "output": "kept"},
            retention_until=1, room_authority=authority)[0] == "created"
        store._conn.execute("UPDATE run_idempotency SET stop_requested=1 WHERE run_id='run-stopped'")
        store._conn.execute("UPDATE run_idempotency SET updated_at=0")
        store._conn.commit()
        successor = _member_claims(authority_epoch=2)
        assert store.observe_room_authority(
            room_run_scope(successor), room_authority(successor), claims=successor)
        outcome, record = store.reserve(
            room_run_scope(successor), "room:task-1:1", "fp-2", "run-again", {"status": "queued"},
            room_authority=room_authority(successor))
        assert outcome == "reused" and record["run_id"] == "run-done"
        fenced = store._conn.execute(
            "SELECT status_json FROM run_room_fenced_attempts WHERE idempotency_key='room:task-1:1'"
        ).fetchone()
        assert fenced is not None
        fenced_status = json.loads(fenced[0])
        assert "output" not in fenced_status
        assert fenced_status["body_discarded"] is True
        kept = store.reserve(
            room_run_scope(successor), "room:task-2:1", "fp", "run-new", {"status": "queued"},
            room_authority=room_authority(successor))
        assert kept[0] == "reused" and kept[1]["run_id"] == "run-stopped"
    finally:
        store.close()


def test_duplicate_lineage_rows_are_both_stopped(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    try:
        claims = _member_claims()
        scope, authority = room_run_scope(claims), room_authority(claims)
        assert store.observe_room_authority(scope, authority, claims=claims)
        assert store.reserve(
            scope, "room:task-1:1", "fp-live", "run-live", {"status": "running"},
            room_authority=authority)[0] == "created"
        key = authority[0]
        store._conn.execute(
            """INSERT INTO run_idempotency(
               scope, idempotency_key, fingerprint, run_id, status_json, owner_pid, owner_started,
               retention_until, created_at, updated_at, room_authority_key, room_authority_epoch)
               VALUES('other-scope','room:task-1:1','fp-other','run-other',?,0,0,0,1,1,?,1)""",
            ('{"status":"running"}', key))
        store._conn.commit()
        successor = _member_claims(authority_epoch=2)
        assert store.observe_room_authority(
            room_run_scope(successor), room_authority(successor), claims=successor)
        outcome, record = store.reserve(
            room_run_scope(successor), "room:task-1:1", "fp-2", "run-stop",
            {"status": "cancelled", "admission_cancelled": True}, cancel_if_missing=True,
            room_authority=room_authority(successor))
        assert outcome == "reused" and record["run_id"] == "run-live"
        flags = store._conn.execute(
            "SELECT run_id, stop_requested FROM run_idempotency ORDER BY run_id").fetchall()
        assert flags == [("run-live", 1), ("run-other", 1)]
    finally:
        store.close()


def test_run_id_stop_flags_the_sibling_lineage_row(tmp_path):
    """The receipt route flags every pre-fix row of the generation, not only its run id."""
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    try:
        claims = _member_claims()
        scope, authority = room_run_scope(claims), room_authority(claims)
        assert store.observe_room_authority(scope, authority, claims=claims)
        assert store.reserve(
            scope, "room:task-1:1", "fp-live", "run-live", {"status": "running"},
            room_authority=authority)[0] == "created"
        key = authority[0]
        store._conn.execute(
            """INSERT INTO run_idempotency(
               scope, idempotency_key, fingerprint, run_id, status_json, owner_pid, owner_started,
               retention_until, created_at, updated_at, room_authority_key, room_authority_epoch)
               VALUES('other-scope','room:task-1:1','fp-other','run-other',?,0,0,0,1,1,?,1)""",
            ('{"status":"running"}', key))
        store._conn.commit()
        successor = _member_claims(authority_epoch=2)
        successor_scope, successor_authority = room_run_scope(successor), room_authority(successor)
        assert store.observe_room_authority(successor_scope, successor_authority, claims=successor)
        store.request_stop(successor_scope, "run-live", room_authority=successor_authority)
        flags = store._conn.execute(
            "SELECT run_id, stop_requested FROM run_idempotency ORDER BY run_id").fetchall()
        assert flags == [("run-live", 1), ("run-other", 1)]
        assert store.stop_requested("run-other") is True
    finally:
        store.close()
