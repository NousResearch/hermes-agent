"""Durable runs must not mistake a recycled container PID for their live owner."""

import hashlib
import os
import subprocess
import sys

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from gateway.status import get_process_start_time
from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter, _use_idempotency_db


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_closed", [False, True])
async def test_durable_run_tracks_store_owner_lifetime(tmp_path, owner_closed):
    path = tmp_path / "runs.db"
    scope = hashlib.sha256(b"default\0unauthenticated-test-listener").hexdigest()
    owner = RunIdempotencyStore(str(path))
    owner.reserve(
        scope, "request", "fingerprint", "run_previous",
        {"run_id": "run_previous", "status": "running"},
        owner_pid=os.getpid(), owner_started=get_process_start_time(os.getpid()) or 0,
    )
    observer = _make_adapter()
    _use_idempotency_db(observer, path)
    try:
        async with TestClient(TestServer(_create_runs_app(observer))) as client:
            live = await client.get("/v1/runs/run_previous")
            assert (await live.json())["status"] == "running"
            if owner_closed:
                owner.close()
            response = await client.get("/v1/runs/run_previous")
            body = await response.json()
            assert response.status == 200
            assert body["status"] == ("interrupted" if owner_closed else "running")
            if owner_closed:
                stopped = await client.post("/v1/runs/run_previous/stop")
                assert stopped.status == 200
                assert (await stopped.json())["status"] == "interrupted"
    finally:
        observer._run_idempotency_store.close()
        if not owner_closed:
            owner.close()


def test_owner_lock_released_after_process_kill(tmp_path):
    path = tmp_path / "runs.db"
    script = """
import sys
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
store = RunIdempotencyStore(sys.argv[1])
store.reserve("scope", "request", "fingerprint", "run_previous",
              {"run_id": "run_previous", "status": "running"})
print("ready", flush=True)
sys.stdin.read()
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    ) as owner:
        try:
            assert owner.stdout.readline().strip() == "ready"
            observer = RunIdempotencyStore(str(path))
            try:
                token = observer.status_for_run("scope", "run_previous")["owner_token"]
                assert observer.owner_alive(token)
                owner.kill()
                owner.wait(timeout=10)
                assert not observer.owner_alive(token)
                outcome, record = observer.reserve(
                    "scope", "request", "fingerprint", "run_replacement", {},
                )
                assert outcome == "reused"
                assert record["run_id"] == "run_previous"
            finally:
                observer.close()
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait(timeout=10)
