"""Production-route invariants for separately authorized cron execution receipts."""

from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


JOB_ID = "aabbccddeeff"
SCHEDULED_AT = "2026-09-07T10:00:00+00:00"
OBSERVER_KEY = "observer-secret-value-123456"
API_KEY = "general-api-secret-value-123456"


def _receipt_output():
    return json.dumps({
        "version": "hermes.cron.receipt.v1",
        "status": "completed",
        "receipt_id": "receipt-1",
        "result_sha256": "a" * 64,
    })


def _app(adapter):
    app = web.Application()
    adapter._register_http_routes(app)
    return app


@pytest.mark.asyncio
async def test_scheduler_receipt_route_is_dedicated_content_free_and_fail_closed(
    monkeypatch, tmp_path
):
    """Real no-agent stdout -> atomic ledger -> the one registered observer projection."""
    home = tmp_path / ".hermes"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CRON_RECEIPT_OBSERVER_KEY", OBSERVER_KEY)
    import cron.executions as executions
    import cron.scheduler as scheduler

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    script = scripts / "receipt.sh"
    script.write_text("#!/bin/sh\nprintf '%s\\n' '" + _receipt_output() + "'\n")
    script.chmod(0o700)
    # Keep this integration at the scheduler seam: its real no-agent script path,
    # ledger, and aiohttp registration run; unrelated jobs-store/delivery effects
    # stay local deterministic stubs.
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda _job: False)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: None)
    job = {
        "id": JOB_ID,
        "name": "receipt script",
        "no_agent": True,
        "script": "receipt.sh",
        "deliver": "local",
        "_scheduled_instant": SCHEDULED_AT,
    }
    assert scheduler.run_one_job(job) is True
    execution_id = job["execution_id"]

    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"key": API_KEY},
        )
    )
    path = f"/v1/cron/jobs/{JOB_ID}/executions/{execution_id}/receipt"
    async with TestClient(TestServer(_app(adapter))) as client:
        for headers in (
            {},
            {"Authorization": f"Bearer {API_KEY}"},
            {"Authorization": "Bearer wrong-observer-key"},
        ):
            response = await client.get(path, headers=headers)
            assert response.status == 401
            assert response.headers["Cache-Control"] == "no-store"
        response = await client.get(
            path,
            headers=[
                ("Authorization", f"Bearer {OBSERVER_KEY}"),
                ("Authorization", "Bearer wrong-observer-key"),
            ],
        )
        assert response.status == 401
        assert response.headers["Cache-Control"] == "no-store"
        response = await client.get(
            path, headers={"Authorization": f"Bearer {OBSERVER_KEY}"}
        )
        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        body = await response.json()
        assert set(body) == {
            "version",
            "job_id",
            "execution_id",
            "scheduled_at",
            "fired_at",
            "finished_at",
            "fire_authority_sha256",
            "receipt_id",
            "result_sha256",
        }
        assert body["job_id"] == JOB_ID and body["execution_id"] == execution_id
        assert (
            body["scheduled_at"] == SCHEDULED_AT and body["receipt_id"] == "receipt-1"
        )
        assert set(body).isdisjoint({"output", "prompt", "path", "pid"})

        assert (
            await client.get(
                "/v1/cron/jobs", headers={"Authorization": f"Bearer {OBSERVER_KEY}"}
            )
        ).status == 404
        assert (
            await client.post(path, headers={"Authorization": f"Bearer {OBSERVER_KEY}"})
        ).status == 405
        assert (
            await client.head(path, headers={"Authorization": f"Bearer {OBSERVER_KEY}"})
        ).status == 405
        assert (
            await client.get(
                "/api/jobs", headers={"Authorization": f"Bearer {OBSERVER_KEY}"}
            )
        ).status == 401
        assert (
            await client.post(
                "/api/jobs", headers={"Authorization": f"Bearer {OBSERVER_KEY}"}
            )
        ).status == 401
        assert (
            await client.get(
                f"/p/default/v1/cron/jobs/{JOB_ID}/executions/{execution_id}/receipt",
                headers={"Authorization": f"Bearer {OBSERVER_KEY}"},
            )
        ).status == 404
        assert (
            await client.get(
                path.replace(JOB_ID, "ffeeddccbbaa"),
                headers={"Authorization": f"Bearer {OBSERVER_KEY}"},
            )
        ).status == 404
        assert (
            await client.get(
                path.replace(execution_id, "0" * 32),
                headers={"Authorization": f"Bearer {OBSERVER_KEY}"},
            )
        ).status == 404
        assert (
            await client.get(
                path.replace(execution_id, "not-an-execution-id"),
                headers={"Authorization": f"Bearer {OBSERVER_KEY}"},
            )
        ).status == 404

        in_progress = executions.create_execution(
            JOB_ID, source="builtin", scheduled_instant=SCHEDULED_AT
        )
        response = await client.get(
            f"/v1/cron/jobs/{JOB_ID}/executions/{in_progress['id']}/receipt",
            headers={"Authorization": f"Bearer {OBSERVER_KEY}"},
        )
        assert response.status == 404
        assert response.headers["Cache-Control"] == "no-store"
        assert executions.mark_execution_running(in_progress["id"]) is not None
        assert (
            executions.finish_execution(
                in_progress["id"], success=False, error="failed"
            )
            is not None
        )
        assert (
            await client.get(
                f"/v1/cron/jobs/{JOB_ID}/executions/{in_progress['id']}/receipt",
                headers={"Authorization": f"Bearer {OBSERVER_KEY}"},
            )
        ).status == 404
        unknown = executions.create_execution(
            JOB_ID, source="builtin", scheduled_instant=SCHEDULED_AT
        )
        assert executions.mark_execution_running(unknown["id"]) is not None
        with monkeypatch.context() as recovered:
            recovered.setattr(executions, "_PROCESS_ID", "replacement-owner")
            recovered.setattr(
                executions, "_owner_is_live", lambda _pid, _started: False
            )
            assert executions.recover_interrupted_executions() == 1
        assert (
            await client.get(
                f"/v1/cron/jobs/{JOB_ID}/executions/{unknown['id']}/receipt",
                headers={"Authorization": f"Bearer {OBSERVER_KEY}"},
            )
        ).status == 404

    monkeypatch.setenv("CRON_RECEIPT_OBSERVER_KEY", "placeholder")
    placeholder = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"key": API_KEY},
        )
    )
    async with TestClient(TestServer(_app(placeholder))) as client:
        assert (
            await client.get(path, headers={"Authorization": "Bearer placeholder"})
        ).status == 404
    monkeypatch.setenv("CRON_RECEIPT_OBSERVER_KEY", API_KEY + " ")
    reused = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))
    async with TestClient(TestServer(_app(reused))) as client:
        assert (
            await client.get(path, headers={"Authorization": f"Bearer {API_KEY}"})
        ).status == 404
    monkeypatch.delenv("CRON_RECEIPT_OBSERVER_KEY")
    config_only = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"key": API_KEY, "cron_receipt_observer_key": OBSERVER_KEY},
        )
    )
    async with TestClient(TestServer(_app(config_only))) as client:
        assert (
            await client.get(path, headers={"Authorization": f"Bearer {OBSERVER_KEY}"})
        ).status == 404


@pytest.mark.asyncio
async def test_observer_receipt_missing_store_is_non_mutating(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    ledger = tmp_path / "missing-profile" / "cron" / "executions.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CRON_RECEIPT_OBSERVER_KEY", OBSERVER_KEY)
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", ledger)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))
    path = f"/v1/cron/jobs/{JOB_ID}/executions/{'0' * 32}/receipt"
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.get(path, headers={"Authorization": f"Bearer {OBSERVER_KEY}"})

    assert response.status == 404
    assert response.headers["Cache-Control"] == "no-store"
    assert not ledger.parent.exists()
