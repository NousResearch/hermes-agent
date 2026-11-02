"""Settled webhook admissions remain closeable without repeating their turn."""
import asyncio
import hashlib
import hmac
import json
from contextlib import asynccontextmanager, suppress

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter
from hermes_state_runtime import claim_session_input, settle_session_input
from tests.gateway.fixtures.webhook_route_authority import mount_authority


@asynccontextmanager
async def settled_before_finalizer(*, compress=False):
    config = PlatformConfig(enabled=True, extra={
        "secret": "owned-secret",
        "routes": {"fixture": {"prompt": "{text}", "deliver": "log"}},
    })
    adapter = WebhookAdapter(config)
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        body = json.dumps({"text": "one turn only"}).encode()
        headers = {
            "X-GitHub-Delivery": "settled-before-finalizer",
            "X-Hub-Signature-256": "sha256=" + hmac.new(
                b"owned-secret", body, hashlib.sha256
            ).hexdigest(),
        }
        response = await client.post("/webhooks/fixture", data=body, headers=headers)
        assert response.status == 202, await response.text()
        runner = adapter._message_handler.__self__
        authority = runner.session_authority
        row = dict(authority.db._read_one("SELECT * FROM session_admissions"))
        claimed = claim_session_input(
            authority.db, epoch=authority.epoch, session_id=row["target_session_id"]
        )
        if compress:
            logical = row["target_session_id"]
            entry = runner.session_store.lookup_by_session_id(logical)
            physical = logical + "-compressed"
            authority.db.publish_compression_child(
                parent_session_id=logical,
                child_session_id=physical,
                source="webhook",
                messages=[{"role": "user", "content": "retained summary"}],
                require_compression_lease=False,
            )
            assert runner.session_store.advance_compression_session(
                entry.session_key, logical, physical
            ) is not None
            # The logical webhook still owns terminal settlement/finalization
            # after its physical transcript rotates.
            authority.db._write_sql(
                "UPDATE sessions SET ended_at=NULL WHERE id=?", (logical,)
            )
            row["physical_session_id"] = physical
        settle_session_input(
            authority.db,
            epoch=authority.epoch,
            admission_id=row["admission_id"],
            generation=claimed["generation"],
            outcome="completed",
        )
        # Crash after the durable settlement but before the process-local finalizer.
        for task in list(adapter._background_tasks):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        assert authority.db.get_session(row["target_session_id"])["ended_at"] is None
        yield client, body, headers, runner, authority, row


@pytest.mark.asyncio
async def test_restart_finalizes_settled_webhook_without_provider_retry(monkeypatch):
    async with settled_before_finalizer() as (_client, _body, _headers, runner, authority, row):
        from gateway import run_runtime
        from gateway.session_authority import initialize_session_authority

        async def skip_unrelated_hosted_service(_runner):
            return None

        monkeypatch.setattr(
            "gateway.session_hosted_service.ensure_hosted_service",
            skip_unrelated_hosted_service,
        )
        restarted = await initialize_session_authority(
            runner,
            profile_id=authority.profile_id,
            instance_id="replacement-owner",
            db=authority.db,
        )
        scheduled = []
        restarted._schedule = scheduled.append

        await run_runtime.recover_gateway_native_sessions(runner)

        saved = authority.db.get_session(row["target_session_id"])
        assert saved["end_reason"] == "webhook_complete"
        assert scheduled == []
        assert len(authority.db._read_all("SELECT * FROM session_admissions")) == 1


@pytest.mark.asyncio
async def test_restart_reauthorizes_compressed_route_but_finalizes_logical_webhook():
    async with settled_before_finalizer(compress=True) as (
        _client, _body, _headers, runner, authority, row
    ):
        from gateway.platforms.webhook_ingress import recover_webhook_finalizations
        from gateway.session_authority import initialize_session_authority

        logical = row["target_session_id"]
        physical = row["physical_session_id"]
        entry = runner.session_store.lookup_by_session_id(physical)
        unrelated = physical + "-branch"
        authority.db.create_session(
            unrelated,
            source="webhook",
            parent_session_id=physical,
            model_config={"_branched_from": physical},
        )
        unrelated_before = authority.db.get_session(unrelated)

        restarted = await initialize_session_authority(
            runner,
            profile_id=authority.profile_id,
            instance_id="replacement-owner",
            db=authority.db,
        )
        scheduled = []
        restarted._schedule = scheduled.append
        sent = []

        async def forbid_send(*args, **kwargs):
            sent.append((args, kwargs))

        adapter = runner._adapter_for_source(entry.origin)
        adapter.send = forbid_send

        results = await recover_webhook_finalizations(restarted)

        assert results == {logical: "finalized"}
        assert authority.db.get_session(logical)["end_reason"] == "webhook_complete"
        assert authority.db.get_session(physical)["ended_at"] is None
        assert authority.db.get_session(unrelated) == unrelated_before
        assert scheduled == []
        assert sent == []
        admissions = authority.db._read_all("SELECT * FROM session_admissions")
        assert len(admissions) == 1
        assert admissions[0]["target_session_id"] == logical
        assert admissions[0]["status"] == "terminal"


@pytest.mark.asyncio
async def test_duplicate_request_finalizes_settled_webhook_without_reexecution():
    async with settled_before_finalizer() as (client, body, headers, _runner, authority, row):
        scheduled = []
        authority._schedule = scheduled.append
        response = await client.post("/webhooks/fixture", data=body, headers=headers)

        assert response.status == 200, await response.text()
        assert (await response.json())["status"] == "duplicate"
        assert authority.db.get_session(row["target_session_id"])["end_reason"] == "webhook_complete"
        assert scheduled == []
        assert len(authority.db._read_all("SELECT * FROM session_admissions")) == 1
