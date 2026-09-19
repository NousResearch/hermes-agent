"""Durable busy-event convergence for webhook ``cron_job`` routes.

Production (Herdr, 2026-09-17): a terminal event received HTTP 202 while the
target cron job was already running; ``_run_claimed_job`` then returned
already-running/already-fired and the accepted event context was gone.
``fire_claim`` and the scheduler in-memory running guard are separate
authorities — winning a claim (or being accepted) must not drop the wake.
"""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


def _make_adapter(routes, **extra_kw) -> WebhookAdapter:
    extra = {"host": "127.0.0.1", "port": 0, "routes": routes}
    extra.update(extra_kw)
    return WebhookAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


async def _drain_background_tasks(adapter: WebhookAdapter) -> None:
    tasks = list(adapter._background_tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _event_contexts_on_job(job):
    """Durable event contexts on a job record (pending batch or live claim)."""
    contexts = []
    claim = job.get("fire_claim") if isinstance(job.get("fire_claim"), dict) else {}
    containers = [job.get("pending_event_batch"), claim.get("event_batch"), claim]
    for container in containers:
        if isinstance(container, dict):
            events = container.get("events")
        elif isinstance(container, list):
            events = container
        else:
            continue
        if not isinstance(events, list):
            continue
        for event in events:
            if isinstance(event, dict):
                ctx = event.get("context")
                if ctx:
                    contexts.append(str(ctx))
            elif isinstance(event, str) and event:
                contexts.append(event)
    return contexts


class TestAcceptedBusyWakeRetainsEvent:
    @pytest.mark.asyncio
    async def test_accepted_busy_wake_does_not_drop_event_context(self):
        """HTTP 202 while the in-memory running guard holds the job must still
        persist the event. Admission uses the store; the running set is a
        separate authority and may refuse the claimed snapshot after 202.
        """
        from cron.jobs import create_job, get_job
        from cron.scheduler import release_running_job, try_register_running_job

        job = create_job(
            prompt="controller sweep",
            schedule="every 5m",
            name="herdr-controller",
        )
        delivery_id = "delivery-busy-wake-1"
        event_marker = "terminal event from herdr busy wake"
        adapter = _make_adapter(
            {
                "herdr": {
                    "secret": _INSECURE_NO_AUTH,
                    "cron_job": job["id"],
                    "prompt": event_marker + " {n}",
                }
            }
        )
        assert try_register_running_job(job["id"]) is True
        try:
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                resp = await cli.post(
                    "/webhooks/herdr",
                    data=b'{"n": 9}',
                    headers={
                        "Content-Type": "application/json",
                        "X-GitHub-Delivery": delivery_id,
                        "X-GitHub-Event": "terminal",
                    },
                )
                body = await resp.json()
                public = json.dumps(body)
                assert resp.status == 202
                assert body.get("status") == "accepted"
                assert "controller sweep" not in public
                await _drain_background_tasks(adapter)

                retry = await cli.post(
                    "/webhooks/herdr",
                    data=b'{"n": 9}',
                    headers={
                        "Content-Type": "application/json",
                        "X-GitHub-Delivery": delivery_id,
                    },
                )
                retry_body = await retry.json()
                assert retry.status == 200
                assert retry_body.get("status") == "duplicate"
        finally:
            release_running_job(job["id"])

        refreshed = get_job(job["id"])
        assert refreshed is not None
        contexts = _event_contexts_on_job(refreshed)
        assert any(event_marker in ctx and "9" in ctx for ctx in contexts), (
            "accepted busy wake dropped event context; durable batch was "
            f"{contexts!r} on job {refreshed}"
        )


class TestBusyQueueHttpContract:
    @pytest.mark.asyncio
    async def test_overflow_is_retryable_and_does_not_consume_delivery_id(self):
        from cron.jobs import EVENT_BATCH_MAX_BYTES, create_job

        job = create_job(prompt="sweep", schedule="every 5m", name="overflow-target")
        delivery_id = "delivery-overflow-1"
        adapter = _make_adapter(
            {
                "herdr": {
                    "secret": _INSECURE_NO_AUTH,
                    "cron_job": job["id"],
                    "prompt": "x" * (EVENT_BATCH_MAX_BYTES + 8),
                }
            }
        )
        headers = {
            "Content-Type": "application/json",
            "X-GitHub-Delivery": delivery_id,
        }
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.post("/webhooks/herdr", data=b'{"n": 1}', headers=headers)
            body = await resp.json()
            assert resp.status == 503
            assert body.get("error") == "Cron job is not available"
            retry = await cli.post("/webhooks/herdr", data=b'{"n": 1}', headers=headers)
            retry_body = await retry.json()
            assert retry.status == 503
            assert retry_body.get("status") != "duplicate"
        await _drain_background_tasks(adapter)


class TestImmediateClaimMergesPendingContexts:
    @pytest.mark.asyncio
    async def test_immediate_claim_merging_pending_delivers_every_accepted_context(
        self, monkeypatch
    ):
        """Queued E1/E2 plus a later immediate E3 must run every accepted context once.

        Review F1: admit embeds [E1,E2,E3] on the claim, but the webhook worker
        passed only ctx(E3) as extra_prompt, so E1/E2 were receipted and dropped.
        """
        from cron.jobs import claim_job_for_fire, create_job, get_job, mark_job_run
        import cron.scheduler as sched

        job = create_job(prompt="controller", schedule="every 5m", name="merge-target")
        assert claim_job_for_fire(job["id"], manual=True) is True
        owner = get_job(job["id"])["fire_claim"]["by"]
        adapter = _make_adapter(
            {
                "herdr": {
                    "secret": _INSECURE_NO_AUTH,
                    "cron_job": job["id"],
                    "prompt": "{marker}",
                }
            }
        )
        prompts = []

        def _fake_run_job(_run_job, *, extra_prompt=None, **_kw):
            prompts.append(extra_prompt)
            return True, "out", "final", None

        monkeypatch.setattr(sched, "run_job", _fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_k: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_k: None)

        async with TestClient(TestServer(_create_app(adapter))) as cli:
            for marker, delivery in (
                ("ctx-ONE", "delivery-e1"),
                ("ctx-TWO", "delivery-e2"),
            ):
                resp = await cli.post(
                    "/webhooks/herdr",
                    data=json.dumps({"marker": marker}).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "X-GitHub-Delivery": delivery,
                        "X-GitHub-Event": "terminal",
                    },
                )
                assert resp.status == 202
            assert mark_job_run(job["id"], True, expected_fire_owner=owner) is True
            pending = (get_job(job["id"]).get("pending_event_batch") or {}).get("events") or []
            assert [event["delivery_id"] for event in pending] == ["delivery-e1", "delivery-e2"]

            third = await cli.post(
                "/webhooks/herdr",
                data=json.dumps({"marker": "ctx-THREE"}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Delivery": "delivery-e3",
                    "X-GitHub-Event": "terminal",
                },
            )
            assert third.status == 202
            await _drain_background_tasks(adapter)

        joined = "\n".join(str(p) for p in prompts if p)
        assert "ctx-ONE" in joined, f"lost E1 in prompts={prompts!r}"
        assert "ctx-TWO" in joined, f"lost E2 in prompts={prompts!r}"
        assert "ctx-THREE" in joined, f"lost E3 in prompts={prompts!r}"
        assert joined.count("ctx-THREE") == 1
        refreshed = get_job(job["id"])
        assert not (refreshed.get("pending_event_batch") or {}).get("events")
