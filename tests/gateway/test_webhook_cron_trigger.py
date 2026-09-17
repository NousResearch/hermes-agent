"""Tests for the webhook adapter's ``cron_job`` route mode.

``cron_job`` routes turn an existing cron job into an event-triggered task:
an inbound webhook event fires the referenced job through the same
claimed-run body a manual ``cronjob(action='run')`` uses, instead of
starting a fresh webhook agent session.

Covers:
- Admission claims the job synchronously; a worker runs the claimed
  snapshot with the rendered prompt as transient per-run context
- The normal webhook agent session is NOT started (``handle_message``
  never called)
- HTTP 202 means durable store admission (claim or queued batch), not
  "a background task was created"
- Paused/unknown/unrunnable targets are retryable non-2xx and do not
  consume the delivery ID
- A busy in-flight job durably queues the wake and still returns 202
- A claim-exception / store-failure is the same retryable refusal:
  503 + Retry-After, ID unconsumed, no dispatch;
  ``execute_job_for_event`` returns the error dict
- Startup validation rejects routes that set both ``cron_job`` and
  ``deliver_only``
- ``execute_job_for_event`` resolves refs and fails cleanly on unknowns
"""

import asyncio
import json
import threading
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(routes, **extra_kw) -> WebhookAdapter:
    extra = {"host": "127.0.0.1", "port": 0, "routes": routes}
    extra.update(extra_kw)
    config = PlatformConfig(enabled=True, extra=extra)
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


async def _drain_background_tasks(adapter: WebhookAdapter) -> None:
    tasks = list(adapter._background_tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ===================================================================
# Core behaviour: event fires the cron job, not a webhook session
# ===================================================================

class TestCronJobTrigger:
    @pytest.mark.asyncio
    async def test_post_fires_job_with_event_context(self):
        from cron.jobs import create_job

        job = create_job(
            prompt="sweep reviews", schedule="every 5m", name="review-sweeper"
        )
        routes = {
            "pr-feedback": {
                "secret": _INSECURE_NO_AUTH,
                "cron_job": "review-sweeper",
                "prompt": "PR #{number} received feedback: {review.body}",
            }
        }
        adapter = _make_adapter(routes)

        handle_message_calls = []

        async def _capture(event):
            handle_message_calls.append(event)

        adapter.handle_message = _capture

        fired = []

        def _fake_run(claimed_job, extra_prompt=None):
            fired.append((claimed_job, extra_prompt))
            return {"claimed": True, "success": True, "error": None}

        app = _create_app(adapter)
        body = json.dumps(
            {"number": 7, "review": {"body": "needs tests"}}
        ).encode()

        with patch("tools.cronjob_tools._run_claimed_job", side_effect=_fake_run):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/webhooks/pr-feedback",
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-GitHub-Delivery": "delivery-cron-1",
                        "X-GitHub-Event": "pull_request_review",
                    },
                )
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "accepted"
                assert data["cron_job"] == "review-sweeper"
                await _drain_background_tasks(adapter)

        assert len(fired) == 1
        claimed_job, extra_prompt = fired[0]
        assert claimed_job["id"] == job["id"]
        assert claimed_job.get("fire_claim")
        assert "PR #7 received feedback: needs tests" in extra_prompt
        assert "pull_request_review" in extra_prompt  # event provenance
        assert handle_message_calls == []

    @pytest.mark.asyncio
    async def test_unknown_job_is_retryable_and_does_not_consume_delivery_id(self):
        routes = {
            "flaky": {"secret": _INSECURE_NO_AUTH, "cron_job": "gone-job"}
        }
        adapter = _make_adapter(routes)
        app = _create_app(adapter)
        headers = {
            "Content-Type": "application/json",
            "X-GitHub-Delivery": "delivery-cron-2",
        }

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/webhooks/flaky", data=b"{}", headers=headers)
            body = await resp.json()
            assert resp.status == 503
            assert body.get("error") == "Cron job is not available"
            assert "gone-job" not in json.dumps(body)

            retry = await cli.post("/webhooks/flaky", data=b"{}", headers=headers)
            retry_body = await retry.json()
            assert retry.status == 503
            assert retry_body.get("status") != "duplicate"

        await _drain_background_tasks(adapter)

    @pytest.mark.asyncio
    async def test_routed_profile_scope_reaches_the_job_run(self, tmp_path, monkeypatch):
        """/p/<profile>/ routes must fire the job from THAT profile's cron store, not the gateway's
        default home (and route-level skills stay out of the per-run context — the job's own apply)."""
        from cron.jobs import create_job, use_cron_store

        home = tmp_path / ".hermes"
        sec = home / "profiles" / "sec"
        sec.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr("hermes_cli.profiles._get_default_hermes_home", lambda: home)
        monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: home / "profiles")
        with use_cron_store(sec):
            job = create_job(prompt="sweep", schedule="every 5m", name="sweeper")
        adapter = _make_adapter({"ev": {"secret": _INSECURE_NO_AUTH, "cron_job": job["id"], "profile": "sec",
                                        "skills": ["some-skill"], "prompt": "hello {n}"}})
        monkeypatch.setattr(adapter, "_resolve_request_profile", lambda request: "sec")
        monkeypatch.setattr(adapter, "_apply_skills", lambda prompt, skills: pytest.fail("skills applied"))
        seen = []

        def _fake_run(claimed_job, extra_prompt=None):
            from hermes_constants import get_hermes_home
            seen.append((get_hermes_home(), extra_prompt, claimed_job.get("id")))
            return {"claimed": True, "success": True, "error": None}

        with patch("tools.cronjob_tools._run_claimed_job", side_effect=_fake_run):
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                resp = await cli.post("/webhooks/ev", data=b'{"n": 1}',
                                      headers={"Content-Type": "application/json", "X-GitHub-Delivery": "d-3"})
                assert resp.status == 202
                await _drain_background_tasks(adapter)
        assert seen and seen[0][0] == sec.resolve()
        assert "hello 1" in seen[0][1]
        assert seen[0][2] == job["id"]


    @pytest.mark.asyncio
    async def test_paused_job_is_retryable_and_does_not_consume_delivery_id(self):
        """A paused cron_job target must not 202 or consume the delivery ID.

        Production 2026-09-17: a Herdr relay treated 202 as durable admission
        for ``cron_job: 5a6924edf71e`` while that job was paused, ACK'd the
        producer, and never retried. No controller ran.
        """
        from cron.jobs import create_job, get_job

        secret_prompt = "SECRET_CONTROLLER_PROMPT_do_not_leak"
        paused_reason = "operator paused herdr controller"
        job = create_job(
            prompt=secret_prompt,
            schedule="every 5m",
            name="herdr-controller",
            paused=True,
            paused_reason=paused_reason,
        )
        delivery_id = "delivery-paused-5a6924edf71e"
        adapter = _make_adapter(
            {
                "herdr": {
                    "secret": _INSECURE_NO_AUTH,
                    "cron_job": job["id"],
                    "prompt": "event {n}",
                }
            }
        )
        handle_message_calls = []

        async def _capture(event):
            handle_message_calls.append(event)

        adapter.handle_message = _capture
        headers = {
            "Content-Type": "application/json",
            "X-GitHub-Delivery": delivery_id,
        }

        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.post("/webhooks/herdr", data=b'{"n": 1}', headers=headers)
            body = await resp.json()
            public = json.dumps(body)
            assert resp.status == 503
            assert body.get("status") != "accepted"
            assert "SECRET_CONTROLLER_PROMPT" not in public
            assert paused_reason not in public
            assert body.get("error") == "Cron job is not available"

            retry = await cli.post("/webhooks/herdr", data=b'{"n": 1}', headers=headers)
            retry_body = await retry.json()
            assert retry.status == 503
            assert retry_body.get("status") != "duplicate"

        await _drain_background_tasks(adapter)
        assert handle_message_calls == []
        refreshed = get_job(job["id"])
        assert refreshed is not None
        assert not refreshed.get("fire_claim")

    @pytest.mark.asyncio
    async def test_runnable_job_claims_before_202_and_worker_does_not_reclaim(self):
        """202 is returned only after durable store admission; the worker
        executes that snapshot and must not claim again. A second distinct
        event while the claim is held is queued, not dropped."""
        from cron.jobs import admit_job_event, create_job, get_job
        from tools import cronjob_tools

        job = create_job(prompt="sweep reviews", schedule="every 5m", name="review-sweeper")
        original_admit = admit_job_event
        claim_started = threading.Event()
        release_claim = threading.Event()
        admit_calls = []
        ran = []

        def _blocking_admit(job_ref, **kwargs):
            admit_calls.append(job_ref)
            claim_started.set()
            assert release_claim.wait(timeout=2)
            return original_admit(job_ref, **kwargs)

        def _fake_run(claimed_job, extra_prompt=None):
            ran.append((claimed_job, extra_prompt))
            return {"claimed": True, "success": True, "error": None}

        adapter = _make_adapter(
            {
                "pr-feedback": {
                    "secret": _INSECURE_NO_AUTH,
                    "cron_job": job["id"],
                    "prompt": "PR #{number} received feedback",
                }
            }
        )
        handle_message_calls = []

        async def _capture(event):
            handle_message_calls.append(event)

        adapter.handle_message = _capture

        with patch.object(cronjob_tools, "admit_job_event", side_effect=_blocking_admit), patch.object(
            cronjob_tools, "_run_claimed_job", side_effect=_fake_run
        ):
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                request_task = asyncio.create_task(
                    cli.post(
                        "/webhooks/pr-feedback",
                        data=b'{"number": 7}',
                        headers={
                            "Content-Type": "application/json",
                            "X-GitHub-Delivery": "delivery-runnable-1",
                            "X-GitHub-Event": "pull_request_review",
                        },
                    )
                )
                assert await asyncio.to_thread(claim_started.wait, 2)
                await asyncio.sleep(0)
                assert not request_task.done()
                release_claim.set()
                resp = await request_task
                data = await resp.json()
                assert resp.status == 202
                assert data["status"] == "accepted"
                assert data["cron_job"] == job["id"]
                busy = await cli.post(
                    "/webhooks/pr-feedback",
                    data=b'{"number": 8}',
                    headers={
                        "Content-Type": "application/json",
                        "X-GitHub-Delivery": "delivery-runnable-2",
                    },
                )
                busy_body = await busy.json()
                assert busy.status == 202
                assert busy_body.get("status") == "accepted"
                await _drain_background_tasks(adapter)

        assert admit_calls == [job["id"], job["id"]]
        assert len(ran) == 1
        claimed_job, extra_prompt = ran[0]
        assert claimed_job["id"] == job["id"]
        assert claimed_job.get("fire_claim")
        assert "PR #7 received feedback" in extra_prompt
        assert "pull_request_review" in extra_prompt
        assert handle_message_calls == []
        pending = (get_job(job["id"]) or {}).get("pending_event_batch") or {}
        pending_ctx = " ".join(
            str(event.get("context") or "")
            for event in (pending.get("events") or [])
            if isinstance(event, dict)
        )
        assert "PR #8 received feedback" in pending_ctx

    @pytest.mark.asyncio
    async def test_claim_exception_is_retryable_and_does_not_consume_delivery_id(self):
        """``_claim_for_manual_run`` may return claimed:true with no snapshot.

        That shape must not KeyError into a bare 500: the producer gets a
        retryable 503, the delivery ID stays unconsumed, and nothing runs.
        """
        from cron.jobs import create_job, get_job
        from tools import cronjob_tools

        secret_prompt = "SECRET_CLAIM_EXCEPTION_PROMPT_do_not_leak"
        job = create_job(
            prompt=secret_prompt, schedule="every 5m", name="claim-exception-target"
        )
        delivery_id = "delivery-claim-exception-1"
        adapter = _make_adapter(
            {
                "herdr": {
                    "secret": _INSECURE_NO_AUTH,
                    "cron_job": job["id"],
                    "prompt": "event {n}",
                }
            }
        )
        handle_message_calls = []

        async def _capture(event):
            handle_message_calls.append(event)

        adapter.handle_message = _capture
        ran = []

        def _fake_run(claimed_job, extra_prompt=None):
            ran.append((claimed_job, extra_prompt))
            return {"claimed": True, "success": True, "error": None}

        headers = {
            "Content-Type": "application/json",
            "X-GitHub-Delivery": delivery_id,
        }

        with patch.object(
            cronjob_tools, "admit_job_event", side_effect=OSError("disk")
        ), patch.object(cronjob_tools, "_run_claimed_job", side_effect=_fake_run):
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                resp = await cli.post(
                    "/webhooks/herdr", data=b'{"n": 1}', headers=headers
                )
                body = await resp.json()
                public = json.dumps(body)
                assert resp.status == 503
                assert resp.headers.get("Retry-After") == "60"
                assert body.get("status") != "accepted"
                assert body.get("error") == "Cron job is not available"
                assert "disk" not in public
                assert "SECRET_CLAIM_EXCEPTION_PROMPT" not in public

                retry = await cli.post(
                    "/webhooks/herdr", data=b'{"n": 1}', headers=headers
                )
                retry_body = await retry.json()
                assert retry.status == 503
                assert retry.headers.get("Retry-After") == "60"
                assert retry_body.get("status") != "duplicate"

        await _drain_background_tasks(adapter)
        assert handle_message_calls == []
        assert ran == []
        refreshed = get_job(job["id"])
        assert refreshed is not None
        assert not refreshed.get("fire_claim")


# ===================================================================
# Startup validation
# ===================================================================

class TestCronJobRouteValidation:
    @pytest.mark.asyncio
    async def test_cron_job_plus_deliver_only_rejected_at_connect(self):
        routes = {
            "bad": {
                "secret": "s3cret",
                "cron_job": "some-job",
                "deliver_only": True,
                "deliver": "telegram",
            }
        }
        adapter = _make_adapter(routes)
        with pytest.raises(ValueError, match="mutually exclusive"):
            await adapter.connect()


# ===================================================================
# execute_job_for_event unit behaviour
# ===================================================================

class TestExecuteJobForEvent:
    def test_unknown_job_returns_error(self):
        from tools import cronjob_tools

        result = cronjob_tools.execute_job_for_event("nope")
        assert result["claimed"] is False
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_ambiguous_ref_returns_error(self):
        from cron.jobs import create_job
        from tools import cronjob_tools

        create_job(prompt="a", schedule="every 5m", name="x")
        create_job(prompt="b", schedule="every 5m", name="x")
        result = cronjob_tools.execute_job_for_event("x")
        assert result["claimed"] is False
        assert result["success"] is False
        assert "ambiguous" in result["error"].lower()

    def test_resolved_job_fires_with_extra_prompt(self):
        from cron.jobs import create_job
        from tools import cronjob_tools

        job = create_job(prompt="sweep", schedule="every 5m", name="sweeper")
        with patch.object(
            cronjob_tools,
            "_run_claimed_job",
            return_value={"claimed": True, "success": True, "error": None},
        ) as mock_exec:
            result = cronjob_tools.execute_job_for_event(
                "sweeper", extra_prompt="event context"
            )
        assert result["success"] is True
        assert mock_exec.call_count == 1
        claimed, kwargs = mock_exec.call_args[0][0], mock_exec.call_args.kwargs
        if mock_exec.call_args.args[1:]:
            extra = mock_exec.call_args.args[1]
        else:
            extra = kwargs.get("extra_prompt")
        assert claimed["id"] == job["id"]
        assert claimed.get("fire_claim")
        assert extra == "event context"

    def test_claim_exception_returns_error_dict_without_raising(self):
        """Store-failure results must not KeyError into a run.

        Public contract is the error dict, no run.
        """
        from cron.jobs import create_job
        from tools import cronjob_tools

        create_job(prompt="sweep", schedule="every 5m", name="sweeper")
        with patch.object(
            cronjob_tools, "admit_job_event", side_effect=OSError("disk")
        ), patch.object(
            cronjob_tools, "_run_claimed_job"
        ) as mock_exec:
            result = cronjob_tools.execute_job_for_event("sweeper")
        assert result["success"] is False
        assert result.get("error") == "disk"
        assert "job" not in result
        mock_exec.assert_not_called()
