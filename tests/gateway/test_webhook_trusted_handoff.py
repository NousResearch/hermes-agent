"""Behavior contracts for authenticated webhook profile handoffs."""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.platforms.base import MessageEvent, ProcessingOutcome, SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH
from gateway.session import (
    SessionContext,
    SessionEntry,
    SessionSource,
    build_session_context_prompt,
)
from tests.gateway.restart_test_helpers import make_restart_runner


def _signature(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _v2_headers(body: bytes, *, request_id: str, timestamp: str) -> dict[str, str]:
    signed = timestamp.encode() + b"." + body
    return {
        "Content-Type": "application/json",
        "X-Webhook-Timestamp": timestamp,
        "X-Webhook-Signature-V2": hmac.new(
            b"relay-secret", signed, hashlib.sha256
        ).hexdigest(),
        "X-Request-ID": request_id,
    }


def _svix_headers(body: bytes, *, message_id: str, timestamp: str) -> dict[str, str]:
    signed = message_id.encode() + b"." + timestamp.encode() + b"." + body
    signature = base64.b64encode(
        hmac.new(b"relay-secret", signed, hashlib.sha256).digest()
    ).decode()
    return {
        "Content-Type": "application/json",
        "svix-id": message_id,
        "svix-timestamp": timestamp,
        "svix-signature": f"v1,{signature}",
    }


def _adapter(route: dict, *, multiplex: bool = True) -> WebhookAdapter:
    adapter = WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 0, "routes": {"relay": route}},
        )
    )

    class Runner:
        config = GatewayConfig(
            multiplex_profiles=multiplex,
            multiplex_profile_allowlist=["dispatcher", "market-analysis", "server-development"],
        )

        @staticmethod
        def _profile_name_for_source(source):
            return None

    adapter.gateway_runner = Runner()
    return adapter


def _app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application(client_max_size=adapter._max_body_bytes)
    app.router.add_post("/p/{profile}/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _trusted_route(**overrides) -> dict:
    route = {
        "secret": "relay-secret",
        "profile": "dispatcher",
        "prompt": "Task: {task}",
        "allowed_target_profiles": ["market-analysis", "server-development"],
        "allowed_target_toolsets": {
            "market-analysis": ["web", "terminal"],
            "server-development": ["web", "terminal", "file"],
        },
        "max_handoff_depth": 1,
        "max_handoff_concurrency": 2,
        "deliver": "discord",
        "deliver_extra": {"chat_id": "market-room"},
    }
    route.update(overrides)
    return route


@pytest.fixture
def served_profiles(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex, profile_allowlist=None: [
            (name, f"/profiles/{name}")
            for name in ("default", "dispatcher", "market-analysis", "server-development")
        ],
    )


@pytest.fixture(autouse=True)
def isolated_handoff_receipts(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))


@pytest.mark.asyncio
async def test_authenticated_selector_hands_off_to_allowlisted_profile(served_profiles):
    adapter = _adapter(_trusted_route())
    events: list[MessageEvent] = []

    async def capture(event: MessageEvent):
        events.append(event)

    adapter.handle_message = capture
    payload = {
        "_hermes": {
            "target_profile": "market-analysis",
            "handoff_depth": 1,
            "delivery_id": "handoff-allowed",
        },
        "task": "Summarize the market with the local CLI",
    }
    body = json.dumps(payload).encode()

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _signature(body, "relay-secret"),
                "X-GitHub-Delivery": "handoff-allowed",
            },
        )
        assert response.status == 202
        accepted = await response.json()
        assert accepted["target_profile"] == "market-analysis"

    await asyncio.sleep(0.05)
    assert len(events) == 1
    source = events[0].source
    assert source.platform.value == "webhook"
    assert source.profile == "market-analysis"
    assert source.transport_profile == "dispatcher"
    assert source.trusted_handoff_depth == 1
    assert source.transport_deliver == "discord"
    assert source.transport_deliver_extra == {"chat_id": "market-room"}
    assert "_hermes" not in events[0].raw_message
    assert sorted(adapter.toolsets_for_source(source)) == ["terminal", "web"]
    assert source.provenance == {
        "ingress_platform": "webhook",
        "ingress_route": "relay",
        "source_profile": "dispatcher",
        "target_profile": "market-analysis",
        "effective_toolsets": ["terminal", "web"],
        "delivery_platform": "discord",
        "delivery_chat_id": "market-room",
        "handoff_depth": 1,
    }
    assert SessionSource.from_dict(source.to_dict()).provenance == source.provenance


@pytest.mark.asyncio
async def test_generic_v2_handoff_replay_ignores_unsigned_request_id_after_restart(
    served_profiles,
):
    payload = {
        "_hermes": {
            "target_profile": "market-analysis",
            "handoff_depth": 1,
            "delivery_id": "signed-v2-delivery",
        },
        "task": "run once",
    }
    body = json.dumps(payload).encode()
    timestamp = str(int(time.time()))
    events: list[MessageEvent] = []

    async def capture(event: MessageEvent):
        events.append(event)

    first_adapter = _adapter(_trusted_route())
    first_adapter.handle_message = capture
    async with TestClient(TestServer(_app(first_adapter))) as client:
        first = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers=_v2_headers(body, request_id="unsigned-A", timestamp=timestamp),
        )
        assert first.status == 202
        assert (await first.json())["delivery_id"] == "signed-v2-delivery"

    await asyncio.sleep(0.05)
    restarted_adapter = _adapter(_trusted_route())
    restarted_adapter.handle_message = capture
    async with TestClient(TestServer(_app(restarted_adapter))) as client:
        replay = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers=_v2_headers(body, request_id="unsigned-B", timestamp=timestamp),
        )
        assert replay.status == 200
        assert await replay.json() == {
            "status": "duplicate",
            "delivery_id": "signed-v2-delivery",
        }

    await asyncio.sleep(0.05)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_svix_signed_handoff_id_accepts_once_and_deduplicates_retry(
    served_profiles,
):
    adapter = _adapter(_trusted_route())
    events: list[MessageEvent] = []

    async def capture(event: MessageEvent):
        events.append(event)

    adapter.handle_message = capture
    payload = {
        "_hermes": {
            "target_profile": "market-analysis",
            "handoff_depth": 1,
            "delivery_id": "signed-svix-delivery",
        },
        "task": "run once",
    }
    body = json.dumps(payload).encode()
    headers = _svix_headers(
        body,
        message_id="msg_provider_signed",
        timestamp=str(int(time.time())),
    )

    async with TestClient(TestServer(_app(adapter))) as client:
        first = await client.post(
            "/p/dispatcher/webhooks/relay", data=body, headers=headers
        )
        retry = await client.post(
            "/p/dispatcher/webhooks/relay", data=body, headers=headers
        )
        assert first.status == 202
        assert retry.status == 200
        assert (await retry.json())["status"] == "duplicate"

    await asyncio.sleep(0.05)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_handoff_replay_store_failure_blocks_dispatch(
    served_profiles, monkeypatch
):
    adapter = _adapter(_trusted_route())
    adapter.handle_message = pytest.fail
    payload = {
        "_hermes": {
            "target_profile": "market-analysis",
            "handoff_depth": 1,
            "delivery_id": "store-failure",
        },
        "task": "must not run",
    }
    body = json.dumps(payload).encode()

    def fail_claim(**_kwargs):
        raise OSError("state database unavailable")

    monkeypatch.setattr(
        "gateway.webhook_replay.claim_handoff_delivery", fail_claim
    )
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _signature(body, "relay-secret"),
            },
        )
        assert response.status == 503


@pytest.mark.asyncio
async def test_atomic_handoff_claim_has_exactly_one_winner():
    from gateway.webhook_replay import claim_handoff_delivery

    claims = await asyncio.gather(
        *(
            asyncio.to_thread(
                claim_handoff_delivery,
                route_name="relay",
                source_profile="dispatcher",
                delivery_id="concurrent-signed-id",
            )
            for _ in range(8)
        )
    )
    assert claims.count(True) == 1
    assert claims.count(False) == 7


@pytest.mark.asyncio
async def test_duplicate_handoff_skips_route_script_after_adapter_restart(
    served_profiles,
):
    payload = {
        "_hermes": {
            "target_profile": "market-analysis",
            "handoff_depth": 1,
            "delivery_id": "script-once",
        },
        "task": "run once",
    }
    body = json.dumps(payload).encode()
    timestamp = str(int(time.time()))
    script_calls: list[dict] = []

    def run_script(_script, script_payload):
        script_calls.append(script_payload)
        return True, script_payload

    first_adapter = _adapter(_trusted_route(script="fake-script"))
    first_adapter._route_processor.run_route_script = run_script
    first_adapter.handle_message = AsyncMock()
    async with TestClient(TestServer(_app(first_adapter))) as client:
        first = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers=_v2_headers(body, request_id="script-A", timestamp=timestamp),
        )
        assert first.status == 202

    restarted_adapter = _adapter(_trusted_route(script="fake-script"))
    restarted_adapter._route_processor.run_route_script = run_script
    restarted_adapter.handle_message = pytest.fail
    async with TestClient(TestServer(_app(restarted_adapter))) as client:
        retry = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers=_v2_headers(body, request_id="script-B", timestamp=timestamp),
        )
        assert retry.status == 200
        assert (await retry.json())["status"] == "duplicate"

    assert len(script_calls) == 1


@pytest.mark.asyncio
async def test_concurrent_handoffs_reserve_capacity_atomically(
    served_profiles, monkeypatch
):
    from gateway import webhook_replay

    real_claim = webhook_replay.claim_handoff_delivery

    def slow_claim(**kwargs):
        time.sleep(0.05)
        return real_claim(**kwargs)

    monkeypatch.setattr(webhook_replay, "claim_handoff_delivery", slow_claim)
    adapter = _adapter(_trusted_route(max_handoff_concurrency=1))
    events: list[MessageEvent] = []
    processing_started = asyncio.Event()
    processing_release = asyncio.Event()

    async def process(event: MessageEvent):
        events.append(event)
        processing_started.set()
        await processing_release.wait()

    adapter.set_message_handler(process)

    def request_body(delivery_id: str) -> bytes:
        return json.dumps(
            {
                "_hermes": {
                    "target_profile": "market-analysis",
                    "handoff_depth": 1,
                    "delivery_id": delivery_id,
                },
                "task": delivery_id,
            }
        ).encode()

    first_body = request_body("concurrent-A")
    second_body = request_body("concurrent-B")
    async with TestClient(TestServer(_app(adapter))) as client:
        first, second = await asyncio.gather(
            client.post(
                "/p/dispatcher/webhooks/relay",
                data=first_body,
                headers={
                    "Content-Type": "application/json",
                    "X-Hub-Signature-256": _signature(first_body, "relay-secret"),
                },
            ),
            client.post(
                "/p/dispatcher/webhooks/relay",
                data=second_body,
                headers={
                    "Content-Type": "application/json",
                    "X-Hub-Signature-256": _signature(second_body, "relay-secret"),
                },
            ),
        )
        assert sorted((first.status, second.status)) == [202, 429]
        await asyncio.wait_for(processing_started.wait(), timeout=1)
        processing_release.set()

    await asyncio.sleep(0.05)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_retry_returns_duplicate_while_route_is_full(served_profiles):
    from gateway.webhook_replay import claim_handoff_delivery

    assert claim_handoff_delivery(
        route_name="relay",
        source_profile="dispatcher",
        delivery_id="already-running",
    )
    adapter = _adapter(_trusted_route(max_handoff_concurrency=1))
    adapter._active_handoffs["relay"].add("webhook:relay:other-run")
    adapter.handle_message = pytest.fail
    payload = {
        "_hermes": {
            "target_profile": "market-analysis",
            "handoff_depth": 1,
            "delivery_id": "already-running",
        },
        "task": "retry",
    }
    body = json.dumps(payload).encode()
    async with TestClient(TestServer(_app(adapter))) as client:
        retry = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _signature(body, "relay-secret"),
            },
        )
        assert retry.status == 200
        assert (await retry.json())["status"] == "duplicate"


@pytest.mark.asyncio
async def test_ordinary_delivery_id_cannot_overwrite_trusted_egress(
    served_profiles,
):
    adapter = _adapter(
        _trusted_route(deliver_extra={"chat_id": "{destination}"})
    )
    events: list[MessageEvent] = []

    async def capture(event: MessageEvent):
        events.append(event)

    adapter.handle_message = capture
    trusted_payload = {
        "_hermes": {
            "target_profile": "market-analysis",
            "handoff_depth": 1,
            "delivery_id": "shared-id",
        },
        "destination": "trusted-room",
        "task": "trusted",
    }
    trusted_body = json.dumps(trusted_payload).encode()
    ordinary_body = json.dumps(
        {"destination": "attacker-room", "task": "ordinary"}
    ).encode()

    async with TestClient(TestServer(_app(adapter))) as client:
        trusted = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=trusted_body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _signature(
                    trusted_body, "relay-secret"
                ),
            },
        )
        ordinary = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=ordinary_body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _signature(
                    ordinary_body, "relay-secret"
                ),
                "X-Request-ID": "shared-id",
            },
        )
        assert trusted.status == 202
        assert ordinary.status == 202

    await asyncio.sleep(0.05)
    trusted_source = next(
        event.source for event in events if event.source.provenance is not None
    )
    assert trusted_source.chat_id == "webhook:relay:trusted-handoff:shared-id"
    assert adapter._delivery_info[trusted_source.chat_id]["deliver_extra"] == {
        "chat_id": "trusted-room"
    }
    assert adapter._delivery_info["webhook:relay:shared-id"]["deliver_extra"] == {
        "chat_id": "attacker-room"
    }


@pytest.mark.asyncio
async def test_post_claim_exception_releases_handoff_capacity(
    served_profiles, monkeypatch
):
    adapter = _adapter(_trusted_route(max_handoff_concurrency=1))

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(adapter, "_render_prompt", fail_render)
    payload = {
        "_hermes": {
            "target_profile": "market-analysis",
            "handoff_depth": 1,
            "delivery_id": "render-failure",
        },
        "task": "must not leak capacity",
    }
    body = json.dumps(payload).encode()
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _signature(body, "relay-secret"),
            },
        )
        assert response.status == 500

    await asyncio.sleep(0)
    assert not adapter._active_handoffs.get("relay")


@pytest.mark.asyncio
@pytest.mark.parametrize("start_processing", [False, None])
async def test_handoff_releases_capacity_when_processing_never_starts(
    served_profiles, monkeypatch, start_processing
):
    adapter = _adapter(_trusted_route(max_handoff_concurrency=1))
    if start_processing is False:
        adapter.set_message_handler(AsyncMock())
        monkeypatch.setattr(
            adapter, "_start_session_processing", lambda *_args, **_kwargs: False
        )
    payload = {
        "_hermes": {
            "target_profile": "market-analysis",
            "handoff_depth": 1,
            "delivery_id": f"no-processing-{start_processing}",
        },
        "task": "must release capacity",
    }
    body = json.dumps(payload).encode()
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _signature(body, "relay-secret"),
            },
        )
        assert response.status == 202

    await asyncio.sleep(0.05)
    assert not adapter._active_handoffs.get("relay")


@pytest.mark.asyncio
async def test_persisted_handoff_resolves_ingress_adapter_and_revalidates_bounds(
    served_profiles,
):
    adapter = _adapter(_trusted_route())
    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:relay:trusted-handoff:restart",
        profile="market-analysis",
        transport_profile="dispatcher",
        trusted_handoff_depth=1,
        transport_deliver="discord",
        transport_deliver_extra={"chat_id": "market-room"},
        transport_delivery_policy_hash=adapter._delivery_policy_hash(
            adapter._routes["relay"]
        ),
        provenance={"source_profile": "descriptive-only-and-not-trusted"},
    )
    now = datetime.now()
    restored = SessionEntry.from_dict(
        SessionEntry(
            session_key="market-analysis:webhook:restart",
            session_id="restart-session",
            created_at=now,
            updated_at=now,
            origin=source,
            resume_pending=True,
        ).to_dict()
    ).origin
    assert restored is not None
    assert getattr(restored, "_transport_adapter_ref", None) is None

    class Runner(GatewayAuthorizationMixin):
        adapters = {}
        _profile_adapters = {"dispatcher": {Platform.WEBHOOK: adapter}}
        config = GatewayConfig(
            multiplex_profiles=True,
            multiplex_profile_allowlist=[
                "dispatcher",
                "market-analysis",
                "server-development",
            ],
        )

        @staticmethod
        def _active_profile_name():
            return "default"

    runner = Runner()
    target = AsyncMock()
    target.send.return_value = SendResult(success=True)
    runner.adapters = {Platform.DISCORD: target}
    adapter.gateway_runner = runner
    adapter._delivery_info[restored.chat_id] = {
        "deliver": "discord",
        "deliver_extra": {"chat_id": "attacker-room"},
    }
    assert adapter._handoff_config_error(adapter._routes["relay"]) is None
    assert "market-analysis" in adapter._served_profile_names()
    assert restored.transport_deliver == "discord"
    assert restored.transport_deliver_extra == {"chat_id": "market-room"}
    assert adapter.validate_restored_source(restored) is True
    assert adapter._delivery_info[restored.chat_id]["deliver_extra"] == {
        "chat_id": "market-room"
    }
    assert runner._adapter_for_source(restored) is adapter
    assert adapter.toolsets_for_source(restored) == ["web", "terminal"]
    assert restored.profile == "market-analysis"
    result = await adapter.send(restored.chat_id, "finished after restart")
    assert result.success is True
    target.send.assert_awaited_once_with(
        "market-room", "finished after restart", metadata=None
    )

    adapter._routes["relay"]["deliver"] = "slack"
    assert runner._adapter_for_source(restored) is None
    adapter._routes["relay"]["deliver"] = "discord"

    target.send.reset_mock()
    adapter._routes["relay"]["deliver_extra"] = {"chat_id": "new-room"}
    assert runner._adapter_for_source(restored) is None
    target.send.assert_not_awaited()
    adapter._routes["relay"]["deliver_extra"] = {"chat_id": "market-room"}

    adapter._routes["relay"]["allowed_target_profiles"] = [
        " market-analysis ",
        "server-development",
    ]
    assert runner._adapter_for_source(restored) is adapter
    adapter._routes["relay"]["allowed_target_profiles"] = [
        "market-analysis",
        "server-development",
    ]

    adapter._routes["relay"].pop("profile")
    assert runner._adapter_for_source(restored) is None
    adapter._routes["relay"]["profile"] = "dispatcher"

    adapter._routes["relay"]["allowed_target_profiles"] = ["server-development"]
    assert runner._adapter_for_source(restored) is None
    assert adapter.toolsets_for_source(restored) is None


@pytest.mark.asyncio
async def test_auto_resumed_handoff_reserves_route_capacity_until_completion(
    served_profiles,
):
    adapter = _adapter(_trusted_route(max_handoff_concurrency=1))
    runner, _ = make_restart_runner(adapter)
    runner.config = GatewayConfig(
        multiplex_profiles=True,
        multiplex_profile_allowlist=[
            "dispatcher",
            "market-analysis",
            "server-development",
        ],
    )
    runner.adapters = {Platform.WEBHOOK: adapter}
    runner._adapter_for_source = lambda _source: adapter
    runner._persist_active_agents = lambda: None
    adapter.gateway_runner = runner

    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:relay:trusted-handoff:restored-A",
        profile="market-analysis",
        transport_profile="dispatcher",
        trusted_handoff_depth=1,
        transport_deliver="discord",
        transport_deliver_extra={"chat_id": "market-room"},
        transport_delivery_policy_hash=adapter._delivery_policy_hash(
            adapter._routes["relay"]
        ),
        provenance=None,
    )
    pending_entry = SessionEntry(
        session_key="market-analysis:webhook:restored-A",
        session_id="restart-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.WEBHOOK,
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}

    resume_gate = asyncio.Event()

    async def hold_resumed_run(event: MessageEvent) -> None:
        if event.internal:
            async def finish_resumed_run() -> None:
                await resume_gate.wait()
                await adapter.on_processing_complete(
                    event, ProcessingOutcome.SUCCESS
                )

            adapter._session_tasks[pending_entry.session_key] = asyncio.create_task(
                finish_resumed_run()
            )
        else:
            await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter.handle_message = hold_resumed_run
    assert runner._schedule_resume_pending_sessions() == 1
    resume_task = next(iter(runner._background_tasks))
    await asyncio.sleep(0)
    assert adapter._active_handoffs["relay"] == {source.chat_id}

    def signed_body(delivery_id: str) -> tuple[bytes, dict[str, str]]:
        body = json.dumps(
            {
                "_hermes": {
                    "target_profile": "market-analysis",
                    "handoff_depth": 1,
                    "delivery_id": delivery_id,
                },
                "task": "do work",
            }
        ).encode()
        return body, {
            "Content-Type": "application/json",
            "X-Hub-Signature-256": _signature(body, "relay-secret"),
        }

    body_b, headers_b = signed_body("restored-concurrent-B")
    async with TestClient(TestServer(_app(adapter))) as client:
        while_resumed = await client.post(
            "/p/dispatcher/webhooks/relay", data=body_b, headers=headers_b
        )
        assert while_resumed.status == 429

        resume_gate.set()
        await resume_task
        assert not adapter._active_handoffs.get("relay")

        after_completion = await client.post(
            "/p/dispatcher/webhooks/relay", data=body_b, headers=headers_b
        )
        assert after_completion.status == 202

    await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selector", "status"),
    [
        ({"target_profile": "finance-admin", "handoff_depth": 1}, 403),
        ({"target_profile": "market-analysis", "handoff_depth": 2}, 403),
        (
            {
                "target_profile": "market-analysis",
                "handoff_depth": 1,
                "toolsets": ["terminal", "file"],
            },
            403,
        ),
    ],
)
async def test_denied_target_depth_or_toolset_expansion_fails_closed(
    served_profiles, selector, status
):
    adapter = _adapter(_trusted_route())
    adapter.handle_message = pytest.fail
    payload = {"_hermes": selector, "task": "do work"}
    body = json.dumps(payload).encode()

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _signature(body, "relay-secret"),
            },
        )
        assert response.status == status
        assert "error" in await response.json()


@pytest.mark.asyncio
async def test_free_form_target_field_cannot_grant_profile(served_profiles):
    adapter = _adapter(_trusted_route())
    adapter.handle_message = pytest.fail
    payload = {"target_profile": "market-analysis", "task": "do work"}
    body = json.dumps(payload).encode()

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _signature(body, "relay-secret"),
            },
        )
        assert response.status == 403


@pytest.mark.asyncio
async def test_unconfigured_static_route_keeps_safe_profile_and_toolset_defaults(served_profiles):
    adapter = _adapter(
        {
            "secret": "relay-secret",
            "profile": "dispatcher",
            "prompt": "Task: {task}",
            "deliver": "discord",
        }
    )
    events: list[MessageEvent] = []

    async def capture(event: MessageEvent):
        events.append(event)

    adapter.handle_message = capture
    body = json.dumps({"task": "ordinary webhook"}).encode()

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _signature(body, "relay-secret"),
            },
        )
        assert response.status == 202

    await asyncio.sleep(0.05)
    assert events[0].source.profile == "dispatcher"
    assert events[0].source.provenance is None
    assert adapter.toolsets_for_source(events[0].source) is None


@pytest.mark.asyncio
async def test_ordinary_webhook_keeps_distinct_unsigned_delivery_ids(served_profiles):
    adapter = _adapter(
        {
            "secret": "relay-secret",
            "profile": "dispatcher",
            "prompt": "Task: {task}",
            "deliver": "discord",
        }
    )
    events: list[MessageEvent] = []

    async def capture(event: MessageEvent):
        events.append(event)

    adapter.handle_message = capture
    body = json.dumps({"task": "ordinary repeated body"}).encode()
    headers = {
        "Content-Type": "application/json",
        "X-Hub-Signature-256": _signature(body, "relay-secret"),
    }
    async with TestClient(TestServer(_app(adapter))) as client:
        first = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers={**headers, "X-Request-ID": "ordinary-A"},
        )
        second = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers={**headers, "X-Request-ID": "ordinary-B"},
        )
        assert first.status == 202
        assert second.status == 202

    await asyncio.sleep(0.05)
    assert [event.message_id for event in events] == ["ordinary-A", "ordinary-B"]


@pytest.mark.asyncio
async def test_handoff_requires_signed_delivery_id(served_profiles):
    adapter = _adapter(_trusted_route())
    adapter.handle_message = pytest.fail
    payload = {
        "_hermes": {"target_profile": "market-analysis", "handoff_depth": 1},
        "task": "do work",
    }
    body = json.dumps(payload).encode()
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _signature(body, "relay-secret"),
            },
        )
        assert response.status == 403
        assert "delivery_id" in (await response.json())["error"]


@pytest.mark.asyncio
async def test_gitlab_token_cannot_authorize_trusted_handoff(served_profiles):
    adapter = _adapter(_trusted_route())
    adapter.handle_message = pytest.fail
    payload = {
        "_hermes": {
            "target_profile": "market-analysis",
            "handoff_depth": 1,
            "delivery_id": "gitlab-token-only",
        },
        "task": "do work",
    }
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/dispatcher/webhooks/relay",
            json=payload,
            headers={"X-Gitlab-Token": "relay-secret"},
        )
        assert response.status == 403
        assert "body-binding signature" in (await response.json())["error"]


@pytest.mark.asyncio
async def test_handoff_requires_authenticated_static_route(served_profiles):
    adapter = _adapter(_trusted_route(secret=_INSECURE_NO_AUTH))
    adapter.handle_message = pytest.fail
    payload = {
        "_hermes": {"target_profile": "market-analysis", "handoff_depth": 1},
        "task": "do work",
    }

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/dispatcher/webhooks/relay",
            json=payload,
        )
        assert response.status == 403


@pytest.mark.asyncio
async def test_handoff_concurrency_limit_rejects_without_starting_another_run(served_profiles):
    adapter = _adapter(_trusted_route(max_handoff_concurrency=1))
    adapter._active_handoffs["relay"].add("webhook:relay:already-running")
    adapter.handle_message = pytest.fail
    payload = {
        "_hermes": {
            "target_profile": "market-analysis",
            "handoff_depth": 1,
            "delivery_id": "concurrency-check",
        },
        "task": "do work",
    }
    body = json.dumps(payload).encode()

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/dispatcher/webhooks/relay",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _signature(body, "relay-secret"),
            },
        )
        assert response.status == 429


@pytest.mark.asyncio
async def test_handoff_completion_releases_concurrency_slot():
    adapter = _adapter(_trusted_route())
    chat_id = "webhook:relay:trusted-handoff:finished"
    adapter._active_handoffs["relay"].add(chat_id)
    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id=chat_id,
        profile="market-analysis",
        transport_profile="dispatcher",
        provenance={
            "ingress_route": "relay",
            "target_profile": "market-analysis",
        },
    )
    event = MessageEvent(text="task", source=source)

    await adapter.on_processing_complete(event, outcome={"status": "complete"})

    assert "relay" not in adapter._active_handoffs


def test_handoff_config_rejects_unbounded_or_invalid_toolsets():
    missing_bounds = _trusted_route()
    missing_bounds.pop("allowed_target_toolsets")
    assert "allowed_target_toolsets" in WebhookAdapter._handoff_config_error(
        missing_bounds
    )

    invalid_toolset = _trusted_route(
        allowed_target_toolsets={
            "market-analysis": ["web", "not-a-toolset"],
            "server-development": ["web", "terminal"],
        }
    )
    assert "unknown or webhook-restricted" in WebhookAdapter._handoff_config_error(
        invalid_toolset
    )


def test_handoff_provenance_is_visible_in_session_diagnostics():
    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:relay:delivery",
        chat_type="webhook",
        profile="market-analysis",
        provenance={
            "ingress_platform": "webhook",
            "ingress_route": "relay",
            "source_profile": "dispatcher",
            "target_profile": "market-analysis",
            "effective_toolsets": ["terminal", "web"],
            "delivery_platform": "discord",
            "delivery_chat_id": "market-room",
            "handoff_depth": 1,
        },
    )
    prompt = build_session_context_prompt(
        SessionContext(source=source, connected_platforms=[], home_channels={})
    )
    assert "Ingress route: relay" in prompt
    assert "Source profile: dispatcher" in prompt
    assert "Target profile: market-analysis" in prompt
    assert "Effective toolsets: terminal, web" in prompt
    assert "Delivery destination: discord/market-room" in prompt
