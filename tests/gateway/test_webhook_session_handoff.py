"""Webhook-side contracts for durable messaging session handoff."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.base import ProcessingOutcome, SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


def _make_adapter(routes) -> WebhookAdapter:
    return WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 0, "routes": routes},
        )
    )


def _handoff_routes(**overrides):
    route = {
        "secret": _INSECURE_NO_AUTH,
        "prompt": "{message}",
        "handoff_to": "discord",
    }
    route.update(overrides)
    return {"alerts": route}


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    app.router.add_post(
        "/p/{profile}/webhooks/{route_name}", adapter._handle_webhook
    )
    return app


def _delivery_state(
    adapter: WebhookAdapter,
    delivery_id: str,
    *,
    session_id=None,
    platform: str = "discord",
):
    marker = adapter._handoff_delivery_marker(
        profile=None,
        route_name="alerts",
        delivery_id=delivery_id,
    )
    key = adapter._handoff_delivery_state_key(marker)
    value = adapter._handoff_delivery_state_value(
        marker,
        platform,
        session_id=session_id,
    )
    return marker, key, value


def _make_event(adapter: WebhookAdapter, delivery_id: str = "delivery-1"):
    chat_id = f"webhook:alerts:{delivery_id}"
    source = adapter.build_source(
        chat_id=chat_id,
        chat_name="webhook/alerts",
        chat_type="webhook",
        user_id="webhook:alerts",
        user_name="alerts",
    )
    from gateway.platforms.base import MessageEvent

    marker, _, _ = _delivery_state(adapter, delivery_id)
    event = MessageEvent(text="alert", source=source, message_id=delivery_id)
    event.metadata.update(
        {
            "_webhook_handoff_to": "discord",
            "_webhook_handoff_delivery": marker,
        }
    )
    return event, marker


def _wire_lifecycle_runner(
    adapter: WebhookAdapter,
    *,
    request_result=True,
    request_error=None,
    handoff_state=None,
    compare_result=True,
    stored_state=None,
    finalize_result=True,
):
    store = SimpleNamespace(
        peek_session_id=AsyncMock(return_value="session-exact"),
        remove_session_route_and_end=AsyncMock(return_value=finalize_result),
    )
    request = AsyncMock(
        side_effect=request_error,
        return_value=request_result,
    )
    _, _, default_bound_state = _delivery_state(
        adapter,
        "delivery-1",
        session_id="session-exact",
    )
    db = SimpleNamespace(
        compare_and_set_meta=AsyncMock(return_value=compare_result),
        get_meta=AsyncMock(
            return_value=(
                default_bound_state if stored_state is None else stored_state
            )
        ),
        request_handoff_once=request,
        get_handoff_state=AsyncMock(return_value=handoff_state),
    )
    adapter.gateway_runner = SimpleNamespace(
        async_session_store=store,
        _session_db=db,
        _session_key_for_source=lambda source: f"key:{source.chat_id}",
    )
    return store, db


class TestHandoffConfiguration:
    def test_discord_is_the_initial_trusted_target(self):
        assert (
            WebhookAdapter._validate_handoff_target(
                "alerts", {"handoff_to": " Discord "}
            )
            == "discord"
        )

    @pytest.mark.parametrize("target", [None, "", "telegram", "{payload.target}", 1])
    def test_invalid_or_untrusted_target_is_rejected(self, target):
        with pytest.raises(ValueError, match="handoff_to"):
            WebhookAdapter._validate_handoff_target(
                "alerts", {"handoff_to": target}
            )

    def test_deliver_only_is_incompatible(self):
        with pytest.raises(ValueError, match="deliver_only=true"):
            WebhookAdapter._validate_handoff_target(
                "alerts",
                {"handoff_to": "discord", "deliver_only": True},
            )

    @pytest.mark.parametrize("profile", ["work", " Work ", "", None, 7])
    def test_named_or_invalid_profile_handoff_is_rejected(self, profile):
        with pytest.raises(ValueError, match="named multiplex profile"):
            WebhookAdapter._validate_handoff_target(
                "alerts",
                {"handoff_to": "discord", "profile": profile},
            )

    def test_explicit_default_profile_handoff_is_allowed(self):
        assert (
            WebhookAdapter._validate_handoff_target(
                "alerts",
                {"handoff_to": "discord", "profile": "default"},
            )
            == "discord"
        )

    def test_route_without_handoff_is_unchanged(self):
        assert WebhookAdapter._validate_handoff_target("alerts", {}) is None


@pytest.mark.asyncio
async def test_handoff_target_is_config_only_not_payload_interpolation():
    adapter = _make_adapter(_handoff_routes(deliver="discord"))
    claim = AsyncMock(return_value=True)
    adapter.gateway_runner = SimpleNamespace(
        _session_db=SimpleNamespace(set_meta_if_absent=claim)
    )
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture
    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/webhooks/alerts",
            json={"message": "hello", "handoff_to": "telegram"},
            headers={"X-GitHub-Delivery": "trusted-target-1"},
        )
        assert response.status == 202

    await asyncio.sleep(0)
    assert len(captured) == 1
    event = captured[0]
    assert event.metadata["_webhook_handoff_to"] == "discord"
    assert adapter._delivery_info[event.source.chat_id]["handoff_to"] == "discord"
    marker, state_key, accepted_state = _delivery_state(
        adapter, "trusted-target-1"
    )
    assert event.metadata["_webhook_handoff_delivery"] == marker
    claim.assert_awaited_once_with(state_key, accepted_state)


@pytest.mark.asyncio
async def test_default_profile_url_aliases_share_one_durable_claim():
    adapter = _make_adapter(_handoff_routes())
    durable = {}

    async def _set_meta_if_absent(key, value):
        if key in durable:
            return False
        durable[key] = value
        return True

    async def _get_meta(key):
        return durable[key]

    db = SimpleNamespace(
        set_meta_if_absent=AsyncMock(side_effect=_set_meta_if_absent),
        get_meta=AsyncMock(side_effect=_get_meta),
    )
    adapter.gateway_runner = SimpleNamespace(
        config=SimpleNamespace(
            multiplex_profiles=True,
            multiplex_profile_allowlist=[],
        ),
        _session_db=db,
        _profile_name_for_source=lambda _source: None,
    )
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        headers = {"X-GitHub-Delivery": "default-profile-alias"}
        first = await client.post(
            "/webhooks/alerts", json={"message": "same"}, headers=headers
        )
        second = await client.post(
            "/p/default/webhooks/alerts",
            json={"message": "same"},
            headers=headers,
        )
        second_body = await second.json()

    assert first.status == 202
    assert second.status == 200
    assert second_body["status"] == "duplicate"
    assert len(durable) == 1
    assert db.set_meta_if_absent.await_count == 2
    first_claim, second_claim = db.set_meta_if_absent.await_args_list
    assert first_claim.args == second_claim.args
    await asyncio.sleep(0)
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].source.profile == "default"


@pytest.mark.asyncio
async def test_handoff_send_suppresses_legacy_parent_delivery():
    adapter = _make_adapter({})
    chat_id = "webhook:alerts:no-parent-copy"
    adapter._delivery_info[chat_id] = {
        "deliver": "discord",
        "deliver_extra": {"chat_id": "parent-channel"},
        "handoff_to": "discord",
    }
    adapter._deliver_cross_platform = AsyncMock(
        return_value=SendResult(success=True)
    )

    result = await adapter.send(chat_id, "completed response")

    assert result.success is True
    adapter._deliver_cross_platform.assert_not_awaited()


@pytest.mark.asyncio
async def test_handoff_send_stays_suppressed_after_delivery_snapshot_prunes():
    adapter = _make_adapter(_handoff_routes(deliver="discord"))
    adapter.gateway_runner = SimpleNamespace(
        _session_db=SimpleNamespace(
            set_meta_if_absent=AsyncMock(return_value=True)
        )
    )
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture
    adapter._deliver_cross_platform = AsyncMock(
        return_value=SendResult(success=True)
    )

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/webhooks/alerts",
            json={"message": "long-running handoff"},
            headers={"X-GitHub-Delivery": "long-running-handoff"},
        )
        assert response.status == 202

    await asyncio.sleep(0)
    chat_id = captured[0].source.chat_id
    adapter._delivery_info.clear()

    result = await adapter.send(chat_id, "must not reach the legacy target")

    assert result.success is True
    adapter._deliver_cross_platform.assert_not_awaited()


@pytest.mark.asyncio
async def test_success_binds_claim_and_requests_exact_session_once():
    adapter = _make_adapter({})
    store, db = _wire_lifecycle_runner(adapter)
    event, marker = _make_event(adapter)
    event._agent_run_failed = False

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    # A duplicate lifecycle callback in the same process must be a no-op.
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    state_key = adapter._handoff_delivery_state_key(marker)
    accepted_state = adapter._handoff_delivery_state_value(
        marker, "discord", session_id=None
    )
    bound_state = adapter._handoff_delivery_state_value(
        marker, "discord", session_id="session-exact"
    )
    db.compare_and_set_meta.assert_awaited_once_with(
        state_key,
        accepted_state,
        bound_state,
    )
    db.request_handoff_once.assert_awaited_once_with("session-exact", "discord")
    store.remove_session_route_and_end.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_matching_bound_request_is_idempotent():
    adapter = _make_adapter({})
    store, db = _wire_lifecycle_runner(
        adapter,
        compare_result=False,
        request_result=False,
        handoff_state={
            "state": "pending",
            "platform": "discord",
            "error": None,
        },
    )
    event, _ = _make_event(adapter)
    event._agent_run_failed = False

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    db.get_meta.assert_awaited_once()
    db.request_handoff_once.assert_awaited_once_with("session-exact", "discord")
    db.get_handoff_state.assert_awaited_once_with("session-exact")
    store.remove_session_route_and_end.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_only_success_requests_handoff_despite_delivery_failure_outcome():
    """The runner's explicit success stamp overrides Base's text-only accounting."""
    adapter = _make_adapter({})
    store, db = _wire_lifecycle_runner(adapter)
    event, marker = _make_event(adapter)
    event._agent_run_failed = False

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    state_key = adapter._handoff_delivery_state_key(marker)
    accepted_state = adapter._handoff_delivery_state_value(
        marker, "discord", session_id=None
    )
    bound_state = adapter._handoff_delivery_state_value(
        marker, "discord", session_id="session-exact"
    )
    db.compare_and_set_meta.assert_awaited_once_with(
        state_key,
        accepted_state,
        bound_state,
    )
    db.request_handoff_once.assert_awaited_once_with("session-exact", "discord")
    store.remove_session_route_and_end.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_only_success_requests_handoff_through_real_adapter_lifecycle():
    """Attachment extraction must not turn a successful run into a finalization."""
    adapter = _make_adapter({})
    store, db = _wire_lifecycle_runner(adapter)
    event, _ = _make_event(adapter)
    adapter._active_handoff_sessions.add(event.source.chat_id)

    async def _media_only_handler(current_event):
        current_event._agent_run_failed = False
        return "![generated result](https://example.com/result.png)"

    adapter._message_handler = _media_only_handler
    await adapter.handle_message(event)

    for _ in range(100):
        if not adapter._background_tasks:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("webhook background lifecycle did not finish")
    await asyncio.sleep(0)

    db.request_handoff_once.assert_awaited_once_with("session-exact", "discord")
    store.remove_session_route_and_end.assert_not_awaited()
    assert event.source.chat_id not in adapter._active_handoff_sessions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (ProcessingOutcome.FAILURE, "webhook_handoff_failed"),
        (ProcessingOutcome.CANCELLED, "webhook_handoff_cancelled"),
    ],
)
async def test_failure_or_cancellation_removes_source_and_finalizes(outcome, reason):
    adapter = _make_adapter({})
    store, db = _wire_lifecycle_runner(adapter)
    event, _ = _make_event(adapter)

    await adapter.on_processing_complete(event, outcome)

    store.remove_session_route_and_end.assert_awaited_once_with(
        "key:webhook:alerts:delivery-1", "session-exact", reason
    )
    db.request_handoff_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_failure_with_explicit_agent_failure_still_finalizes():
    adapter = _make_adapter({})
    store, db = _wire_lifecycle_runner(adapter)
    event, _ = _make_event(adapter)
    event._agent_run_failed = True

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    store.remove_session_route_and_end.assert_awaited_once_with(
        "key:webhook:alerts:delivery-1",
        "session-exact",
        "webhook_handoff_failed",
    )
    db.request_handoff_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_with_explicit_agent_success_still_finalizes():
    adapter = _make_adapter({})
    store, db = _wire_lifecycle_runner(adapter)
    event, _ = _make_event(adapter)
    event._agent_run_failed = False

    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    store.remove_session_route_and_end.assert_awaited_once_with(
        "key:webhook:alerts:delivery-1",
        "session-exact",
        "webhook_handoff_cancelled",
    )
    db.request_handoff_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_failure_marker_overrides_delivery_success():
    """A rendered error response must not turn an agent failure into handoff."""
    adapter = _make_adapter({})
    store, db = _wire_lifecycle_runner(adapter)
    event, _ = _make_event(adapter)
    event._agent_run_failed = True

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    store.remove_session_route_and_end.assert_awaited_once_with(
        "key:webhook:alerts:delivery-1",
        "session-exact",
        "webhook_handoff_failed",
    )
    db.request_handoff_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_success_without_agent_completion_marker_finalizes():
    """A pre-agent early return cannot create an empty handoff thread."""
    adapter = _make_adapter({})
    store, db = _wire_lifecycle_runner(adapter)
    event, _ = _make_event(adapter)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    store.remove_session_route_and_end.assert_awaited_once_with(
        "key:webhook:alerts:delivery-1",
        "session-exact",
        "webhook_handoff_failed",
    )
    db.request_handoff_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_failure_removes_source_and_finalizes():
    adapter = _make_adapter({})
    store, db = _wire_lifecycle_runner(
        adapter,
        request_error=RuntimeError("database unavailable"),
    )
    event, _ = _make_event(adapter)
    event._agent_run_failed = False

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    store.remove_session_route_and_end.assert_awaited_once_with(
        "key:webhook:alerts:delivery-1",
        "session-exact",
        "webhook_handoff_request_failed",
    )


@pytest.mark.asyncio
async def test_cancellation_during_shielded_request_leaves_pending_for_watcher():
    adapter = _make_adapter({})
    store, db = _wire_lifecycle_runner(adapter)
    event, _ = _make_event(adapter)
    event._agent_run_failed = False
    request_started = asyncio.Event()
    release_request = asyncio.Event()
    durable = {"state": None}

    async def _request_handoff(session_id, platform):
        assert (session_id, platform) == ("session-exact", "discord")
        request_started.set()
        await release_request.wait()
        durable["state"] = "pending"
        return True

    db.request_handoff_once.side_effect = _request_handoff
    success_hook = asyncio.create_task(
        adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    )
    await request_started.wait()

    success_hook.cancel()
    await asyncio.sleep(0)
    assert not success_hook.done()
    release_request.set()
    with pytest.raises(asyncio.CancelledError):
        await success_hook

    # BasePlatformAdapter invokes the hook again with CANCELLED while unwinding.
    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    assert durable["state"] == "pending"
    assert event.metadata["_webhook_handoff_requested"] is True
    db.request_handoff_once.assert_awaited_once_with("session-exact", "discord")
    store.remove_session_route_and_end.assert_not_awaited()


@pytest.mark.asyncio
async def test_durable_duplicate_after_restart_skips_second_agent_run():
    adapter = _make_adapter(_handoff_routes())
    marker, state_key, bound_state = _delivery_state(
        adapter,
        "restart-duplicate-1",
        session_id="original-session",
    )
    db = SimpleNamespace(
        set_meta_if_absent=AsyncMock(return_value=False),
        get_meta=AsyncMock(return_value=bound_state),
        get_session=AsyncMock(
            return_value={"id": "original-session", "ended_at": None}
        ),
        get_handoff_state=AsyncMock(
            return_value={
                "state": "completed",
                "platform": "discord",
                "error": None,
            }
        ),
        request_handoff_once=AsyncMock(),
    )
    adapter.gateway_runner = SimpleNamespace(_session_db=db)
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/webhooks/alerts",
            json={"message": "retry"},
            headers={"X-GitHub-Delivery": "restart-duplicate-1"},
        )
        assert response.status == 200
        assert (await response.json())["status"] == "duplicate"

    adapter.handle_message.assert_not_awaited()
    db.set_meta_if_absent.assert_awaited_once_with(
        state_key,
        adapter._handoff_delivery_state_value(
            marker, "discord", session_id=None
        ),
    )
    db.get_meta.assert_awaited_once_with(state_key)
    db.get_session.assert_awaited_once_with("original-session")
    db.request_handoff_once.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lifecycle", "session_row", "handoff_state"),
    [
        (
            "failed",
            {"id": "original-session", "ended_at": None},
            {"state": "failed", "platform": "discord", "error": "send failed"},
        ),
        (
            "ended",
            {"id": "original-session", "ended_at": 1, "end_reason": "agent_close"},
            None,
        ),
        (
            "reset",
            {"id": "original-session", "ended_at": 1, "end_reason": "session_reset"},
            None,
        ),
        (
            "compression",
            {"id": "original-session", "ended_at": 1, "end_reason": "compression"},
            None,
        ),
    ],
)
async def test_duplicate_tombstone_uses_original_identity_after_lifecycle_change(
    lifecycle,
    session_row,
    handoff_state,
):
    adapter = _make_adapter(_handoff_routes())
    delivery_id = f"duplicate-after-{lifecycle}"
    _, _, bound_state = _delivery_state(
        adapter,
        delivery_id,
        session_id="original-session",
    )
    db = SimpleNamespace(
        set_meta_if_absent=AsyncMock(return_value=False),
        get_meta=AsyncMock(return_value=bound_state),
        get_session=AsyncMock(return_value=session_row),
        get_handoff_state=AsyncMock(return_value=handoff_state),
        request_handoff_once=AsyncMock(),
    )
    adapter.gateway_runner = SimpleNamespace(_session_db=db)
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/webhooks/alerts",
            json={"message": "same provider delivery"},
            headers={"X-GitHub-Delivery": delivery_id},
        )
        body = await response.json()

    assert response.status == 200
    assert body["status"] == "duplicate"
    db.get_session.assert_awaited_once_with("original-session")
    if lifecycle == "failed":
        db.get_handoff_state.assert_awaited_once_with("original-session")
    else:
        db.get_handoff_state.assert_not_awaited()
    db.request_handoff_once.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_recovers_crash_between_binding_and_request():
    adapter = _make_adapter(_handoff_routes())
    _, _, bound_state = _delivery_state(
        adapter,
        "bound-before-request",
        session_id="bound-session",
    )
    db = SimpleNamespace(
        set_meta_if_absent=AsyncMock(return_value=False),
        get_meta=AsyncMock(return_value=bound_state),
        get_session=AsyncMock(
            return_value={"id": "bound-session", "ended_at": None}
        ),
        get_handoff_state=AsyncMock(
            return_value={"state": None, "platform": None, "error": None}
        ),
        request_handoff_once=AsyncMock(return_value=True),
    )
    adapter.gateway_runner = SimpleNamespace(_session_db=db)
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/webhooks/alerts",
            json={"message": "retry after crash"},
            headers={"X-GitHub-Delivery": "bound-before-request"},
        )
        body = await response.json()

    assert response.status == 200
    assert body["status"] == "duplicate"
    db.request_handoff_once.assert_awaited_once_with(
        "bound-session", "discord"
    )
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_with_mismatched_durable_target_fails_closed():
    adapter = _make_adapter(_handoff_routes())
    _, _, conflicting_state = _delivery_state(
        adapter,
        "mismatched-target",
        session_id="bound-session",
        platform="telegram",
    )
    db = SimpleNamespace(
        set_meta_if_absent=AsyncMock(return_value=False),
        get_meta=AsyncMock(return_value=conflicting_state),
        get_session=AsyncMock(),
        request_handoff_once=AsyncMock(),
    )
    adapter.gateway_runner = SimpleNamespace(_session_db=db)
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/webhooks/alerts",
            json={"message": "conflict"},
            headers={"X-GitHub-Delivery": "mismatched-target"},
        )

    assert response.status == 503
    db.get_session.assert_not_awaited()
    db.request_handoff_once.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_durable_claim_failure_returns_503_without_consuming_retry():
    adapter = _make_adapter(_handoff_routes())
    claim = AsyncMock(side_effect=[RuntimeError("db down"), True])
    adapter.gateway_runner = SimpleNamespace(
        _session_db=SimpleNamespace(set_meta_if_absent=claim)
    )
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        headers = {"X-GitHub-Delivery": "retry-after-store-failure"}
        first = await client.post(
            "/webhooks/alerts", json={"message": "retry"}, headers=headers
        )
        second = await client.post(
            "/webhooks/alerts", json={"message": "retry"}, headers=headers
        )

    assert first.status == 503
    assert second.status == 202
    assert claim.await_count == 2
    await asyncio.sleep(0)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_duplicate_claim_starts_exactly_one_agent_run():
    adapter = _make_adapter(_handoff_routes())
    durable = {"value": None, "arrived": 0}
    both_arrived = asyncio.Event()
    claim_lock = asyncio.Lock()

    async def _set_meta_if_absent(_key, value):
        durable["arrived"] += 1
        if durable["arrived"] == 2:
            both_arrived.set()
        await both_arrived.wait()
        async with claim_lock:
            if durable["value"] is not None:
                return False
            durable["value"] = value
            return True

    async def _get_meta(_key):
        return durable["value"]

    db = SimpleNamespace(
        set_meta_if_absent=AsyncMock(side_effect=_set_meta_if_absent),
        get_meta=AsyncMock(side_effect=_get_meta),
        get_session=AsyncMock(
            side_effect=AssertionError("an unbound duplicate has no session to recover")
        ),
    )
    adapter.gateway_runner = SimpleNamespace(_session_db=db)
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        headers = {"X-GitHub-Delivery": "concurrent-claim"}
        first, second = await asyncio.gather(
            client.post(
                "/webhooks/alerts", json={"message": "same"}, headers=headers
            ),
            client.post(
                "/webhooks/alerts", json={"message": "same"}, headers=headers
            ),
        )
        first_body, second_body = await asyncio.gather(
            first.json(), second.json()
        )

    assert sorted([first.status, second.status]) == [200, 202]
    assert sorted([first_body["status"], second_body["status"]]) == [
        "accepted",
        "duplicate",
    ]
    assert db.set_meta_if_absent.await_count == 2
    db.get_meta.assert_awaited_once()
    db.get_session.assert_not_awaited()
    await asyncio.sleep(0)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_route_does_not_consult_durable_handoff_index():
    adapter = _make_adapter(
        {
            "alerts": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "{message}",
                "deliver": "log",
            }
        }
    )
    claim = AsyncMock(
        side_effect=AssertionError("legacy route must not use state_meta claim")
    )
    adapter.gateway_runner = SimpleNamespace(
        _session_db=SimpleNamespace(set_meta_if_absent=claim)
    )
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/webhooks/alerts",
            json={"message": "legacy"},
            headers={"X-GitHub-Delivery": "legacy-1"},
        )
        assert response.status == 202

    await asyncio.sleep(0)
    claim.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
