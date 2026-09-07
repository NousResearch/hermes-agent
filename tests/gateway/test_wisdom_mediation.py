import asyncio
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.platforms.base import BasePlatformAdapter
from gateway.wisdom_command import WisdomAction, WisdomItem, WisdomView
from tests.gateway.test_slack_wisdom import _adapter as slack_adapter
from tests.gateway.test_telegram_wisdom_command import _adapter as telegram_adapter
from tui_gateway.wisdom_mediation import poll


@pytest.mark.asyncio
async def test_idle_boundary_does_not_overlap_busy_turn_and_releases_guard():
    adapter = SimpleNamespace(
        _active_sessions={"busy": asyncio.Event()}, _session_tasks={}
    )
    adapter._drain_pending_after_session_command = AsyncMock(
        side_effect=lambda key, guard: adapter._active_sessions.pop(key)
    )
    work = AsyncMock()
    assert not await BasePlatformAdapter.run_idle_activity(adapter, "busy", work)
    work.assert_not_called()

    async def inside():
        assert "idle" in adapter._active_sessions
        assert adapter._session_tasks["idle"] is asyncio.current_task()

    assert await BasePlatformAdapter.run_idle_activity(adapter, "idle", inside)
    assert (
        "idle" not in adapter._active_sessions and "idle" not in adapter._session_tasks
    )


@pytest.mark.asyncio
async def test_idle_boundary_failure_releases_guard():
    adapter = SimpleNamespace(_active_sessions={}, _session_tasks={})
    adapter._drain_pending_after_session_command = AsyncMock(
        side_effect=lambda key, guard: adapter._active_sessions.pop(key)
    )
    with pytest.raises(RuntimeError):
        await BasePlatformAdapter.run_idle_activity(
            adapter, "session", AsyncMock(side_effect=RuntimeError)
        )
    assert not adapter._active_sessions and not adapter._session_tasks


def view():
    return WisdomView(
        "Collective Wisdom",
        "A recommendation",
        items=[
            WisdomItem(
                "<untrusted>",
                "Canonical warnings",
                actions=[
                    WisdomAction("Not Now", callback_data="wi:agent:defer:identity"),
                    WisdomAction(
                        "Review first", callback_data="wi:agent:inspect:identity"
                    ),
                    WisdomAction(
                        "Install",
                        callback_data="wi:agent:confirm:identity",
                        primary=True,
                    ),
                ],
            )
        ],
    )


@pytest.mark.asyncio
async def test_slack_proactive_advice_cannot_consume_slash_response():
    adapter = slack_adapter()
    adapter._team_clients["T1"].chat_postMessage.return_value = {
        "ok": True,
        "channel": "D1",
        "ts": "123.456",
    }
    adapter._pop_slash_context = Mock(
        side_effect=AssertionError("must not consume slash response")
    )
    receipt = await adapter.send_wisdom_mediation(
        view(), source=SimpleNamespace(chat_id="D1", scope_id="T1", thread_id="123")
    )
    assert receipt.message_id == "123.456" and receipt.scope_id == "T1"
    sent = adapter._team_clients["T1"].chat_postMessage.call_args.kwargs
    assert sent["thread_ts"] == "123"
    actions = [block for block in sent["blocks"] if block["type"] == "actions"][0][
        "elements"
    ]
    assert [button["text"]["text"] for button in actions] == [
        "Not Now",
        "Review first",
        "Install",
    ]
    assert actions[-1]["style"] == "primary"


@pytest.mark.asyncio
async def test_telegram_native_rich_controls_escape_publisher_text():
    adapter = telegram_adapter()
    adapter._bot.do_api_request.return_value = {"message_id": 19, "chat": {"id": 42}}
    receipt = await adapter.send_wisdom_mediation(
        view(), source=SimpleNamespace(chat_id="42", thread_id=None)
    )
    assert receipt.message_id == "19" and receipt.acknowledgement == "provider_accepted"
    sent = adapter._bot.do_api_request.call_args.kwargs["api_kwargs"]
    html = sent["rich_message"]["html"]
    assert "&lt;untrusted&gt;" in html
    assert html.index("Not Now") < html.index("Review first") < html.index("Install")


@pytest.mark.asyncio
async def test_expanded_checks_edit_keeps_full_checklist():
    adapter = telegram_adapter()
    current = view()
    current.items[0].detail = "Review detail " * 60 + "Final professionalism row"
    query = SimpleNamespace(message=SimpleNamespace(chat_id=42, message_id=19))
    await adapter._edit_wisdom_command_view(query, current, full_details=True)
    sent = adapter._bot.do_api_request.call_args.kwargs["api_kwargs"]
    assert "Final professionalism row" in sent["rich_message"]["html"]


@pytest.mark.asyncio
async def test_telegram_ambiguous_send_never_falls_back_to_duplicate_message():
    adapter = telegram_adapter()
    adapter._bot.do_api_request.side_effect = TimeoutError
    adapter._send_message_with_thread_fallback = AsyncMock()
    with pytest.raises(TimeoutError):
        await adapter.send_wisdom_mediation(
            view(), source=SimpleNamespace(chat_id="42", thread_id=None)
        )
    adapter._send_message_with_thread_fallback.assert_not_called()


@pytest.mark.parametrize(
    "busy,connected,approval",
    [(True, True, False), (False, False, False), (False, True, True)],
)
def test_tui_defers_without_model_or_history_changes(
    monkeypatch, busy, connected, approval
):
    monkeypatch.setattr(
        "tools.approval.get_pending_gateway_approval", lambda _: approval
    )
    monkeypatch.setattr("tools.clarify_gateway.has_pending", lambda _: False)
    monkeypatch.setattr(
        "hermes_wisdom.service.WisdomService",
        Mock(side_effect=AssertionError("must not poll")),
    )
    session = {
        "session_key": "s",
        "_wisdom_user_activity": time.time(),
        "agent": object(),
        "history_lock": threading.Lock(),
        "running": busy,
        "history": [{"role": "user", "content": "hello"}],
    }
    emit = Mock()
    poll(
        session,
        emit=emit,
        profile_scope=lambda _: nullcontext(),
        connected=lambda: connected,
    )
    assert session["running"] is busy
    assert session["history"] == [{"role": "user", "content": "hello"}]
    emit.assert_not_called()


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("accepted", [True, False, None, "error"])
def test_tui_assessment_does_not_overwrite_a_new_turn_after_stop(
    monkeypatch, cancel, accepted
):
    monkeypatch.setattr("tools.approval.get_pending_gateway_approval", lambda _: False)
    monkeypatch.setattr("tools.clarify_gateway.has_pending", lambda _: False)
    service = Mock()
    service.store.active_org_id.return_value = "org"
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", lambda: service)
    mediation = Mock()
    mediation.queue.claim_refresh.return_value = False
    monkeypatch.setattr(
        "tui_gateway.wisdom_mediation.WisdomMediation", lambda _: mediation
    )
    session = {
        "session_key": "s",
        "_wisdom_user_activity": time.time(),
        "agent": Mock(),
        "history_lock": threading.Lock(),
        "running": False,
        "history": [],
    }

    def prepare(*args, **kwargs):
        assert session["running"]
        if cancel:
            session["_queued_prompt_generation"] = 1
        return [
            {
                "assessment": {"id": "event", "lease_token": "token"},
                "advice": {
                    "title": "Advice",
                    "explanation": "Useful",
                    "relevance": "digest",
                },
                "interaction": None,
            }
        ]

    mediation.prepare.side_effect = prepare
    mediation.begin_delivery.side_effect = lambda org, items: items
    emit = Mock(return_value=accepted)
    if accepted == "error":
        emit.side_effect = RuntimeError("write failed")
    if accepted == "error" and not cancel:
        with pytest.raises(RuntimeError):
            poll(session, emit=emit, profile_scope=lambda _: nullcontext())
    else:
        poll(session, emit=emit, profile_scope=lambda _: nullcontext())
    assert session["history"] == []
    assert session["running"] is cancel
    if cancel:
        mediation.begin_delivery.assert_not_called()
        emit.assert_not_called()
    else:
        emit.assert_called_once()
        assert emit.call_args.args[0] == "notification.show"
        if accepted is True:
            mediation.queue.complete_delivery.assert_called_once()
            receipt = mediation.queue.complete_delivery.call_args.kwargs["receipt"]
            assert receipt.acknowledgement == "transport_accepted"
            assert receipt.destination == "local:s"
            mediation.queue.uncertain_delivery.assert_not_called()
        else:
            mediation.queue.complete_delivery.assert_not_called()
            mediation.queue.uncertain_delivery.assert_called_once_with(
                "org", "event", "token"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, {}, {"message_id": 1, "chat": {"id": 99}}])
async def test_telegram_missing_receipt_does_not_trigger_legacy_resend(response):
    adapter = telegram_adapter()
    adapter._bot.do_api_request.return_value = response
    adapter._send_message_with_thread_fallback = AsyncMock()
    with pytest.raises(ValueError):
        await adapter.send_wisdom_mediation(
            view(), source=SimpleNamespace(chat_id="42", thread_id=None)
        )
    adapter._send_message_with_thread_fallback.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_definite_rejection_uses_fallback_receipt():
    from telegram.error import BadRequest

    adapter = telegram_adapter()
    adapter._bot.do_api_request.side_effect = BadRequest("unsupported")
    adapter._send_message_with_thread_fallback = AsyncMock(
        return_value={"message_id": 21, "chat": {"id": 42}}
    )
    receipt = await adapter.send_wisdom_mediation(
        view(), source=SimpleNamespace(chat_id="42", thread_id=None)
    )
    assert receipt.message_id == "21"
    adapter._send_message_with_thread_fallback.assert_awaited_once()


@pytest.mark.asyncio
async def test_slack_unacknowledged_response_is_not_success():
    adapter = slack_adapter()
    adapter._team_clients["T1"].chat_postMessage.return_value = {"ok": False}
    with pytest.raises(ValueError):
        await adapter.send_wisdom_mediation(
            view(), source=SimpleNamespace(chat_id="D1", scope_id="T1", thread_id=None)
        )
    adapter._team_clients["T1"].chat_postMessage.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        "accepted",
        "timeout",
        "missing_receipt",
        "persistence_failure",
        "cancelled",
        "disconnected",
        "expired_reservation",
    ],
)
async def test_scheduler_commits_receipt_or_uncertainty_to_real_ledger(
    tmp_path, monkeypatch, outcome
):
    from gateway.wisdom_mediation import schedule
    from hermes_wisdom.delivery import DeliveryReceipt
    from hermes_wisdom.client_delivery import ClientDeliveryResponse
    from hermes_wisdom.delivery_outbox import DeliveryOutbox
    from hermes_wisdom.mediation_store import MediationStore
    from hermes_wisdom.store import WisdomStore

    store = WisdomStore(tmp_path / "wisdom")
    store.activate_installation_identity("installation", "org")
    queue = MediationStore(store)
    queue.register_session(
        "org",
        session_key="session",
        session_id="session",
        platform="telegram",
        actor_id="user",
        private=True,
        available=True,
        user_activity=True,
        address={"chat_id": "42"},
    )
    identity = queue.enqueue(
        "org", "feed:event", {"kind": "skill", "skill_id": "skill", "version": 1}
    )
    job = queue.claim("org", "session")[0]
    advice = {"title": "Arrival", "explanation": "For review", "relevance": "digest"}
    assert queue.save_advice("org", identity, job["lease_token"], advice)
    job["state"] = "ready"
    item = {"assessment": job, "advice": advice, "interaction": None}
    service = Mock(store=store)
    service.client.identity = {"owner": "account-user"}
    service.client.display_org_id = "org"
    outbox = DeliveryOutbox(service, clock=queue.clock)

    def reserve(request_id, reference):
        from datetime import datetime, timezone

        return ClientDeliveryResponse(
            org_id="org",
            recipient_user_id="account-user",
            event_id="event",
            request_id=request_id,
            reference=reference,
            state="claimed",
            reason=None,
            lease_until=datetime
            .fromtimestamp(time.time() + 120, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        )

    service.client.claim_notification_delivery.side_effect = reserve
    service.client.settle_notification_delivery.side_effect = (
        lambda event_id, request_id, reference, **kwargs: ClientDeliveryResponse(
            org_id="org",
            recipient_user_id="account-user",
            event_id=event_id,
            request_id=request_id,
            reference=reference,
            state=kwargs["outcome"],
            lease_until=None,
            reason=None,
        )
    )
    if outcome == "persistence_failure":
        with store.transaction() as db:
            db.execute(
                "CREATE TRIGGER reject_receipt BEFORE INSERT ON wisdom_delivery_receipt BEGIN SELECT RAISE(ABORT,'injected'); END"
            )
    mediation = Mock(queue=queue)
    queue.claim_refresh = Mock(return_value=False)
    mediation.prepare.return_value = [item]

    def begin(org, group):
        selected = []
        for item in group:
            job = item["assessment"]
            request_id = outbox.reserve(org, job)
            if request_id and queue.begin_delivery(
                org, job["id"], job["lease_token"], request_id=request_id
            ):
                selected.append(item)
        if outcome == "disconnected":
            adapter._active_sessions["session"].set()
        elif outcome == "expired_reservation":
            with store.transaction() as db:
                db.execute("UPDATE wisdom_remote_delivery SET lease_until=0")
        return selected

    mediation.begin_delivery.side_effect = begin
    mediation.delivery_ready.side_effect = lambda org, items: all(
        queue.delivery_ready(
            org,
            i["assessment"]["id"],
            i["assessment"]["lease_token"],
            user_id="account-user",
        )
        for i in items
    )
    mediation.cancel_delivery.side_effect = lambda org, items: [
        queue.cancel_delivery(
            org, i["assessment"]["id"], i["assessment"]["lease_token"]
        )
        for i in items
    ]
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", lambda: service)
    monkeypatch.setattr("gateway.wisdom_mediation.WisdomMediation", lambda _: mediation)
    monkeypatch.setattr("gateway.wisdom_mediation.delivery_mode", lambda: "agent")
    monkeypatch.setattr("tools.approval.get_pending_gateway_approval", lambda _: None)
    monkeypatch.setattr("tools.clarify_gateway.has_pending", lambda _: False)

    async def immediate_sleep(_):
        pass

    monkeypatch.setattr("gateway.wisdom_mediation.asyncio.sleep", immediate_sleep)
    adapter = SimpleNamespace(_active_sessions={}, _background_tasks=set())

    async def scoped(fn):
        return fn()

    adapter._run_wisdom_profile_operation = scoped
    adapter.send_wisdom_mediation = AsyncMock(
        return_value=DeliveryReceipt(
            platform="telegram",
            destination="42",
            message_id="19",
            acknowledgement="provider_accepted",
        )
    )
    if outcome == "timeout":
        adapter.send_wisdom_mediation.side_effect = TimeoutError
    elif outcome == "cancelled":
        adapter.send_wisdom_mediation.side_effect = asyncio.CancelledError
    elif outcome == "missing_receipt":
        adapter.send_wisdom_mediation.return_value = None
    gateway = SimpleNamespace(
        _agent_cache_lock=threading.Lock(),
        _agent_cache={
            "session": SimpleNamespace(
                _session_messages=[], provider="test", model="test"
            )
        },
        _is_user_authorized=lambda _: True,
        _session_key_for_source=lambda _: "session",
    )

    async def idle(key, tick):
        adapter._active_sessions[key] = asyncio.Event()
        try:
            await tick()
        finally:
            gateway._wisdom_mediation_active_until[key] = 0
            adapter._active_sessions.pop(key)

    adapter.run_idle_activity = idle
    source = SimpleNamespace(
        platform="telegram", chat_type="dm", chat_id="42", user_id="user"
    )
    assert await schedule(gateway, adapter, source, "session")
    task = next(iter(adapter._background_tasks))
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    row = queue.assessments("org")[0]
    if outcome in {"disconnected", "expired_reservation"}:
        assert row["state"] == "ready" and row["advice"] == advice
        adapter.send_wisdom_mediation.assert_not_awaited()
        outbox.flush("org")
        assert (
            service.client.settle_notification_delivery.call_args.kwargs["outcome"]
            == "not_sent"
        )
        return
    assert row["state"] == (
        "delivered" if outcome == "accepted" else "delivery_uncertain"
    )
    assert queue.introduced("org") is (outcome == "accepted")
    with store.transaction() as db:
        count = db.execute("SELECT COUNT(*) FROM wisdom_delivery_receipt").fetchone()[0]
    assert count == int(outcome == "accepted")
    assert queue.claim("org", "session") == []
    adapter.send_wisdom_mediation.assert_awaited_once()
    outbox.flush("org")
    assert service.client.claim_notification_delivery.call_count == 1
    assert service.client.settle_notification_delivery.call_args.kwargs["outcome"] == (
        "acknowledged" if outcome == "accepted" else "uncertain"
    )
