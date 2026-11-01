"""Cross-process session lease-wait refresh must not flood HUMAN chat gateways (#89166).

While a durable session's turn lease is held by another Hermes process,
``agent.turn_facade_lease.admit_durable_turn_lease``'s ``_on_wait`` emits an initial
wait notice once and then a periodic "Still waiting ... (Ns)" refresh roughly every
15s. The refresh only makes sense on surfaces that can update the initial notice in
place: on human-chat adapters without ``send_or_update_status`` (WeCom, Weixin, QQ,
Signal, ... — all SUPPORTS_MESSAGE_EDITING = False) the status path falls back to a
plain send, and a two-minute wait produced eight standalone chat messages that
drowned the eventual delivery.

``_should_suppress_lease_wait_refresh`` gates the refresh by surface class, not by
edit-method absence alone: raw programmatic surfaces (webhook, api_server, ...) keep
the periodic heartbeat — their machine consumers read it as lease liveness (F-001 in
the #89180 review) — while the initial notice and the lease-timeout warning use
different wording and are always delivered.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent.turn_facade_lease import (
    SESSION_TURN_LEASE_WAIT_REFRESH_STATUS_TEMPLATE,
    admit_durable_turn_lease,
)
from gateway.config import Platform
from gateway.run import (
    _prepare_gateway_status_message,
    _send_or_update_status_coro,
)
from gateway.run_lease_refresh import _should_suppress_lease_wait_refresh

INITIAL_WAIT_NOTICE = (
    "⏳ Another Hermes process is using this session; "
    "waiting for it to finish before starting your turn..."
)
LEASE_TIMEOUT_WARNING = (
    "⏳ Another Hermes process kept this session busy too "
    "long. Your message was not processed - wait for the "
    "other process to finish, then send it again."
)


def _refresh(elapsed_seconds: int) -> str:
    return SESSION_TURN_LEASE_WAIT_REFRESH_STATUS_TEMPLATE.format(
        elapsed_seconds=elapsed_seconds
    )


def _human_chat_adapter_without_edits():
    # Shape of the WeCom/Weixin/QQ/Signal adapters: plain send, no
    # send_or_update_status, SUPPORTS_MESSAGE_EDITING = False.
    return SimpleNamespace(send=AsyncMock(), SUPPORTS_MESSAGE_EDITING=False)


def _editing_adapter():
    # Shape of the Telegram/Slack adapters: refreshes edit the same bubble.
    return SimpleNamespace(
        send=AsyncMock(),
        send_or_update_status=AsyncMock(),
        SUPPORTS_MESSAGE_EDITING=True,
    )


def _webhook_shape_adapter():
    # WebhookAdapter delivers interim statuses to machine consumers through plain
    # send and has no edit helper; the gate must NOT classify it as a flooding
    # human chat (F-001).
    return SimpleNamespace(send=AsyncMock())


def _prepared(platform, message):
    prepared = _prepare_gateway_status_message(platform, "lifecycle", message)
    assert prepared is not None, f"status must survive the noise filter: {message!r}"
    return prepared


def test_refresh_suppressed_on_human_chat_adapter_without_in_place_updates():
    adapter = _human_chat_adapter_without_edits()
    for elapsed in (15, 30, 105, 120):
        assert (
            _should_suppress_lease_wait_refresh(
                Platform.WECOM, adapter, _prepared(Platform.WECOM, _refresh(elapsed))
            )
            is True
        )


def test_refresh_delivered_when_adapter_edits_in_place():
    adapter = _editing_adapter()
    assert (
        _should_suppress_lease_wait_refresh(
            Platform.TELEGRAM, adapter, _prepared(Platform.TELEGRAM, _refresh(15))
        )
        is False
    )


def test_refresh_heartbeats_preserved_on_raw_programmatic_surfaces():
    # F-001: edit-method absence alone must not discard webhook lease liveness.
    for platform in (Platform.WEBHOOK, Platform.API_SERVER, Platform.MSGRAPH_WEBHOOK):
        adapter = _webhook_shape_adapter()
        assert (
            _should_suppress_lease_wait_refresh(
                platform, adapter, _prepared(platform, _refresh(15))
            )
            is False
        ), platform


def test_initial_wait_notice_never_suppressed():
    adapter = _human_chat_adapter_without_edits()
    assert (
        _should_suppress_lease_wait_refresh(
            Platform.WECOM, adapter, _prepared(Platform.WECOM, INITIAL_WAIT_NOTICE)
        )
        is False
    )


def test_lease_timeout_warning_never_suppressed():
    adapter = _human_chat_adapter_without_edits()
    assert (
        _should_suppress_lease_wait_refresh(
            Platform.WECOM, adapter, _prepared(Platform.WECOM, LEASE_TIMEOUT_WARNING)
        )
        is False
    )


def test_unrelated_lifecycle_status_never_suppressed():
    adapter = _human_chat_adapter_without_edits()
    assert (
        _should_suppress_lease_wait_refresh(
            Platform.WECOM, adapter,
            _prepared(Platform.WECOM, "Resumed session after gateway restart"),
        )
        is False
    )


class _LeaseWaitDB:
    """Fake store whose lease never frees: drives ``_on_wait`` through the initial
    notice (0s) and one 15s refresh, then reports the wait timeout."""

    def get_session(self, session_id):
        return {"session_id": session_id}

    def acquire_session_turn_lease(self, session_id, holder, *, ttl_seconds, wait_seconds,
                                   on_wait, on_contended, should_abort):
        assert not should_abort()
        on_wait(0.0)   # initial "Another Hermes process is using this session..."
        on_wait(15.0)  # the periodic refresh under route test
        return False   # wait exhausted -> lease timeout


def _drive_lease_wait_statuses():
    """Run the PRODUCTION lease-wait emitter; return every status it emitted, in order."""
    emitted = []

    agent = SimpleNamespace(
        _session_db=_LeaseWaitDB(),
        _persist_disabled=False,
        _interrupt_requested=False,
        _emit_status=lambda message: emitted.append(message),
        _emit_warning=lambda message: emitted.append(message),
    )
    admission = admit_durable_turn_lease(
        agent,
        session_id="s-lease",
        relay_turn_id="turn-1",
        task_context={"platform": "webhook"},
        conversation_history=[],
    )
    assert admission.early_result is not None
    assert admission.early_result["failure_reason"] == "session_busy"
    return emitted


def _deliver_statuses(platform, adapter, statuses):
    """Push emitted statuses through the gateway's real routing decision chain."""
    for message in statuses:
        prepared = _prepare_gateway_status_message(platform, "lifecycle", message)
        if prepared is None:
            continue
        if _should_suppress_lease_wait_refresh(platform, adapter, prepared):
            continue
        asyncio.run(
            _send_or_update_status_coro(adapter, "chat-1", "lifecycle", prepared, {})
        )


def test_composed_lease_wait_route_matrix():
    emitted = _drive_lease_wait_statuses()
    # The production emitter produced exactly: initial notice, 15s refresh, timeout warning.
    assert emitted == [
        INITIAL_WAIT_NOTICE,
        _refresh(15),
        LEASE_TIMEOUT_WARNING,
    ]

    # WEBHOOK (raw programmatic surface): every lease-wait heartbeat reaches send (F-001).
    webhook_adapter = _webhook_shape_adapter()
    _deliver_statuses(Platform.WEBHOOK, webhook_adapter, emitted)
    delivered = [c.args[1] for c in webhook_adapter.send.await_args_list]
    assert INITIAL_WAIT_NOTICE in delivered
    assert _refresh(15) in delivered
    assert LEASE_TIMEOUT_WARNING in delivered

    # WECOM (human chat, no in-place updates): initial + timeout still land; the periodic
    # refresh — and only it — is suppressed.
    wecom_adapter = _human_chat_adapter_without_edits()
    _deliver_statuses(Platform.WECOM, wecom_adapter, emitted)
    delivered = [c.args[1] for c in wecom_adapter.send.await_args_list]
    assert INITIAL_WAIT_NOTICE in delivered
    assert LEASE_TIMEOUT_WARNING in delivered
    assert _refresh(15) not in delivered

    # TELEGRAM-style (edits in place): the refresh routes through send_or_update_status.
    editing_adapter = _editing_adapter()
    _deliver_statuses(Platform.TELEGRAM, editing_adapter, emitted)
    updated = [c.args[2] for c in editing_adapter.send_or_update_status.await_args_list]
    assert _refresh(15) in updated
    assert editing_adapter.send.await_args_list == []
