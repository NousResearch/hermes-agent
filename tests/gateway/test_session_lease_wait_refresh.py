"""Cross-process session lease-wait refresh must not flood chat gateways (#89166).

While a durable session's turn lease is held by another Hermes process,
``run_agent._on_session_turn_lease_wait`` emits an initial wait notice once
and then a periodic "Still waiting ... (Ns)" refresh roughly every 15s. The
refresh only makes sense on surfaces that can update the initial notice in
place: on adapters without ``send_or_update_status`` (WeCom, Weixin, QQ,
Signal, ... — all SUPPORTS_MESSAGE_EDITING = False) the status path falls
back to a plain send, and a two-minute wait produced eight standalone
chat messages that drowned the eventual delivery.

``_should_suppress_lease_wait_refresh`` gates the refresh by adapter
capability; the initial notice and the lease-timeout warning use different
wording and must always be delivered.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform
from gateway.run import (
    _prepare_gateway_status_message,
    _session_turn_lease_refresh_re,
    _should_suppress_lease_wait_refresh,
)
from run_agent import SESSION_TURN_LEASE_WAIT_REFRESH_STATUS_TEMPLATE
from session_turn_lease import (
    SESSION_TURN_LEASE_WAIT_REFRESH_STATUS_TEMPLATE as OWNER_TEMPLATE,
    format_session_turn_lease_wait_refresh,
    session_turn_lease_refresh_re,
)

INITIAL_WAIT_NOTICE = (
    "⏳ Another Hermes process is using this session; "
    "waiting for it to finish before starting your turn..."
)
LEASE_TIMEOUT_WARNING = (
    "⏳ Another Hermes process kept this session busy too "
    "long. Your message was not processed - wait for the "
    "other process to finish, then send it again."
)


def _adapter_without_status_update():
    # Shape of the WeCom/Weixin/QQ/Signal adapters: plain send, no
    # send_or_update_status, SUPPORTS_MESSAGE_EDITING = False.
    return SimpleNamespace(send=AsyncMock(), SUPPORTS_MESSAGE_EDITING=False)


def _adapter_with_status_update():
    # Shape of the Telegram/Slack adapters: refreshes edit the same bubble.
    return SimpleNamespace(
        send=AsyncMock(),
        send_or_update_status=AsyncMock(),
        SUPPORTS_MESSAGE_EDITING=True,
    )


def _prepared_refresh(elapsed_seconds: int) -> str:
    message = format_session_turn_lease_wait_refresh(elapsed_seconds)
    prepared = _prepare_gateway_status_message(Platform.WECOM, "lifecycle", message)
    assert prepared is not None, "refresh must survive the noise filter first"
    return prepared


def test_refresh_owner_preserves_original_import_seams():
    assert SESSION_TURN_LEASE_WAIT_REFRESH_STATUS_TEMPLATE is OWNER_TEMPLATE
    assert _session_turn_lease_refresh_re is session_turn_lease_refresh_re


def test_refresh_suppressed_on_adapter_without_in_place_updates():
    adapter = _adapter_without_status_update()
    for elapsed in (15, 30, 105, 120):
        assert (
            _should_suppress_lease_wait_refresh(adapter, _prepared_refresh(elapsed))
            is True
        )


def test_refresh_delivered_when_adapter_updates_in_place():
    adapter = _adapter_with_status_update()
    assert _should_suppress_lease_wait_refresh(adapter, _prepared_refresh(15)) is False


def test_initial_wait_notice_never_suppressed():
    adapter = _adapter_without_status_update()
    prepared = _prepare_gateway_status_message(
        Platform.WECOM, "lifecycle", INITIAL_WAIT_NOTICE
    )
    assert prepared is not None
    assert _should_suppress_lease_wait_refresh(adapter, prepared) is False


def test_lease_timeout_warning_never_suppressed():
    adapter = _adapter_without_status_update()
    prepared = _prepare_gateway_status_message(
        Platform.WECOM, "lifecycle", LEASE_TIMEOUT_WARNING
    )
    assert prepared is not None
    assert _should_suppress_lease_wait_refresh(adapter, prepared) is False


def test_unrelated_lifecycle_status_never_suppressed():
    adapter = _adapter_without_status_update()
    prepared = _prepare_gateway_status_message(
        Platform.WECOM, "lifecycle", "Resumed session after gateway restart"
    )
    assert prepared is not None
    assert _should_suppress_lease_wait_refresh(adapter, prepared) is False
