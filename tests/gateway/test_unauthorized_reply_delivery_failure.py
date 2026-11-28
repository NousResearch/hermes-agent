"""An unauthorized sender's reply must not vanish when the platform refuses to deliver it.

Adapters report API-level refusals — a bot without the send scope, a chat that forbids it — by
returning ``SendResult(success=False, error=...)`` instead of raising, and the only per-send logging
they do sits in their own ``except`` branches. Ignoring the result leaves the operator nothing to
diagnose with: the stranger sees silence, and ``hermes pairing list`` advertises a code whose DM
never arrived (#155562).
"""

import logging
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.pairing import PairingStore
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource

REFUSAL = "[99991672] Access denied. One of the following scopes is required: [im:message:send_as_bot]"


def _runner(store, adapter):
    runner = object.__new__(GatewayRunner)
    runner._pairing_store_for = lambda _source: store
    runner._delivery_adapter_for = lambda _source: adapter
    runner.config = None
    return runner


def _source(user_id="stranger"):
    return SessionSource(platform=Platform.FEISHU, chat_id="oc_1", chat_type="dm", user_id=user_id)


def _refusing_adapter():
    adapter = AsyncMock()
    adapter.send.return_value = SendResult(success=False, error=REFUSAL, error_kind="forbidden")
    return adapter


def _undelivered_warnings(caplog):
    return [
        record for record in caplog.records
        if record.levelno >= logging.WARNING and "could not deliver" in record.getMessage().lower()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", ["_hm_offer_pairing_code", "_hm_send_unauthorized_decline"])
async def test_a_refused_unauthorized_reply_names_the_platform_error(caplog, handler):
    caplog.set_level(logging.WARNING, logger="gateway.run")
    adapter = _refusing_adapter()

    await getattr(_runner(PairingStore(), adapter), handler)(_source())

    adapter.send.assert_awaited_once()
    [record] = _undelivered_warnings(caplog)
    message = record.getMessage()
    assert "99991672" in message  # the platform's own error text, not a generic "send failed"
    assert "feishu" in message
    assert "stranger" in message  # ... alongside who never got the reply


@pytest.mark.asyncio
async def test_the_rate_limited_reply_has_the_same_delivery_path(caplog, monkeypatch):
    """``generate_code`` answering None means the sender is rate-limited; that reply is sent through
    the same call, so a refusal there left the sender with silence just the same."""
    caplog.set_level(logging.WARNING, logger="gateway.run")
    store = PairingStore()
    monkeypatch.setattr(store, "generate_code", lambda *_args, **_kwargs: None)
    adapter = _refusing_adapter()

    await _runner(store, adapter)._hm_offer_pairing_code(_source())

    adapter.send.assert_awaited_once()
    assert len(_undelivered_warnings(caplog)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", ["_hm_offer_pairing_code", "_hm_send_unauthorized_decline"])
async def test_a_delivered_reply_stays_quiet(caplog, handler):
    caplog.set_level(logging.WARNING, logger="gateway.run")
    adapter = AsyncMock()
    adapter.send.return_value = SendResult(success=True, message_id="om_1")

    await getattr(_runner(PairingStore(), adapter), handler)(_source())

    adapter.send.assert_awaited_once()
    assert not _undelivered_warnings(caplog)
