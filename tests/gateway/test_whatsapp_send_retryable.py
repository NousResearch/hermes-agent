"""WhatsApp send failures during a bridge reconnect are retryable only when the bridge says so.

#140571: a Baileys bridge recovering from a routine 428 reconnect answers sends with 503
``{"error": "Not connected to WhatsApp"}``, and that body alone cannot distinguish a
reconnecting bridge from a logged-out or dead one. The bridge now reports ``reconnectPending``
in the 503 body; only that verdict marks the ``SendResult`` retryable, so ``_send_with_retry``
resends after the socket rebuilds while a bridge that will not come back keeps the failure
final. A chunked send that already landed its head never inherits the verdict — a whole-payload
resend would duplicate the visible chunks.
"""

import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult


class _FakeResp:
    def __init__(self, status: int, body):
        self.status = status
        self._body = body

    async def json(self):
        return (
            json.loads(self._body) if isinstance(self._body, str) else dict(self._body)
        )

    async def text(self):
        return self._body if isinstance(self._body, str) else json.dumps(self._body)


class _RespCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


def _make_adapter(responses):
    """A WhatsAppAdapter whose bridge answers each POST with the next of ``responses``.

    Built with ``__new__`` like ``test_whatsapp_connect._make_adapter``: the bridge/session
    state is faked, ``_bridge_unavailable`` reports a usable bridge, and ``_bridge_req`` hands
    out the scripted responses so ``send()`` runs its real chunk loop against them.
    """
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter._running = True
    adapter._http_session = object()
    adapter._reply_prefix = None
    adapter._bridge_unavailable = AsyncMock(return_value=None)
    it = iter(responses)
    adapter._bridge_req = lambda method, path, timeout, **kwargs: _RespCtx(next(it))
    return adapter


def _bridge_503(pending) -> _FakeResp:
    body = {"error": "Not connected to WhatsApp"}
    if pending is not None:
        body["reconnectPending"] = pending
    return _FakeResp(503, json.dumps(body))


@pytest.mark.asyncio
async def test_send_503_with_reconnect_pending_is_retryable():
    adapter = _make_adapter([_bridge_503(pending=True)])

    result = await adapter.send("123456789@s.whatsapp.net", "hello")

    assert result.success is False
    assert result.retryable is True
    assert "Not connected to WhatsApp" in (result.error or "")


@pytest.mark.asyncio
async def test_send_503_without_pending_stays_final():
    # An older bridge (no field) or a logged-out session: the failure must not burn retry budget.
    adapter = _make_adapter([_bridge_503(pending=False), _bridge_503(pending=None)])

    for _ in range(2):
        result = await adapter.send("123456789@s.whatsapp.net", "hello")
        assert result.success is False
        assert result.retryable is False


@pytest.mark.asyncio
async def test_send_client_error_is_not_retryable():
    adapter = _make_adapter(
        [_FakeResp(400, json.dumps({"error": "chatId and message are required"}))]
    )

    result = await adapter.send("123456789@s.whatsapp.net", "hello")

    assert result.success is False
    assert result.retryable is False


@pytest.mark.asyncio
async def test_send_split_failure_midway_is_not_retryable():
    # The reconnect window can strike mid-chunk: the landed head must veto a whole-payload resend.
    adapter = _make_adapter(
        [
            _FakeResp(200, json.dumps({"messageId": "head"})),
            _bridge_503(pending=True),
        ]
    )
    adapter.truncate_message = lambda text, limit: ["chunk one", "chunk two"]

    result = await adapter.send("123456789@s.whatsapp.net", "chunk one chunk two")

    assert result.success is False
    assert result.retryable is False


@pytest.mark.asyncio
async def test_bridge_guard_rejection_stays_final():
    # ``_needs_bridge`` fires before the bridge can be asked: an unavailable managed bridge may
    # have exited for good (logged out), so the guard keeps the failure non-retryable.
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter._running = False  # the guard's own "Not connected", no bridge to consult

    result = await adapter.send("123456789@s.whatsapp.net", "hello")

    assert result.success is False
    assert result.retryable is False


@pytest.mark.asyncio
async def test_send_with_retry_lands_after_reconnect():
    # #140571's scenario: both lanes land inside the ~3 s reconnect window. With the verdict
    # carried, the base retry loop's first backoff outlasts the rebuild and the send lands.
    adapter = _make_adapter(
        [
            _bridge_503(pending=True),
            _FakeResp(200, json.dumps({"messageId": "after-reconnect"})),
        ]
    )

    result = await adapter._send_with_retry(
        "123456789@s.whatsapp.net", "hello", base_delay=0.01
    )

    assert result.success is True
    assert result.message_id == "after-reconnect"


@pytest.mark.asyncio
async def test_send_with_retry_keeps_exhausted_reconnect_final():
    # The bridge never comes back within the retry budget: the typed failure stays visible.
    adapter = _make_adapter([_bridge_503(pending=True)] * 3)

    result = await adapter._send_with_retry(
        "123456789@s.whatsapp.net", "hello", base_delay=0.01
    )

    assert result.success is False
    assert result.retryable is True
