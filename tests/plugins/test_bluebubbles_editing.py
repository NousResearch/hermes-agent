import pytest

from gateway.config import PlatformConfig
from plugins.platforms.bluebubbles.adapter import MAX_TEXT_LENGTH

def _make_adapter(monkeypatch, **extra):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    from plugins.platforms.bluebubbles.adapter import BlueBubblesAdapter

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "server_url": "http://localhost:1234",
            "password": "secret",
            **extra,
        },
    )
    return BlueBubblesAdapter(cfg)


class TestBlueBubblesEditOverflow:
    """Oversized edits split across the original bubble + follow-up sends.

    ``GatewayStreamConsumer`` only adopts the returned last-visible
    ``message_id`` when ``continuation_message_ids`` is populated
    (``gateway/stream_consumer.py``), so an overflowing edit must report every
    continuation id or later edits keep targeting the original bubble and
    duplicate the tail.
    """

    @staticmethod
    def _ready(adapter):
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        adapter.client = object()

    @pytest.mark.asyncio
    async def test_edit_overflow_reports_continuation_ids_and_last_id(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        self._ready(adapter)

        async def fake_api_post(path, payload):
            return {"data": {"guid": "edited-guid"}}

        adapter._api_post = fake_api_post

        sent = []

        async def fake_send(chat_id, chunk, **kwargs):
            from gateway.platforms.base import SendResult

            sent.append(chunk)
            return SendResult(success=True, message_id=f"tail-{len(sent)}")

        adapter.send = fake_send

        content = "x" * (MAX_TEXT_LENGTH * 3)
        result = await adapter.edit_message("chat", "orig-guid", content)

        assert result.success is True
        # message_id must be the LAST visible bubble so later edits target it.
        assert result.message_id == f"tail-{len(sent)}"
        # Every continuation id is reported, in send order.
        assert result.continuation_message_ids == tuple(
            f"tail-{i + 1}" for i in range(len(sent))
        )
        assert len(sent) >= 1

    @pytest.mark.asyncio
    async def test_edit_overflow_tail_failure_reports_partial_delivery(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        self._ready(adapter)

        async def fake_api_post(path, payload):
            return {"data": {"guid": "edited-guid"}}

        adapter._api_post = fake_api_post

        calls = []

        async def failing_send(chat_id, chunk, **kwargs):
            from gateway.platforms.base import SendResult

            calls.append(chunk)
            if len(calls) == 1:
                return SendResult(success=True, message_id="tail-1")
            return SendResult(success=False, error="imessage send failed")

        adapter.send = failing_send

        content = "x" * (MAX_TEXT_LENGTH * 3)
        result = await adapter.edit_message("chat", "orig-guid", content)

        # A dropped tail must NOT be reported as a delivered response.
        assert result.success is False
        assert result.error == "overflow_continuation_failed"
        assert result.continuation_message_ids == ("tail-1",)
        assert result.message_id == "tail-1"
        raw = result.raw_response or {}
        assert raw.get("partial_overflow") is True
        assert raw.get("delivered_chunks") == 2
        assert raw.get("last_message_id") == "tail-1"
        # ``delivered_prefix`` must be a true prefix of the formatted text:
        # the consumer gates tail recovery on ``text.startswith(prefix)``
        # (gateway/stream_consumer.py). BlueBubbles' truncate_message omits
        # the "(1/3)" pagination suffixes Telegram has to strip, so joining
        # the delivered chunks is enough.
        delivered_prefix = raw.get("delivered_prefix")
        assert delivered_prefix
        assert adapter.format_message(content).startswith(delivered_prefix)
