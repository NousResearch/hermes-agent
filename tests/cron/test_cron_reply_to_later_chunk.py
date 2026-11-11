"""A reply to ANY chunk of a split cron delivery is framed as a past report (#26714).

Only the first chunk of a wrapped delivery over the platform limit carries the
``Cronjob Response:`` header. A reply to a later chunk (where a report's action items and footer
sit) used to reach the model as a bare quote. Each delivered chunk id is tagged in the sent index
the Telegram adapter already keeps, and the inbound pointer looks the replied-to id up there.
"""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler import _deliver_result
from gateway import rich_sent_store
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource

CHAT_ID = "-1001234567890"
LATER_CHUNK = "status line ok\nTODO: deploy prod now (2/2)"


@pytest.fixture(autouse=True)
def _isolated_sent_index(tmp_path, monkeypatch):
    monkeypatch.setattr(rich_sent_store, "_store_path", lambda: str(tmp_path / "sent.json"))


def _job():
    return {"id": "j1", "name": "nightly-ops", "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": CHAT_ID}}


def _gateway_config():
    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    return config


def _reply_to(message_id, text):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT_ID, chat_type="group", user_name="A")
    event = MessageEvent(text="help me debug login", source=source,
                         reply_to_message_id=message_id, reply_to_text=text)
    return GatewayRunner._prepend_inbound_reply_context(event, source, "help me debug login")


def _deliver(live_result=None, standalone_result=None):
    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    router = MagicMock()

    async def _deliver_to_platform(target, text, metadata, transport=None):
        return live_result

    router._deliver_to_platform = _deliver_to_platform

    async def _send_to_platform(*_args, **_kwargs):
        return standalone_result

    live = live_result is not None
    with patch("gateway.config.load_gateway_config", return_value=_gateway_config()), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": True}}), \
         patch("cron.scheduler_delivery._record_delivery_verification"), \
         patch("gateway.delivery.DeliveryRouter", return_value=router), \
         patch("tools.send_message_tool._send_to_platform", _send_to_platform), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        adapters = {Platform.TELEGRAM: MagicMock()} if live else None
        return _deliver_result(_job(), "x" * 5000 + "\nTODO: deploy prod now", adapters=adapters,
                               loop=loop if live else None)


@pytest.mark.parametrize("lane", ["live", "standalone"])
def test_reply_to_a_later_chunk_of_a_cron_delivery_is_framed(lane):
    ids = {"message_ids": ["101", "102"]}
    if lane == "live":
        error = _deliver(live_result=SimpleNamespace(success=True, message_id="101", raw_response=ids))
    else:
        error = _deliver(standalone_result={"success": True, "message_id": "102", "raw_response": ids})
    assert error is None

    framed = _reply_to("102", LATER_CHUNK)
    assert framed.startswith("[Replying to an earlier scheduled cron delivery")
    assert 'Cronjob Response: nightly-ops\n…\nstatus line ok' in framed
    # An ordinary message in the same chat is untouched.
    assert _reply_to("103", "Japan is great.").startswith('[Replying to: "Japan is great."]')


def test_telegram_standalone_sender_reports_every_chunk_id(monkeypatch):
    """The out-of-gateway sender exposes all chunk ids, not only the last one."""
    from tests.tools.test_telegram_send_message_chunk_mdv2 import _install_telegram_mock, _no_proxy
    from plugins.platforms.telegram.standalone import _send_telegram

    _no_proxy(monkeypatch)
    sent = iter(range(500, 510))
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=lambda **_kw: SimpleNamespace(message_id=next(sent)))
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    result = asyncio.run(_send_telegram("tok", "123", "line of text\n" * 700))
    assert result["success"] is True, result
    assert result["raw_response"]["message_ids"] == ["500", "501", "502"]
