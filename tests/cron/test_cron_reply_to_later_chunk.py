"""A reply to ANY chunk of a split cron delivery is framed as a past report (#26714).

Only the first chunk of a wrapped delivery over the platform limit carries the
``Cronjob Response:`` header. A reply to a later chunk (where a report's action items and footer
sit) used to reach the model as a bare quote. Each delivered chunk id is tagged in the sent index
the Telegram adapter already keeps, and the inbound pointer looks the replied-to id up there.
"""

import asyncio
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_agent.cron.scheduler import _deliver_result
from hermes_agent.gateway import rich_sent_store
from hermes_agent.gateway.config import Platform, PlatformConfig
from hermes_agent.gateway.platforms.event import MessageEvent
from hermes_agent.gateway.run import GatewayRunner
from hermes_agent.gateway.session import SessionSource
from tests.waiting import wait_until

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


def _deliver(live_result=None, standalone_result=None, release=None):
    loop = MagicMock()
    loop.is_running.return_value = True

    started = threading.Event()

    def fake_run_coro(coro, _loop):
        # The gateway loop runs the send on its own thread; the scheduler only waits on the future.
        # Handed back once the send has STARTED, so a short confirm wait always meets an in-flight
        # send, never the "never dispatched" fallback.
        future = Future()

        def _drive():
            try:
                future.set_result(asyncio.run(coro))
            except BaseException as exc:
                future.set_exception(exc)
        threading.Thread(target=_drive, daemon=True).start()
        assert started.wait(10), "the send never started"
        return future

    router = MagicMock()

    async def _deliver_to_platform(target, text, metadata, transport=None):
        started.set()
        if release is not None:  # still sending after the scheduler stops waiting
            await asyncio.to_thread(release.wait, 10)
        if isinstance(live_result, Exception):
            raise live_result
        return live_result

    router._deliver_to_platform = _deliver_to_platform

    async def _send_to_platform(*_args, **_kwargs):
        return standalone_result

    live = live_result is not None
    with patch("hermes_agent.gateway.config.load_gateway_config", return_value=_gateway_config()), \
         patch("hermes_agent.cron.scheduler.load_config", return_value={"cron": {"wrap_response": True}}), \
         patch("hermes_agent.cron.scheduler_delivery._record_delivery_verification"), \
         patch("hermes_agent.gateway.delivery.DeliveryRouter", return_value=router), \
         patch("hermes_agent.tools.send_message_tool._send_to_platform", _send_to_platform), \
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


def test_chunks_of_a_partial_send_and_a_send_past_the_confirm_wait_are_tagged(monkeypatch):
    """Ids are tagged when the send itself finishes: a partial split send raises with the landed
    chunks, and a slow one is still running when the scheduler stops waiting (60 s)."""
    from hermes_agent.cron import scheduler_delivery as sd
    from hermes_agent.gateway.delivery import PartialDeliveryError

    partial = SimpleNamespace(success=False, message_id=None, error="flood", raw_response={
        "partial_overflow": True, "last_message_id": "202", "continuation_message_ids": ("202",),
        "delivered_message_ids": ("201", "202")})
    _deliver(live_result=PartialDeliveryError("flood", partial))
    assert _reply_to("202", LATER_CHUNK).startswith("[Replying to an earlier scheduled cron delivery")

    monkeypatch.setattr(sd, "_LIVE_SEND_CONFIRM_TIMEOUT_SECS", 2.0)
    release = threading.Event()
    _deliver(live_result=SimpleNamespace(success=True, message_id="301",
                                         raw_response={"message_ids": ["301", "302"]}), release=release)
    assert _reply_to("302", LATER_CHUNK).startswith('[Replying to: "')  # scheduler returned first
    release.set()
    wait_until(lambda: _reply_to("302", LATER_CHUNK).startswith("[Replying to an earlier scheduled"),
               timeout=5, what="the slow send's chunk ids to be tagged")


def test_an_unreadable_sent_index_leaves_ordinary_replies_working(tmp_path, monkeypatch):
    unreadable = tmp_path / "index-is-a-directory"
    unreadable.mkdir()
    monkeypatch.setattr(rich_sent_store, "_store_path", lambda: str(unreadable))
    assert _reply_to("42", "Japan is great.") == '[Replying to: "Japan is great."]\n\nhelp me debug login'


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
    assert result["delivered_message_ids"] == ["500", "501", "502"]


def test_a_partial_standalone_split_send_frames_replies_to_its_landed_chunks(monkeypatch):
    """The REAL scheduler and Telegram standalone sender: chunks 1-2 land (501, 502), chunk 3 fails.
    A reply to 502 quotes a continuation chunk without the header, so it is framed as a past report
    only if the failed send's landed ids were tagged."""
    from tests.tools.test_telegram_send_message_chunk_mdv2 import _install_telegram_mock, _no_proxy
    from plugins.platforms.telegram.standalone import _send_telegram
    from hermes_agent.gateway.inbound_reply_context import reply_pointer

    _no_proxy(monkeypatch)
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=[SimpleNamespace(message_id=501), SimpleNamespace(message_id=502),
                                              RuntimeError("transport refused final chunk")])
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    cfg = MagicMock()
    cfg.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    cfg.get_home_channel = lambda p: None
    job = {"id": "j", "name": "nightly", "deliver": "telegram:-100123",
           "origin": {"platform": "telegram", "chat_id": "-100123"}}

    async def send(platform, pconfig, chat_id, text, **kwargs):
        return await _send_telegram("tok", chat_id, text)

    with patch("hermes_agent.gateway.config.load_gateway_config", return_value=cfg), \
         patch("hermes_agent.cron.scheduler.load_config", return_value={"cron": {"wrap_response": True}}), \
         patch("hermes_agent.tools.send_message_tool._send_to_platform", send), \
         patch("hermes_agent.cron.jobs.update_job"):
        assert _deliver_result(job, "status line ok\n" * 700)
    assert bot.send_message.await_count == 3
    pointer = reply_pointer("status line ok", replied_to=("telegram", "-100123", "502"))
    assert pointer.startswith("[Replying to an earlier scheduled cron delivery"), pointer
