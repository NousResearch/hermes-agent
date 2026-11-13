"""A cron report the standalone lane may already have put in the chat is never queued for replay.

The live lane rejects with a retryable "Not connected" (the bot client is rebuilding), then the REAL
registered standalone sender runs. Cron queues the payload for a post-reconnect replay only when
that sender positively reports nothing was sent (``retryable`` on its result dict). A lost answer
(the request left, no reply came back) or a partial multi-chunk send must not queue: the redelivery
sweep would post the report a second time (#153474 review).
"""
import asyncio
import sys
import threading
import time

import httpx
import pytest

import hermes_agent.cron.scheduler_delivery as sd
from hermes_agent.gateway import delivery_ledger as dl
from hermes_agent.gateway.config import GatewayConfig, Platform
from hermes_agent.gateway.delivery import DeliveryRouter, DeliverySendError, DeliveryTarget
from hermes_agent.gateway.platforms.base import SendResult


@pytest.fixture(autouse=True)
def _fresh_ledger(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)
    monkeypatch.setattr(sd, "_maybe_mirror_cron_delivery", lambda *a, **k: None)


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    threading.Thread(target=lp.run_forever, daemon=True).start()
    yield lp
    lp.call_soon_threadsafe(lp.stop)


def _target(lp, platform=Platform.TELEGRAM, name="telegram", extra=None, token="tok"):
    class Transport:
        adapter = type("Adapter", (), {"_owner_profile": "primary"})()
        is_relay = False

        async def send(self, platform, chat_id, content, metadata=None):
            return SendResult(success=False, error="Not connected", retryable=True)

    fields = {n: None for n in sd._TargetDelivery.__dataclass_fields__}
    fields.update(job={"id": "job-1"}, platform=platform, platform_name=name, chat_id="-100", thread_id=None,
                  transport=Transport(), config=GatewayConfig(), loop=lp, target_adapters={}, mirror_text="",
                  origin={}, pconfig=type("P", (), {"token": token, "extra": extra or {}})())
    return sd._TargetDelivery(**fields)


def _queued_rows(t, content):
    te, de = [], []
    assert not sd._deliver_via_live_adapter(t, content, [], target_errors=te, delivery_errors=de, unverified_targets=[])
    sd._deliver_standalone(t, content, [], te, de)
    return dl.sweep_failed_for_runtime(t.platform_name, now=time.time() + 600, profile="primary")


def _telegram_bot(monkeypatch, bot):
    from hermes_agent.tools.send_message_senders import _plugin_standalone_sender
    sender, _ = _plugin_standalone_sender("telegram")  # the module the registry really calls
    mod = sys.modules[sender.__module__]
    monkeypatch.setattr(mod, "_telegram_bot", lambda token, extra=None: bot)
    monkeypatch.setattr(mod, "_telegram_retry_delay", lambda exc, attempt: None)


def _connect_error():
    from telegram.error import NetworkError
    err = NetworkError("httpx.ConnectError: All connection attempts failed")  # PTB's wrapping
    err.__cause__ = httpx.ConnectError("All connection attempts failed")
    return err


def test_rich_send_lost_answer_is_not_queued(monkeypatch, loop):
    """rich_messages on + a table: sendRichMessage times out, so the report may have landed."""
    pytest.importorskip("telegram")
    from telegram.error import TimedOut

    class Bot:
        async def do_api_request(self, method, api_kwargs=None, **kw):
            raise TimedOut("Timed out")

    _telegram_bot(monkeypatch, Bot())
    assert _queued_rows(_target(loop, extra={"rich_messages": True}), "| a | b |\n|---|---|\n| 1 | 2 |") == []


def test_failure_after_a_landed_chunk_is_not_queued(monkeypatch, loop):
    """Chunk 1 lands, chunk 2 fails before leaving: replaying the whole report repeats chunk 1."""
    pytest.importorskip("telegram")
    landed = []

    class Bot:
        async def send_message(self, **kw):
            if landed:
                raise _connect_error()
            landed.append(kw["text"])
            return type("M", (), {"message_id": 1})()

    _telegram_bot(monkeypatch, Bot())
    assert _queued_rows(_target(loop), ("word " * 1200).strip()) == []
    assert len(landed) == 1


def test_telegram_connect_failure_before_any_chunk_is_queued_once(monkeypatch, loop):
    """The case the queue exists for: nothing left, so exactly one row waits for the replay."""
    pytest.importorskip("telegram")

    class Bot:
        async def send_message(self, **kw):
            raise _connect_error()

    _telegram_bot(monkeypatch, Bot())
    assert [r["content"] for r in _queued_rows(_target(loop), "the report")] == ["the report"]


def _qqbot_post(monkeypatch, message_error):
    posted = []

    async def fake_post(self, url, *a, **kw):
        if "getAppAccessToken" in str(url):
            return httpx.Response(200, json={"access_token": "x"}, request=httpx.Request("POST", url))
        posted.append(url)
        raise message_error(httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return posted


def test_qqbot_read_timeout_after_the_post_is_not_queued(monkeypatch, loop):
    posted = _qqbot_post(monkeypatch, lambda req: httpx.ReadTimeout("read timed out", request=req))
    t = _target(loop, platform=Platform("qqbot"), name="qqbot", extra={"app_id": "1"}, token="s")
    assert _queued_rows(t, "the report") == []
    assert len(posted) == 1


def test_qqbot_connect_refused_is_queued(monkeypatch, loop):
    _qqbot_post(monkeypatch, lambda req: httpx.ConnectError("connection refused", request=req))
    t = _target(loop, platform=Platform("qqbot"), name="qqbot", extra={"app_id": "1"}, token="s")
    assert [r["content"] for r in _queued_rows(t, "the report")] == ["the report"]


@pytest.mark.parametrize("result", [
    {"success": False, "error": "Not connected", "retryable": True},
    SendResult(success=False, error="Not connected", retryable=True),
], ids=["dict", "SendResult"])
def test_router_keeps_the_retryable_verdict_of_dict_and_object_results(loop, result):
    """Adapters return a SendResult or a plain dict; the router's raise carries ``retryable`` from both."""
    class Transport:
        adapter = None
        is_relay = False

        async def send(self, platform, chat_id, content, metadata=None):
            return result

    router = DeliveryRouter(GatewayConfig(), {})
    target = DeliveryTarget(platform=Platform.TELEGRAM, chat_id="-100", is_explicit=True)
    coro = router._deliver_to_platform(target, "hi", {}, transport=Transport())
    with pytest.raises(DeliverySendError) as excinfo:
        asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=10)
    assert excinfo.value.send_retryable is True



def test_chunked_send_drops_nothing_sent_once_a_chunk_landed():
    """The shared chunk loop: a sender's ``retryable`` (nothing sent) holds for chunk 1 only; after a
    landed chunk the payload is partly on screen, so the failure must not read as replayable."""
    from hermes_agent.tools.send_message_tool import _send_chunks

    def run(fail_at):
        async def send_one(chunk, is_last):
            if chunk == fail_at:
                return {"error": "connect refused", "retryable": True}
            return {"success": True}
        return asyncio.run(_send_chunks(["one", "two"], send_one))

    assert run("one") == {"error": "connect refused", "retryable": True}
    assert run("two") == {"error": "connect refused", "delivery": "partial"}


def test_retryable_live_failure_after_a_post_is_not_queued(monkeypatch, loop):
    """QQBot's live ``_send_chunk`` marks ANY non-permanent error retryable, a reset after the message
    POST included: "can try again" is not "nothing sent". Even when standalone then provably sends
    nothing (its token request fails), the live copy may be on screen, so no replay."""
    async def token_refused(self, url, *a, **kw):
        return httpx.Response(503, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", token_refused)
    t = _target(loop, platform=Platform("qqbot"), name="qqbot", extra={"app_id": "1"}, token="s")

    class ResetAfterPost:
        adapter = type("Adapter", (), {"_owner_profile": "primary"})()
        is_relay = False

        async def send(self, platform, chat_id, content, metadata=None):
            return SendResult(success=False, error="Server disconnected without sending a response.", retryable=True)

    t.transport = ResetAfterPost()
    assert _queued_rows(t, "the report") == []


def test_discord_refused_standalone_post_is_queued(monkeypatch, loop):
    """Discord's live lane rejects ``send_path_degraded``; its standalone POST gets a 429. Discord
    refused the request, so nothing is on screen and the report waits for the reconnect (as on main)."""
    pytest.importorskip("aiohttp")
    import aiohttp

    class Resp:
        status = 429
        content = None
        charset = "utf-8"

        async def text(self):
            return '{"message": "You are being rate limited."}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class Session:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def post(self, *a, **kw):
            return Resp()

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    t = _target(loop, platform=Platform.DISCORD, name="discord", token="tok")

    class Degraded:
        adapter = type("Adapter", (), {"_owner_profile": "primary"})()
        is_relay = False

        async def send(self, platform, chat_id, content, metadata=None):
            return SendResult(success=False, error="send_path_degraded", retryable=True)

    t.transport = Degraded()
    t.thread_id = "555"  # a thread target: no forum probe
    assert [r["content"] for r in _queued_rows(t, "the report")] == ["the report"]


def test_rich_send_that_never_left_is_queued(monkeypatch, loop):
    """sendRichMessage fails to connect: the request never left, so the rich path reports it as
    nothing sent and the report is queued once."""
    pytest.importorskip("telegram")

    class Bot:
        async def do_api_request(self, method, api_kwargs=None, **kw):
            raise _connect_error()

    _telegram_bot(monkeypatch, Bot())
    rows = _queued_rows(_target(loop, extra={"rich_messages": True}), "| a | b |\n|---|---|\n| 1 | 2 |")
    assert len(rows) == 1


def _server_error_then_connect_error():
    from telegram.error import NetworkError
    raised = []

    async def send_message(**kw):
        raised.append(kw["text"])
        if len(raised) == 1:
            raise NetworkError("Internal Server Error")  # Telegram's own 5xx: may follow a committed send
        raise _connect_error()
    return raised, send_message


@pytest.mark.parametrize("platform", ["telegram", "qqbot"])
def test_a_maybe_delivered_attempt_is_never_erased_by_a_later_connect_failure(monkeypatch, loop, platform):
    """One logical send: an attempt answered with a 5xx (the server may have committed the message)
    is retried, and the retry cannot connect. The final error proves only that the LAST request
    never left; the payload may be on screen from the first, so nothing is queued for a replay."""
    if platform == "telegram":
        pytest.importorskip("telegram")
        attempts, send_message = _server_error_then_connect_error()
        _telegram_bot(monkeypatch, type("Bot", (), {"send_message": staticmethod(send_message)})())
        from hermes_agent.tools.send_message_senders import _plugin_standalone_sender
        mod = sys.modules[_plugin_standalone_sender("telegram")[0].__module__]
        monkeypatch.setattr(mod, "_telegram_retry_delay", lambda exc, attempt: 0)  # the real retry policy, no sleep
        t = _target(loop)
    else:
        attempts = []

        async def fake_post(self, url, *a, **kw):
            req = httpx.Request("POST", url)
            if "getAppAccessToken" in str(url):
                return httpx.Response(200, json={"access_token": "x"}, request=req)
            attempts.append(url)
            if len(attempts) == 1:
                return httpx.Response(500, json={"message": "failed after accepting"}, request=req)
            raise httpx.ConnectError("connection refused", request=req)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        t = _target(loop, platform=Platform("qqbot"), name="qqbot", extra={"app_id": "1"}, token="s")
    assert _queued_rows(t, "the report") == []
    assert len(attempts) >= 2  # the 5xx attempt went out, then the connect failure
    assert t.standalone_in_flight is True
