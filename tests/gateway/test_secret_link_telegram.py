"""Telegram side of secret links: ``/secret`` sends one HTTPS link (preview off) in the same chat,
refuses without a public HTTPS URL or past the per-chat cap, and the confirmation reaches the
same chat through the runner's delivery adapter. Also the cleanup watcher and shutdown clear."""

import asyncio
import re
import threading
from unittest.mock import AsyncMock

import pytest

import hermes_yaml as yaml
from gateway import secret_link
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.secret_link_store import MAX_OPEN_LINKS_PER_CHAT
from hermes_cli.commands import resolve_command
from tests.gateway.secret_link_helpers import (
    BASE, SENTINEL, clock_fixture, env_text, get_form, hermes_root_fixture, restarts_fixture, secret_client,
    store_fixture, submit)
from tests.waiting import async_wait_until

pytestmark = pytest.mark.asyncio
_LINK_RE = re.compile(re.escape(BASE) + r"/secret-link/([A-Za-z0-9_-]+)")


class _Adapter:
    platform = Platform.TELEGRAM

    def __init__(self):
        self.send = AsyncMock(return_value=SendResult(success=True, message_id="1"))


def _runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")})
    adapter = _Adapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    return runner, adapter


def _event(text="/secret STRIPE_SECRET_KEY", platform=Platform.TELEGRAM, chat_id="4242"):
    from gateway.platforms.event import MessageEvent
    source = SessionSource(platform=platform, chat_id=chat_id, chat_type="dm", user_id="7")
    return MessageEvent(text=text, source=source)


def _sent(adapter):
    return [(c.args[0], c.args[1], c.kwargs.get("metadata") or {}) for c in adapter.send.await_args_list]


async def test_secret_command_sends_one_https_link_in_the_same_chat(hermes_root, store):
    runner, adapter = _runner()
    result = await runner._handle_secret_command(_event())
    assert result is None
    [(chat_id, text, metadata)] = _sent(adapter)
    assert chat_id == "4242"
    token = _LINK_RE.search(text).group(1)
    assert len(token) == 43
    assert "STRIPE_SECRET_KEY" in text and "UTC" in text and "10 minutes" in text
    assert metadata.get("disable_link_preview") is True
    assert len(store._records) == 1 and token not in repr(vars(store._records[0]))


async def test_link_flow_end_to_end_confirms_in_the_same_chat(hermes_root, store, restarts):
    runner, adapter = _runner()
    await runner._handle_secret_command(_event())
    token = _LINK_RE.search(_sent(adapter)[0][1]).group(1)
    async with secret_client() as client:
        _r, html, nonce = await get_form(client, token)
        assert 'value="STRIPE_SECRET_KEY"' in html
        resp, _ = await submit(client, token, nonce, profiles=("alpha",))
    assert resp.status == 200
    await async_wait_until(lambda: adapter.send.await_count == 2)
    chat_id, text, _meta = _sent(adapter)[1]
    assert (chat_id, text) == ("4242", "Saved STRIPE_SECRET_KEY.")
    assert all(SENTINEL not in t for _c, t, _m in _sent(adapter))


async def test_an_explicit_secret_save_restarts_the_owner_profile(hermes_root, store, restarts):
    restarts.running = {"default", "alpha"}
    runner, adapter = _runner()  # owner profile: default
    await runner._handle_secret_command(_event())
    token = _LINK_RE.search(_sent(adapter)[0][1]).group(1)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, _ = await submit(client, token, nonce, profiles=("default", "alpha"))
    assert resp.status == 200
    await asyncio.gather(*list(secret_link._FOLLOWUPS))
    assert restarts == ["alpha", "default"]


@pytest.mark.parametrize("base", [None, "http://hermes.example.com", "ftp://x", "https://"])
async def test_no_https_public_url_sends_the_not_configured_notice(hermes_root, store, base):
    cfg = {"secret_links": {"public_base_url": base}} if base else {}
    (hermes_root / "config.yaml").write_text(yaml.safe_dump(cfg))
    runner, adapter = _runner()
    await runner._handle_secret_command(_event())
    assert [t for _c, t, _m in _sent(adapter)] == [secret_link.NOT_CONFIGURED_MESSAGE]
    assert store._records == []


async def test_a_sixth_open_link_in_one_chat_is_refused_and_others_stay_valid(hermes_root, store):
    runner, adapter = _runner()
    for _ in range(MAX_OPEN_LINKS_PER_CHAT):
        await runner._handle_secret_command(_event())
    await runner._handle_secret_command(_event())
    texts = [t for _c, t, _m in _sent(adapter)]
    assert texts[-1] == secret_link.TOO_MANY_LINKS_MESSAGE
    tokens = [_LINK_RE.search(t).group(1) for t in texts[:-1]]
    assert len(set(tokens)) == MAX_OPEN_LINKS_PER_CHAT
    assert all(store.lookup(t) is not None for t in tokens)


async def test_other_platforms_get_a_refusal_and_no_link(hermes_root, store):
    runner, adapter = _runner()
    result = await runner._handle_secret_command(_event(platform=Platform.DISCORD))
    assert result == secret_link.TELEGRAM_ONLY_MESSAGE
    adapter.send.assert_not_awaited()
    assert store._records == []


async def test_an_invalid_suggested_key_is_refused(hermes_root, store):
    runner, adapter = _runner()
    assert await runner._handle_secret_command(_event("/secret LD_PRELOAD")) == "Invalid key name."
    adapter.send.assert_not_awaited()


async def test_secret_is_a_registered_idle_gateway_command():
    runner, _adapter = _runner()
    assert resolve_command("secret").name == "secret"
    assert runner._gateway_idle_command_handlers()["secret"] == runner._handle_secret_command


async def test_telegram_adapter_honours_the_per_message_preview_flag():
    from plugins.platforms.telegram.adapter import TelegramAdapter
    adapter = object.__new__(TelegramAdapter)
    adapter._disable_link_previews = False
    assert adapter._link_preview_kwargs() == {}
    off = adapter._link_preview_kwargs({"disable_link_preview": True})
    assert set(off) in ({"disable_web_page_preview"}, {"link_preview_options"})
    assert adapter._rich_payload_base("1", "x", {"disable_link_preview": True})["link_preview_options"] == {
        "is_disabled": True}
    assert "link_preview_options" not in adapter._rich_payload_base("1", "x", None)


async def test_cleanup_watcher_sweeps_and_shutdown_clears(hermes_root, store, monkeypatch):
    monkeypatch.setattr(secret_link, "CLEANUP_INTERVAL_SECONDS", 0.01)
    runner, _adapter = _runner()

    async def _drop(_text):
        return None
    _t, used = store.create(chat_key="c", owner_home=None, suggested_key="", notify=_drop)
    store.claim(used)
    _t, expired = store.create(chat_key="c", owner_home=None, suggested_key="", notify=_drop)
    expired.expires_at = 0
    done = []
    store.create(chat_key="c", owner_home=None, suggested_key="", notify=_drop, on_done=done.append)
    task = asyncio.create_task(runner._secret_link_cleanup_watcher())
    await async_wait_until(lambda: len(store._records) == 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store._records == [] and done == [False]


async def test_cleanup_watcher_runs_with_the_gateway(hermes_root):
    from gateway.run_startup import GatewayStartupMixin
    watchers = GatewayStartupMixin._PRE_RECONNECT_WATCHERS + GatewayStartupMixin._POST_RECONNECT_WATCHERS
    assert "_secret_link_cleanup_watcher" in watchers


async def test_link_message_takes_the_plain_send_path_that_stores_no_copy():
    """The rich-send path records sent text on disk (rich_sent_store); the link must not use it."""
    from plugins.platforms.telegram.adapter import TelegramAdapter
    adapter = object.__new__(TelegramAdapter)
    text = secret_link.link_message(f"{BASE}/secret-link/{'a_b-' * 10}abc", "12:00 UTC", "STRIPE_SECRET_KEY")
    assert adapter._needs_rich_rendering(text) is False
