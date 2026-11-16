"""Telegram channel posts through the REAL gateway + adapter against the Bot API stand-in.

A channel post carries no ``from`` user; it must still run allowed_chats / require_mention instead of
taking the unrestricted DM path.
"""

from __future__ import annotations

import time

import pytest

from tests.e2e.core.platforms import _contract as C
from tests.e2e.core.platforms._drv_telegram import TelegramDriver
from tests.e2e.core.platforms._helpers import Director, GatewayUnderTest
from tests.fakes.fake_llm_provider import FakeLLMServer, Text
from tests.fakes.platforms._standin import wait_until
from tests.fakes.platforms.telegram_standin import BOT_USERNAME

pytestmark = [pytest.mark.spawns_gateway_lookalike, pytest.mark.platforms("posix")]

ALLOWED, EXCLUDED = -1005550001, -1005550002


def _channel_post(drv: TelegramDriver, chat_id: int, text: str, *, mention: bool) -> None:
    chat = {"id": chat_id, "type": "channel", "title": f"Channel {chat_id}"}
    fields = {"text": text}
    if mention:
        handle = f"@{BOT_USERNAME}"
        fields = {"text": f"{handle} {text}", "entities": [{"type": "mention", "offset": 0, "length": len(handle)}]}
    st = drv.standin
    st.chats[str(chat_id)] = chat
    st._push({"update_id": next(st._update_ids), "channel_post": {
        "message_id": next(st._inbound_ids), "date": int(time.time()), "chat": chat, "sender_chat": chat, **fields}})


def test_channel_posts_honor_allowed_chats_and_require_mention(tmp_path) -> None:
    drv = TelegramDriver()
    drv.start()
    director = Director()
    llm = FakeLLMServer(director)
    llm.start()
    cfg = C.merge(drv.gateway_config(), {"platforms": {"telegram": {"extra": {"allowed_chats": [str(ALLOWED)]}}}})
    # Channels authenticate as the channel id; authorize both so only the routing gates decide.
    users = f"{drv.user_id},{ALLOWED},{EXCLUDED}"
    gw = GatewayUnderTest(tmp_path / "rig", llm_base_url=llm.base_url, config=cfg,
                          env={**drv.gateway_env(), "TELEGRAM_ALLOWED_USERS": users}, ready=drv.connected)
    try:
        gw.start()
        for tag in ("excluded", "excluded2", "unmentioned", "addressed"):
            director.script(tag, Text(tag.upper()))
        _channel_post(drv, EXCLUDED, "outside the allowlist [in:excluded]", mention=True)
        _channel_post(drv, ALLOWED, "no mention [in:unmentioned]", mention=False)
        # Gated posts are consumed by the same getUpdates batch; a later addressed post in each chat
        # dispatches only after the earlier ones were routed, so its turn bounds the wait.
        _channel_post(drv, EXCLUDED, "follow-up outside [in:excluded2]", mention=True)
        _channel_post(drv, ALLOWED, "addressed [in:addressed]", mention=True)
        wait_until(lambda: director.turns.get("addressed"), "the addressed channel post reached the model")
        assert not director.turns.get("excluded")
        assert not director.turns.get("excluded2")
        assert not director.turns.get("unmentioned")
    finally:
        gw.stop()
        llm.stop()
        drv.stop()
