"""Telegram ingress shapes through the REAL gateway + adapter against the Bot API stand-in.

A shared contact card (no ``text``) must reach the model, and a photo still in the adapter's
burst buffer when ``/stop`` arrives must not surface as a new turn after the stop.
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from tests.e2e.core.platforms._drv_telegram import TelegramDriver
from tests.e2e.core.platforms._helpers import Director, GatewayUnderTest
from tests.fakes.fake_llm_provider import FakeLLMServer
from tests.fakes.platforms._standin import wait_until

pytestmark = [pytest.mark.spawns_gateway_lookalike, pytest.mark.platforms("posix")]

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def test_contact_card_reaches_model_and_stop_drops_buffered_photo(tmp_path) -> None:
    drv = TelegramDriver()
    drv.start()
    st = drv.standin
    st.files["ph-1"], st.file_paths["ph-1"] = _PNG, "photos/p1.png"
    director = Director()
    llm = FakeLLMServer(director)
    llm.start()
    gw = GatewayUnderTest(tmp_path / "rig", llm_base_url=llm.base_url, config=drv.gateway_config(),
                          env=drv.gateway_env(), ready=drv.connected)
    uid = int(drv.user_id)
    chat = {"id": uid, "type": "private", "first_name": "Tester", "username": f"user{uid}"}

    def bodies_with(needle: str) -> list[str]:
        return [tok for tok, bs in director.requests.items() for b in bs if needle in json.dumps(b.get("messages", []))]
    try:
        gw.start()
        drv.dm("warm [in:warm]")
        wait_until(lambda: director.turns.get("warm"), "warm turn")

        st._inbound(chat, uid, contact={"phone_number": "+15550001234", "first_name": "Ada", "last_name": "Lovelace"})
        wait_until(lambda: bodies_with("+15550001234"), "the contact card reached the model")
        assert "Ada Lovelace" in json.dumps(director.requests[bodies_with("+15550001234")[0]])

        st._inbound(chat, uid, photo=[{"file_id": "ph-1", "file_unique_id": "u-ph-1", "width": 1, "height": 1,
                                       "file_size": len(_PNG)}], caption="old photo caption")
        time.sleep(0.3)  # health: allow HX013 -- /stop must land inside the 0.8 s photo-burst window
        st._inbound(chat, uid, text="/stop", entities=[{"type": "bot_command", "offset": 0, "length": 5}])
        drv.dm("fresh question [in:fresh]")
        wait_until(lambda: director.turns.get("fresh"), "fresh turn")
        time.sleep(3.0)  # health: allow HX013 -- absence proof: a leaked photo flushes within 0.8 s
        assert bodies_with("old photo caption") == []
    finally:
        gw.stop()
        drv.stop()
