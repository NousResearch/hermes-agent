"""Telegram observe mode through the REAL gateway + adapter against the Bot API stand-in.

A non-forum observed group shares ONE conversation: a slash command must act on that shared session
(not a per-user one keyed by the sender), and an addressed turn's tools must still see who sent it.
"""

from __future__ import annotations

import json
import re
import sqlite3

import pytest

from tests.e2e.core.platforms import _contract as C
from tests.e2e.core.platforms._drv_telegram import TelegramDriver
from tests.e2e.core.platforms._helpers import Director, GatewayUnderTest
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall
from tests.fakes.platforms._standin import wait_until
from tests.fakes.platforms.telegram_standin import BOT_USERNAME

pytestmark = [
    pytest.mark.spawns_gateway_lookalike,
    pytest.mark.platforms("posix"),
]


def _session_keys(db) -> set[str]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        return {row[0] for row in conn.execute("SELECT DISTINCT session_key FROM gateway_routing")}
    finally:
        conn.close()


def _await_turn(director: Director, tag: str) -> None:
    wait_until(lambda: director.turns.get(tag), f"turn {tag} reached the model")


def test_observed_group_command_resets_shared_session_and_tools_see_the_sender(tmp_path) -> None:
    drv = TelegramDriver()
    drv.start()
    director = Director()
    llm = FakeLLMServer(director)
    llm.start()
    gid = drv.group_chat_id
    cfg = C.merge(drv.gateway_config(), {"platforms": {"telegram": {"extra": {
        "observe_unmentioned_group_messages": True, "group_allowed_chats": [gid]}}}})
    gw = GatewayUnderTest(tmp_path / "rig", llm_base_url=llm.base_url, config=cfg,
                          env={**drv.gateway_env(), "TELEGRAM_GROUP_ALLOWED_USERS": drv.user_id}, ready=drv.connected)
    try:
        gw.start()
        director.script("t1", Text("ONE"))
        director.script("t2", ToolCall("terminal", {"command": "echo ACTOR=$HERMES_SESSION_ACTOR_ID ROUTE=$HERMES_SESSION_USER_ID."}), Text("TWO"))
        drv.group("hello [in:t1]", mention=True)
        _await_turn(director, "t1")
        shared = f"agent:main:telegram:group:{gid}"
        assert _session_keys(gw.db_path) == {shared}

        command = f"/new@{BOT_USERNAME}"
        drv.standin._inbound({"id": int(gid), "type": "supergroup", "title": "Standin Group"}, int(drv.user_id),
                             text=command, entities=[{"type": "bot_command", "offset": 0, "length": len(command)}])
        wait_until(lambda: any("session" in str((c.params or {}).get("text") or "").lower()
                               for c in drv.standin.calls_of("sendMessage")), "the /new reply was sent",
                   on_timeout=lambda: repr([(c.params or {}).get("text") for c in drv.standin.calls_of("sendMessage")]))

        drv.group("again [in:t2]", mention=True)
        _await_turn(director, "t2")
        wait_until(lambda: len(director.requests.get("t2") or []) >= 2, "the tool result reached the model")
        requests = director.requests["t2"]
        # /new reset the SHARED conversation: no per-user session, and the next turn starts fresh.
        assert _session_keys(gw.db_path) == {shared}
        assert not any("[in:t1]" in json.dumps(m, ensure_ascii=False) for m in requests[0].get("messages") or [])
        # The addressed turn's tool sees the real sender even though the shared source is sender-less.
        tool_out = json.dumps(requests[-1].get("messages") or []).replace("ACTOR=$", "").replace("ROUTE=$", "")
        assert (re.search(r"ACTOR=(\w*)", tool_out) or [None, None])[1] == drv.user_id
        # ...while the route tools rebuild deliveries from stays the shared, sender-less one.
        assert "ROUTE=." in tool_out
    finally:
        gw.stop()
        llm.stop()
        drv.stop()


_PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d494441"
                     "5478da63fccfc0500f000485018084a98c210000000049454e44ae426082")


def test_observed_group_photo_points_the_next_addressed_turn_at_the_cached_file(tmp_path) -> None:
    """Observed context stays text-only, but the agent learns where the photo is and how to look at it."""
    drv = TelegramDriver()
    drv.start()
    director = Director()
    llm = FakeLLMServer(director)
    llm.start()
    gid = drv.group_chat_id
    cfg = C.merge(drv.gateway_config(), {"platforms": {"telegram": {"extra": {
        "observe_unmentioned_group_messages": True, "group_allowed_chats": [gid]}}}})
    gw = GatewayUnderTest(tmp_path / "rig", llm_base_url=llm.base_url, config=cfg,
                          env={**drv.gateway_env(), "TELEGRAM_GROUP_ALLOWED_USERS": drv.user_id}, ready=drv.connected)
    try:
        gw.start()
        drv.standin.files["ph-1"], drv.standin.file_paths["ph-1"] = _PNG, "photos/desk.png"
        drv.standin._inbound({"id": int(gid), "type": "supergroup", "title": "Standin Group"}, int(drv.user_id),
                             photo=[{"file_id": "ph-1", "file_unique_id": "u-ph-1", "width": 1, "height": 1,
                                     "file_size": len(_PNG)}], caption="desk sheet")
        wait_until(lambda: "Cached observed group image" in gw.tail(), "the unmentioned photo was observed")
        drv.group("what was in that image? [in:p1]", mention=True)
        _await_turn(director, "p1")
        user = [m for m in director.requests["p1"][0]["messages"] if m.get("role") == "user"][-1]
        assert isinstance(user["content"], str), "observed media must not ride inline"
        cached = re.search(r"saved at: (\S+\.png)\] \(not attached inline; to look at it use vision_analyze with "
                           r"this path as image_url\)", user["content"])
        assert cached, user["content"]
        assert open(cached[1], "rb").read() == _PNG
    finally:
        gw.stop()
        llm.stop()
        drv.stop()
