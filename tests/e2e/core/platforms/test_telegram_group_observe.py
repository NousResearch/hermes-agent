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
