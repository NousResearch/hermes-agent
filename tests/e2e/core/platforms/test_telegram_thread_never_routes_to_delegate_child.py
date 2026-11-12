"""A Telegram DM thread whose state still names a delegate subagent's session hands the user's
next message back to the thread's own conversation, never to the subagent (#138210).

REAL gateway + Telegram adapter against the Bot API stand-in. The store is seeded the way a
pre-fix delegation left it: the subagent row carries the parent's routing identity, and in the
other legs the DM-topic binding or the routing index also names it.
"""

from __future__ import annotations

import datetime
import json
import sqlite3

import pytest

from hermes_state import SessionDB
from tests.e2e.core.platforms._drv_telegram import TelegramDriver
from tests.e2e.core.platforms._helpers import Director, GatewayUnderTest
from tests.fakes.fake_llm_provider import FakeLLMServer
from tests.fakes.platforms._standin import wait_until

pytestmark = [pytest.mark.spawns_gateway_lookalike, pytest.mark.platforms("posix")]

THREAD = "54287"
PARENT, CHILD = "20261014_020931_3881b279", "20261014_040222_5d0bcc"


def _seed(db_path, uid: str, key: str, leg: str, hermes_home) -> None:
    db = SessionDB(db_path=db_path)
    try:
        db.create_session(PARENT, "telegram", user_id=uid, session_key=key, chat_id=uid, chat_type="dm",
                          thread_id=THREAD)
        db.append_message(PARENT, "user", "host apps on the cluster")
        db.create_session(CHILD, "subagent", parent_session_id=PARENT, model_config={"_delegate_from": PARENT})
        db.append_message(CHILD, "user", "worker goal")
        db.record_gateway_session_peer(CHILD, source="telegram", user_id=uid, session_key=key, chat_id=uid,
                                       chat_type="dm", thread_id=THREAD)
        if leg == "binding":
            db.enable_telegram_topic_mode(chat_id=uid, user_id=uid)
            db.bind_telegram_topic(chat_id=uid, thread_id=THREAD, user_id=uid, session_key=key, session_id=CHILD)
        if leg == "route":
            now = datetime.datetime.now().isoformat()
            entry = {"session_key": key, "session_id": CHILD, "created_at": now, "updated_at": now,
                     "platform": "telegram", "chat_type": "dm",
                     "origin": {"platform": "telegram", "chat_id": uid, "chat_type": "dm", "user_id": uid,
                                "thread_id": THREAD}}
            db.save_gateway_routing_entry(key, json.dumps(entry))
            (hermes_home / "sessions").mkdir(parents=True, exist_ok=True)
            (hermes_home / "sessions" / "sessions.json").write_text(json.dumps({key: entry}))
    finally:
        db.close()


@pytest.mark.parametrize("leg", ["peer", "binding", "route"])
def test_a_thread_left_naming_a_delegate_child_answers_in_its_own_conversation(tmp_path, leg):
    drv = TelegramDriver()
    drv.start()
    llm = FakeLLMServer(Director())
    llm.start()
    gw = GatewayUnderTest(tmp_path / "rig", llm_base_url=llm.base_url, config=drv.gateway_config(),
                          env=drv.gateway_env(), ready=drv.connected)
    uid = drv.user_id
    gw.hermes_home.mkdir(parents=True, exist_ok=True)
    _seed(gw.db_path, uid, f"agent:main:telegram:dm:{uid}:{THREAD}", leg, gw.hermes_home)
    chat = {"id": int(uid), "type": "private", "first_name": "Tester", "username": f"user{uid}"}
    try:
        gw.start()
        drv.standin._inbound(chat, int(uid), message_thread_id=int(THREAD), is_topic_message=True,
                             text="am I talking to a subagent? [in:q1]")
        wait_until(lambda: any("reply to q1" in v.text for v in drv.visible(uid)), "the reply", on_timeout=gw.tail)
        gw.wait_idle()
        con = sqlite3.connect(f"file:{gw.db_path}?mode=ro", uri=True)
        try:
            owners = [r[0] for r in con.execute("SELECT session_id FROM messages WHERE content LIKE '%[in:q1]%'")]
        finally:
            con.close()
        assert owners == [PARENT]
    finally:
        gw.stop()
        llm.stop()
        drv.stop()
