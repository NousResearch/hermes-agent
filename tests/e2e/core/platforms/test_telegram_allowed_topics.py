"""Telegram ``allowed_topics`` through the REAL gateway + adapter against the Bot API stand-in.

An allowlisted forum topic is answered; a topic outside the list is dropped, and the drop is named
once in the gateway log instead of leaving the operator with a silent bot.
"""

from __future__ import annotations

import pytest

from tests.e2e.core.platforms import _contract as C
from tests.e2e.core.platforms._drv_telegram import TelegramDriver
from tests.e2e.core.platforms._helpers import Director, GatewayUnderTest
from tests.fakes.fake_llm_provider import FakeLLMServer
from tests.fakes.platforms._standin import wait_until

pytestmark = [
    pytest.mark.spawns_gateway_lookalike,
    pytest.mark.platforms("posix"),
]

FORUM = {"id": -1003946573900, "type": "supergroup", "title": "Forum", "is_forum": True}


def test_allowed_topic_is_answered_and_a_dropped_topic_is_logged_once(tmp_path) -> None:
    drv = TelegramDriver()
    drv.start()
    director = Director()
    llm = FakeLLMServer(director)
    llm.start()
    cfg = C.merge(drv.gateway_config(), {"platforms": {"telegram": {"extra": {
        "require_mention": False, "allowed_topics": ["846"]}}}})
    gw = GatewayUnderTest(tmp_path / "rig", llm_base_url=llm.base_url, config=cfg,
                          env=drv.gateway_env(), ready=drv.connected)
    log_path = gw.hermes_home / "logs" / "gateway.log"
    drop = f"Ignoring messages in chat {FORUM['id']} topic 1: not in allowed_topics"
    # A reply-anchor thread id is not a topic message, so it rides the chat's admission lane behind
    # both General messages: its own drop line proves they were both gated.
    barrier = f"Ignoring messages in chat {FORUM['id']} topic 5: not in allowed_topics"
    try:
        gw.start()
        st, uid = drv.standin, int(drv.user_id)
        st._inbound(FORUM, uid, text="general [in:gen]")
        st._inbound(FORUM, uid, text="general again [in:gen]")
        st._inbound(FORUM, uid, message_thread_id=5, text="anchored [in:gen]")
        st._inbound(FORUM, uid, message_thread_id=846, is_topic_message=True, text="hello [in:t846]")
        wait_until(lambda: director.turns.get("t846"), "allowlisted topic 846 reached the model")
        wait_until(lambda: any(str(c.params.get("message_thread_id")) == "846"
                               for c in st.calls_of("sendMessage")), "reply delivered into topic 846")
        wait_until(lambda: barrier in log_path.read_text(errors="replace"), "the allowed_topics drops were logged",
                   on_timeout=gw.tail)
        assert not director.turns.get("gen")
        assert log_path.read_text(errors="replace").count(drop) == 1
    finally:
        gw.stop()
        llm.stop()
        drv.stop()
