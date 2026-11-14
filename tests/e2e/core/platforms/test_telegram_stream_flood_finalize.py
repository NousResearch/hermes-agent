"""A streamed group reply whose final edit and fallback send are flood-refused ends as ONE complete
message: the frozen preview is finished in place once the RetryAfter window passes, never posted a
second time beside it (#96078, #102402, #53449). REAL gateway + adapter against the Bot API stand-in.
"""

from __future__ import annotations

import threading
import time

import pytest

from tests.e2e.core.platforms._contract import answer, foot, head, merge, norm
from tests.e2e.core.platforms._drv_telegram import TelegramDriver
from tests.e2e.core.platforms._helpers import Director, GatewayUnderTest
from tests.fakes.fake_llm_provider import FakeLLMServer
from tests.fakes.platforms._standin import wait_until

pytestmark = [pytest.mark.spawns_gateway_lookalike, pytest.mark.platforms("posix")]
_FLOOD_SECS = 8  # past the adapter's 5 s inline cap: the edit and the fallback send fail closed


@pytest.fixture
def rig(tmp_path):
    drv = TelegramDriver()
    drv.start()
    director = Director()
    llm = FakeLLMServer(director)
    llm.start()
    cfg = merge(drv.gateway_config(), {"streaming": {"enabled": True, "edit_interval": 0.3, "transport": "edit"},
                                       "display": {"platforms": {"telegram": {"streaming": True}}}})
    gw = GatewayUnderTest(tmp_path / "rig", llm_base_url=llm.base_url, config=cfg, env=drv.gateway_env(),
                          ready=drv.connected)
    try:
        gw.start()
        yield drv, director, gw
    finally:
        gw.stop()
        llm.stop()
        drv.stop()


def _flood_after_first_preview(drv, chat: str) -> dict:
    """Once the first preview lands, refuse every edit and send in ``chat`` with a 429 for a while."""
    st, window = drv.standin, {"until": 0.0}
    body = {"ok": False, "error_code": 429, "description": f"Too Many Requests: retry after {_FLOOD_SECS}",
            "parameters": {"retry_after": _FLOOD_SECS}}

    def flooded(p):
        return str(p.get("chat_id")) == chat and time.monotonic() < window["until"]

    def arm():
        wait_until(lambda: any(not c.faulted and str(c.params.get("chat_id")) == chat
                               for c in st.calls_of("sendMessage")), "first preview", timeout=60)
        window["until"] = time.monotonic() + _FLOOD_SECS
        for method in ("editMessageText", "sendMessage"):
            st.fail(method, body, status=429, times=10_000, match=flooded)

    threading.Thread(target=arm, daemon=True).start()
    return window


def test_flood_refused_final_is_finished_in_place_once(rig):
    drv, director, gw = rig
    aid = "A-flood"
    body = " ".join(f"w{i:04d}" for i in range(250))
    director.script("fl", answer(aid, body, chunk_chars=40, delay_per_chunk=0.02))
    window = _flood_after_first_preview(drv, drv.group_chat_id)
    drv.group("long answer please [in:fl]", mention=True)

    def visible():
        return [v for v in drv.visible(drv.group_chat_id) if not v.deleted]

    wait_until(lambda: any(foot(aid) in norm(v.text) for v in visible()), "the complete reply", timeout=90,
               on_timeout=lambda: f"{drv.describe()}\n{gw.tail(4000)}")
    gw.wait_idle()
    # A same-chat barrier turn: everything the flooded turn still meant to send is out before it.
    director.script("flb", answer("B-flood", "barrier"))
    drv.group("barrier [in:flb]", mention=True)
    wait_until(lambda: any(foot("B-flood") in norm(v.text) for v in visible()), "barrier answered", timeout=60)
    shown = [v for v in visible() if foot("B-flood") not in norm(v.text)]
    assert any(c.faulted for c in drv.standin.calls_of("editMessageText")), "the flood never fired"
    assert window["until"] > 0
    assert sum(head(aid) in norm(v.text) for v in shown) == 1, f"reply shown twice: {drv.describe()}"
    complete = [v for v in shown if head(aid) in norm(v.text) and foot(aid) in norm(v.text)]
    assert len(complete) == 1 and "▉" not in complete[0].text, drv.describe()
    assert director.turns.get("fl") == 1
