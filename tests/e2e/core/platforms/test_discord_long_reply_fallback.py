"""A long Discord reply whose file upload is refused still arrives whole, as paced messages.

A reply past ``MAX_SPLIT_MESSAGES`` goes out as one ``.md`` file (#152012). When Discord refuses the
attachment, the gateway sends the reply as ordinary messages: every part, in order, with no
truncation notice (Teknium's ruling on PR #153476: Discord matches Telegram and never cuts a reply
off), and a 429 in the middle is waited out and retried by discord.py. REAL gateway + adapter
against the Discord stand-in.
"""

from __future__ import annotations

import re
import sys

import pytest

from tests.e2e.core.platforms._contract import merge
from tests.e2e.core.platforms._drv_discord import DiscordDriver
from tests.e2e.core.platforms._helpers import Director, GatewayUnderTest
from tests.fakes.fake_llm_provider import FakeLLMServer, Text
from tests.fakes.platforms._standin import wait_until

pytestmark = [  # as test_discord_contract.py: the general suite (it has discord.py), not the OS-only jobs
    pytest.mark.spawns_gateway_lookalike,
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group gateway harness"),
]


def _reply(chars: int) -> str:
    lines = [" ".join(f"w{i}x{j}" for j in range(9)) for i in range(chars // 60 + 1)]
    return "\n".join(lines)[:chars]


def test_refused_upload_sends_the_whole_reply_through_a_429(tmp_path):
    drv = DiscordDriver()
    drv.start()
    director = Director()
    llm = FakeLLMServer(director)
    llm.start()
    cfg = merge(drv.gateway_config(), {"streaming": {"enabled": False},
                                       "display": {"platforms": {"discord": {"streaming": False}}}})
    gw = GatewayUnderTest(tmp_path / "rig", llm_base_url=llm.base_url, config=cfg, env=drv.gateway_env(),
                          ready=drv.connected)
    st, texts = drv.standin, {"n": 0}

    def fifth_text_send(p):
        if p.get("_files"):
            return False
        texts["n"] += 1
        return texts["n"] == 5

    try:
        gw.start()
        st.fail("create_message", {"message": "Missing Permissions", "code": 50013}, status=403, times=99,
                match=lambda p: bool(p.get("_files")))
        st.fail("create_message", {"message": "You are being rate limited.", "retry_after": 1.0, "global": False},
                status=429, times=1, match=fifth_text_send)
        reply = _reply(40_000)
        director.script("long", Text(reply, chunk_chars=len(reply) + 1))
        chat = drv.dm("long reply please [in:long]").chat_id
        tail = reply.rsplit("\n", 1)[-1]

        def shown():
            return [re.sub(r" \(\d+/\d+\)$", "", v.text) for v in drv.visible(chat) if not v.deleted]

        wait_until(lambda: any(part.endswith(tail) for part in shown()), "the reply's last line", timeout=120,
                   on_timeout=lambda: f"{drv.describe()}\n{gw.tail(4000)}")
        gw.wait_idle()
        parts = shown()
        assert len(parts) > 8  # past the cap: no 8-message cut
        assert re.sub(r"\s+", "", "".join(parts)) == re.sub(r"\s+", "", reply)  # every char, in order, once
        assert not any("truncated" in part.lower() for part in parts)
        assert sum(c.faulted and c.response.get("retry_after") is not None
                   for c in st.calls_of("create_message") if isinstance(c.response, dict)) == 1
    finally:
        gw.stop()
        llm.stop()
        drv.stop()
