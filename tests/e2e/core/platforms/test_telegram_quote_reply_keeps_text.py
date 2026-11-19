"""A quote-reply to an ordinary (non-rich) bot message keeps the quoted text even when Telegram's
``reply_to_message`` echo carries no body (#147287): the adapter resolves it from its own sent-message
index. REAL gateway + adapter against the Bot API stand-in, for a plain send and a streamed final."""

from __future__ import annotations

import time

import pytest

from tests.e2e.core.platforms import _contract as C
from tests.e2e.core.platforms._drv_telegram import TelegramDriver
from tests.e2e.core.platforms._suite import rig_fixtures

pytestmark = [
    pytest.mark.spawns_gateway_lookalike,
    pytest.mark.platforms("posix"),  # POSIX process-group gateway harness
]

rig, rig_stream = rig_fixtures(TelegramDriver)


def _quote_reply_reaches_model(rig: C.Rig, tag: str, user: str, **answer_kw) -> str:
    """Bot answers ``tag``; the user quote-replies with a body-less echo. Returns the model's last user turn."""
    token, aid = f"q-{tag}", f"A-q-{tag}"
    rig.director.script(token, C.answer(aid, f"The deploy window is Tuesday {tag}", **answer_kw))
    first = rig.drv.dm(f"when is the deploy? [in:{token}]", user_id=user)
    C.wait_reply(rig, first.chat_id, aid)
    rig.gw.wait_idle()
    shown = [v for v in rig.drv.visible(first.chat_id) if not v.deleted and C.foot(aid) in C.norm(v.text)]
    assert len(shown) == 1, rig.ctx()

    st, uid = rig.drv.standin, int(user)
    chat = st.chats[first.chat_id]
    echo = {"message_id": int(shown[0].message_id), "date": int(time.time()), "chat": chat, "from": st._bot_user()}
    qtok, qaid = f"qr-{tag}", f"A-qr-{tag}"
    rig.director.script(qtok, C.answer(qaid, "noted"))
    st._inbound(chat, uid, text=f"is that UTC? [in:{qtok}]", reply_to_message=echo)
    C.wait_reply(rig, first.chat_id, qaid)
    users = [m for m in rig.director.requests[qtok][0]["messages"] if m.get("role") == "user"]
    return rig.director._text(users[-1])


def test_quote_reply_to_plain_send_keeps_quoted_text(rig) -> None:
    seen = _quote_reply_reaches_model(rig, "plain", rig.drv.user_id)
    assert "[Replying to" in seen and "The deploy window is Tuesday plain" in seen, seen


def test_quote_reply_to_streamed_final_keeps_quoted_text(rig_stream) -> None:
    seen = _quote_reply_reaches_model(
        rig_stream, "stream", rig_stream.fresh_user(), chunk_chars=8, delay_per_chunk=0.03)
    assert "[Replying to" in seen and "The deploy window is Tuesday stream" in seen, seen
