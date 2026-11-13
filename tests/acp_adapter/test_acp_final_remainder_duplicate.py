"""The settled reply must not repeat a body that is already on screen.

A turn can close a text segment between the streamed body and ``message.complete``: a
``message.interim`` (the owner's verify-on-stop candidate) or a ``tool.start``. ``_close_segment``
then empties ``_streamed``, and the settle used to fall through to "the whole reply is new".

Measured case (2026-11-12 23:14): the body streamed in 58 deltas, a ``message.interim`` closed that
segment, and the settle emitted the whole body again — plus the runtime footer the gateway appends
to the final text — under a new ``messageId``. The client painted the reply twice.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import acp
import pytest

from acp_adapter.gateway_reply import GatewayReplyMixin

BODY = "\n\nSí. Es `e2e-stg`, la prueba de punta a punta."
FOOTER = "\n\ndeepseek-v4.1-flash · 22% · ~"
SID, EDITOR, AID = "session-1", "editor-1", "admission-1"


class _Projection(GatewayReplyMixin):
    def __init__(self):
        self._conn = MagicMock(spec=acp.Client)
        self._conn.session_update = AsyncMock()
        self._streamed = {}
        self._segments = {}
        self._cut = {}
        self._message_ids = {}
        self._tool_args = {}
        self._settling = set()
        self._receipted = set()


def _reply_chunks(conn):
    """``(messageId, text)`` for every assistant chunk the editor received, in order."""
    chunks = []
    for call in conn.session_update.await_args_list:
        update = call.kwargs.get("update") if "update" in call.kwargs else call.args[1]
        if getattr(update, "session_update", None) != "agent_message_chunk":
            continue
        chunks.append((update.message_id, update.content.text))
    return chunks


def _shown_per_message(chunks):
    shown = {}
    for message_id, text in chunks:
        shown[message_id] = shown.get(message_id, "") + text
    return shown


async def _settled_turn(close_with):
    agent = _Projection()
    conn = agent._conn
    for part in (BODY[:24], BODY[24:]):
        agent._streamed[AID] = agent._streamed.get(AID, "") + part
        await conn.session_update(session_id=EDITOR, update=agent._chunk(SID, part))
    if close_with is not None:
        agent._close_segment(AID)
        agent._end_message(SID)
    await agent._finish_reply(SID, EDITOR, AID, {"text": BODY + FOOTER, "outcome": "completed"})
    return _shown_per_message(_reply_chunks(conn))


@pytest.mark.parametrize("close_with", ["interim", "tool"])
def test_settle_does_not_repeat_a_body_the_editor_already_has(close_with):
    shown = asyncio.run(_settled_turn(close_with))

    assert sum(text.count(BODY.strip()) for text in shown.values()) == 1, shown
    assert any(FOOTER.strip() in text for text in shown.values()), shown


def test_settle_without_a_closed_segment_still_shows_the_footer():
    shown = asyncio.run(_settled_turn(None))

    assert len(shown) == 1, shown
    assert sum(text.count(BODY.strip()) for text in shown.values()) == 1, shown
    assert sum(text.count(FOOTER.strip()) for text in shown.values()) == 1, shown
