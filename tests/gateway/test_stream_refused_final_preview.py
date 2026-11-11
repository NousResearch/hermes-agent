"""A streamed reply whose final edit is refused leaves no stuck preview (group chats, edit transport).

The adapter refuses every edit that carries the answer's ending (a message that can no longer be
edited) and, like Telegram, defers a preview edit while the chat's shared send+edit slot is busy.
The consumer then sends the unseen tail as a new message. The preview it leaves behind must not
keep the streaming cursor, and a preview that holds only a fragment of the first word must not be
followed by a continuation that starts mid-word.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

CURSOR = " ▉"


class _SlotAdapter:
    """Edit transport with Telegram's per-chat send+edit slot: a preview edit (``finalize=False``)
    fired while the slot is busy is skipped and reported as ``raw_response={"skipped": True}``;
    a final edit is never skipped. Edits carrying ``refuse`` are rejected."""

    MAX_MESSAGE_LENGTH = 4096
    SUPPORTS_MESSAGE_EDITING = True

    def __init__(self, refuse: str, slot_secs: float = 5.0):
        self.refuse = refuse
        self.slot_secs = slot_secs
        self.slot_until = 0.0
        self.messages: dict[str, str] = {}
        self.deleted: list[str] = []
        self._next = 100

    def _arm(self) -> None:
        self.slot_until = asyncio.get_running_loop().time() + self.slot_secs

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self._arm()
        self._next += 1
        self.messages[str(self._next)] = content
        return SimpleNamespace(success=True, message_id=str(self._next), raw_response=None)

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
        if not finalize and asyncio.get_running_loop().time() < self.slot_until:
            return SimpleNamespace(success=True, message_id=message_id, raw_response={"skipped": True})
        self._arm()
        if self.refuse in content:
            return SimpleNamespace(success=False, message_id=None, raw_response=None,
                                   error="Bad Request: message can't be edited")
        self.messages[message_id] = content
        return SimpleNamespace(success=True, message_id=message_id, raw_response=None)

    async def delete_message(self, chat_id, message_id):
        self.deleted.append(message_id)
        self.messages.pop(message_id, None)
        return True


async def _stream(adapter: _SlotAdapter, pieces: list[str]) -> None:
    consumer = GatewayStreamConsumer(
        adapter, "-100123",
        config=StreamConsumerConfig(cursor=CURSOR, edit_interval=0.01, buffer_threshold=1))
    task = asyncio.create_task(consumer.run())
    for piece in pieces:
        consumer.on_delta(piece)
        await asyncio.sleep(0.03)  # health: allow HX013 -- stream pacing: lets the consumer's edit ticks fire between deltas, the timing this bug depends on
    consumer.finish()
    await asyncio.wait_for(task, 10)


def _pieces(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


@pytest.mark.asyncio
async def test_refused_final_edit_leaves_no_cursor_on_the_preview():
    body = " ".join(f"q{i:03d}" for i in range(40))
    answer = f"<<head>> {body} <<end>>"
    adapter = _SlotAdapter(refuse="<<end>>")

    await _stream(adapter, _pieces(answer, 12))

    shown = list(adapter.messages.values())
    assert not any(CURSOR.strip() in text for text in shown), shown
    joined = " ".join(shown)
    assert joined.count("<<head>>") == 1 and joined.count("<<end>>") == 1, shown


@pytest.mark.asyncio
async def test_preview_holding_part_of_the_first_word_is_replaced_by_the_whole_answer():
    first_word = "<<" + "x" * 60 + ">>"
    answer = f"{first_word} q000 q001 q002 <<end>>"
    adapter = _SlotAdapter(refuse="<<end>>")

    await _stream(adapter, _pieces(answer, 12))

    shown = list(adapter.messages.values())
    # One message holds the whole answer; nothing starts mid-word or keeps a fragment beside it.
    assert shown == [answer], shown


@pytest.mark.asyncio
async def test_a_first_word_fragment_whose_send_timed_out_is_continued_not_repeated():
    first_word = "<<" + "x" * 60 + ">>"
    answer = f"{first_word} q000 q001 q002 <<end>>"
    adapter = _SlotAdapter(refuse="<<end>>")
    send = adapter.send

    async def first_send_times_out(chat_id, content, reply_to=None, metadata=None):
        adapter.send = send
        await send(chat_id, content, reply_to, metadata)  # it landed, but no id came back
        return SimpleNamespace(success=False, message_id=None, retryable=False, error="Timed out",
                               raw_response={"delivery_uncertain": True})

    adapter.send = first_send_times_out
    await _stream(adapter, _pieces(answer, 12))

    shown = list(adapter.messages.values())
    # The fragment cannot be deleted (its id is unknown), so the whole answer must not follow it.
    joined = "".join(shown)
    assert joined.count("<<x") == 1 and joined.count("<<end>>") == 1, shown
