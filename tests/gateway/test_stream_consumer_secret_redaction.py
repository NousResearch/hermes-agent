"""Streamed chat egress is secret-redacted like the non-streamed final send (#56039).

Every message the consumer puts on a platform — previews, sealed overflow heads, the split
tail — must carry no credential, including one the overflow split would otherwise cut in two
(each half alone matches no redaction pattern).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import BasePlatformAdapter
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.waiting import async_wait_until

_KEY = "sk-proj-" + "Q7mZ2xW9rT4kL8vB3nH6pD1sF5gJ0cY"


def _adapter():
    cls = type("ChatAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 1000})
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    ids = iter(f"m{i}" for i in range(100))
    adapter.send = AsyncMock(side_effect=lambda **kw: SimpleNamespace(success=True, message_id=next(ids)))
    adapter.edit_message = AsyncMock(
        side_effect=lambda **kw: SimpleNamespace(success=True, message_id=kw["message_id"]))
    return adapter


def _delivered(adapter) -> list[str]:
    return [c.kwargs["content"] for c in adapter.send.call_args_list + adapter.edit_message.call_args_list]


def _leaked_fragments(texts: list[str]) -> list[str]:
    """Any 8-char run of the key body (masks show at most its last 4 chars)."""
    body = _KEY[len("sk-proj-"):]
    runs = {body[i:i + 8] for i in range(len(body) - 7)}
    return [r for r in runs for t in texts if r in t]


async def _stream(adapter, *deltas: str) -> None:
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(
        edit_interval=0.01, buffer_threshold=5, cursor=""))
    task = asyncio.create_task(consumer.run())
    for delta in deltas:
        # One delta per consumer tick: wait until this one has reached the wire.
        before = len(_delivered(adapter))
        consumer.on_delta(delta)
        await async_wait_until(lambda: consumer._queue.empty() and len(_delivered(adapter)) > before,
                               timeout=5, what="delta delivered")
    consumer.finish()
    await task


@pytest.mark.asyncio
async def test_short_streamed_reply_is_redacted():
    adapter = _adapter()
    await _stream(adapter, f"Your key is {_KEY} - keep it safe.")
    texts = _delivered(adapter)
    assert texts and all(_KEY not in t for t in texts)
    assert not _leaked_fragments(texts)


@pytest.mark.asyncio
async def test_first_send_overflow_split_never_cuts_a_secret_in_two():
    # No whitespace before the key: the adapter's splitter has to hard-cut, and the cut lands
    # inside the key.
    adapter = _adapter()
    await _stream(adapter, "/" * 880 + _KEY + "\nTail line.")
    texts = _delivered(adapter)
    assert len(adapter.send.call_args_list) >= 2  # really split
    assert not _leaked_fragments(texts)


@pytest.mark.asyncio
async def test_sealed_overflow_head_never_cuts_a_secret_in_two():
    adapter = _adapter()
    # An open preview first, so overflow goes through the seal-the-existing-message path.
    await _stream(adapter, "Starting the report.", "/" * 860 + _KEY + "\nTail line.")
    texts = _delivered(adapter)
    assert len(adapter.send.call_args_list) >= 2  # really sealed + continued
    assert not _leaked_fragments(texts)


@pytest.mark.asyncio
async def test_secret_still_arriving_at_the_split_is_redacted_once_complete():
    # The overflow tick sees only the first half of the key at the end of the buffer; the
    # second half arrives on a later tick.
    adapter = _adapter()
    await _stream(adapter, "Starting the report.", " word" * 174 + " " + _KEY[:20],
                  _KEY[20:] + " done.")
    texts = _delivered(adapter)
    assert len(adapter.send.call_args_list) >= 2  # really sealed + continued
    assert not _leaked_fragments(texts)


_PW = "Hunter2" + "xQ9vLm4RtZ8k"


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    'Here it is:\n{"pass' + 'word": "' + _PW + '", "user": "bob"}\nDone.',
    "Run: mysql --pass" + "word " + _PW + " -h db\nDone.",
], ids=["json-field", "cli-flag"])
async def test_streamed_final_still_reconciles_when_a_second_redaction_pass_changes_it(reply):
    # redact_for_egress is not idempotent here (``Hunter...tZ8k`` -> ``***``). The gateway
    # reconciles the raw final_response against what was streamed; a mismatch makes it edit the
    # message with that raw text (run_turn.py stale-finalize path) or send the reply twice.
    adapter = _adapter()
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(cursor=""))
    consumer.on_delta(reply)
    consumer.finish(reply)
    await consumer.run()
    assert consumer.delivered_final_matches(reply) is True
    assert all(_PW[6:-4] not in t for t in _delivered(adapter))


@pytest.mark.asyncio
async def test_failed_reply_file_resumes_without_cutting_into_a_masked_secret():
    # Past the message cap the reply goes out as a file; the upload fails and it resumes as messages
    # from where the chat left off. A mask in the live tail shortens it, so a resume offset counted
    # from it lands inside the raw final and sends a key fragment no pattern matches.
    from pathlib import Path

    password = "Hunter2" + "xQ9vLm4RtZ8k"
    para = "lorem ipsum dolor sit amet " * 6 + "\n"
    reply = para * 9 + '{"pass' + 'word": "' + password + '", "user": "bob"}\n' + para * 21 + "Done."
    cls = type("CapAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 600, "MAX_SPLIT_MESSAGES": 4})
    cls.__abstractmethods__ = frozenset()

    async def send_reply_document(self, chat_id, path, **kw):
        assert password[2:-2] not in Path(path).read_text()
        return SimpleNamespace(success=False, message_id=None)

    cls.send_reply_document = send_reply_document
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    shown = []
    ids = iter(f"m{i}" for i in range(1000))

    async def send(**kw):
        shown.append(kw["content"])
        return SimpleNamespace(success=True, message_id=next(ids))

    async def edit_message(**kw):
        shown.append(kw["content"])
        return SimpleNamespace(success=True, message_id=kw["message_id"])

    adapter.send, adapter.edit_message = send, edit_message
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(
        edit_interval=0.001, buffer_threshold=1, cursor=""))
    drains = []
    drain = consumer._drain_queue
    consumer._drain_queue = lambda: (drains.append(None), drain())[1]
    task = asyncio.create_task(consumer.run())
    for i in range(0, len(reply), 40):
        consumer.on_delta(reply[i:i + 40])
        await async_wait_until(lambda: consumer._queue.empty(), timeout=5, what="delta drained")
        seen = len(drains)
        await async_wait_until(lambda: len(drains) > seen, timeout=5, what="tick done")
    consumer.finish(reply)
    await asyncio.wait_for(task, 30)
    assert not any(password[2:-2] in text for text in shown)
    assert "Done." in shown[-1]
