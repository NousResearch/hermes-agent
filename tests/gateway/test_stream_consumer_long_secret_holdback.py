"""A credential value longer than any fixed scan window still never streams raw (#56039).

AWS STS ``SessionToken`` values are 600-2000 base64 chars, and JSON ``api_key``/``client_secret``
values can be as long. The hold for a labelled value must last from the label until the value
ends, whatever its length, and the end-of-segment redaction must reach back to the label.
"""

import asyncio
import random
from itertools import pairwise
from types import SimpleNamespace

import pytest

from agent.redact import redact_for_egress
from gateway.platforms.base import BasePlatformAdapter
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from gateway.stream_consumer_redact import redact_settled
from tests.waiting import async_wait_until

_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def _value(length: int, seed: int) -> str:
    rnd = random.Random(seed)
    return "".join(rnd.choice(_B64) for _ in range(length))


def _sts(label: str, value: str) -> str:
    return ('Temporary credentials:\n```json\n{\n  "Credentials": {\n'
            f'    "{label}": "{value}",\n    "Expiration": "2026-11-12T12:00:00Z"\n  }}\n}}\n```\nDone.')


def _deltas(text: str, seed: int) -> list[str]:
    rnd, out, i = random.Random(seed), [], 0
    while i < len(text):
        k = rnd.randint(1, 4)
        out.append(text[i:i + k])
        i += k
    return out


def _raw_runs(texts, secret: str, width: int = 8) -> set[str]:
    runs = {secret[i:i + width] for i in range(len(secret) - width + 1)}
    return {r for r in runs for t in texts if r in t}


_CASES = [(label, length) for label in ("Session" + "Token", "api_" + "key") for length in (1000, 2000)]


@pytest.mark.parametrize(("label", "length"), _CASES)
def test_long_labelled_value_streamed_in_small_deltas_never_settles_raw(label, length):
    value = _value(length, seed=length)
    text = _sts(label, value)
    assert not _raw_runs([redact_for_egress(text)], value)  # the non-streamed send masks it
    acc = prefix = ""
    frames = []
    for delta in _deltas(text, seed=length):
        acc, settled = redact_settled(acc + delta, prefix, done=False)
        prefix = acc[:settled]
        frames.append(prefix)
    final, _ = redact_settled(acc, prefix, done=True)
    assert not _raw_runs(frames, value)
    assert not _raw_runs([final], value)
    assert all(later.startswith(earlier) for earlier, later in pairwise(frames))
    assert final.startswith(frames[-1])


def test_segment_end_redacts_a_value_longer_than_the_lookback_from_its_label():
    value = _value(1500, seed=7)
    text = 'Config:\n{"api_' + 'key": "' + value + '", "region": "us-east-1"}\nDone.'
    head = text[:text.index(value) + 1200]  # settled up to deep inside the value
    final, _ = redact_settled(text, head, done=True)
    assert not _raw_runs([final], value)


def _adapter(frames: list):
    cls = type("ChatAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 4000})
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    ids = iter(f"m{i}" for i in range(1000))

    async def send(**kw):
        frames.append(kw["content"])
        return SimpleNamespace(success=True, message_id=next(ids))

    async def edit_message(**kw):
        frames.append(kw["content"])
        return SimpleNamespace(success=True, message_id=kw["message_id"])

    adapter.send, adapter.edit_message = send, edit_message
    return adapter


@pytest.mark.asyncio
async def test_long_session_token_is_never_shown_raw_by_the_consumer():
    value = _value(1000, seed=11)
    text = _sts("Session" + "Token", value)
    frames = []
    consumer = GatewayStreamConsumer(_adapter(frames), "chat", StreamConsumerConfig(
        edit_interval=0.001, buffer_threshold=1, cursor=""))
    drains = []
    drain = consumer._drain_queue
    consumer._drain_queue = lambda: (drains.append(None), drain())[1]
    task = asyncio.create_task(consumer.run())
    for i, delta in enumerate(_deltas(text, seed=11)):
        consumer.on_delta(delta)
        if i % 25 == 0:  # let a tick land while the value is still arriving
            await async_wait_until(lambda: consumer._queue.empty(), timeout=5, what="delta drained")
            seen = len(drains)
            await async_wait_until(lambda: len(drains) > seen, timeout=5, what="tick done")
    consumer.finish()
    await task
    assert frames
    assert not _raw_runs(frames, value)


def test_long_double_quoted_value_holding_an_apostrophe_never_settles_raw():
    # The last quote on the line is the apostrophe, 600 chars past the label: the value's own
    # opening quote must still anchor the label search.
    value = _value(600, seed=5) + "'" + "bravo charlie delta"
    text = 'Config:\n{"pass' + 'word": "' + value + '", "user": "bob"}\nDone.'
    assert not _raw_runs([redact_for_egress(text)], value)
    acc = prefix = ""
    frames = []
    for delta in _deltas(text, seed=5):
        acc, settled = redact_settled(acc + delta, prefix, done=False)
        prefix = acc[:settled]
        frames.append(prefix)
    assert not _raw_runs(frames, value)
