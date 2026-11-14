"""Streamed egress never emits text past the settled boundary (#56039).

A secret still arriving — a key whose tail is in a later delta, ``Bearer`` before its token, a
PEM block before its END — must not be sealed into a split head, cut by a splitter, or shown in a
preview/draft frame, where a later re-mask cannot reach it. Assertions are on what the chat ends
up showing per message (sends + edits replayed), and on every frame the transport received.
"""

import asyncio
from itertools import pairwise
from types import SimpleNamespace

import pytest

from gateway.platforms.base import BasePlatformAdapter
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.waiting import async_wait_until

_BODY = "Q7mZ2xW9rT4kL8vB3nH6pD1sF5gJ0cY"
_KEY = "sk-proj-" + _BODY
_TOKEN = "Zq8vN3mK1pX7wL4tR9yB2cF6hJ0dS5gAxx"
_AUTH = "Authorization: " + "Bearer "
_PEM_LINES = ("MIIEpAIBAAKCAQEA1x2y3z4abcdefghijklmnopqrstuvwxyz0123456789ABCD",
              "QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm0123456789ab",
              "ZXCVBNMasdfghjklQWERTYUIOP0987654321zyxwvutsrqponmlkjihgfedcbaXY")
_PEM_BEGIN, _PEM_END = "-----BEGIN RSA PRIVATE KEY-----", "-----END RSA PRIVATE KEY-----"


def _adapter(**attrs):
    cls = type("ChatAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 1000})
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    adapter.ops = []
    ids = iter(f"m{i}" for i in range(100))

    async def send(**kw):
        adapter.ops.append(("send", next(ids), kw["content"]))
        return SimpleNamespace(success=True, message_id=adapter.ops[-1][1])

    async def edit_message(**kw):
        adapter.ops.append(("edit", kw["message_id"], kw["content"]))
        return SimpleNamespace(success=True, message_id=kw["message_id"])

    adapter.send, adapter.edit_message = send, edit_message
    for name, value in attrs.items():
        setattr(adapter, name, value)
    return adapter


def _on_screen(adapter) -> dict[str, str]:
    shown = {}
    for _op, message_id, content in adapter.ops:
        shown[message_id] = content
    return shown


async def _stream(adapter, *deltas: str, **cfg) -> None:
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(
        edit_interval=0.01, buffer_threshold=5, cursor="", **cfg))
    drains = []
    drain = consumer._drain_queue
    consumer._drain_queue = lambda: (drains.append(None), drain())[1]
    task = asyncio.create_task(consumer.run())
    for delta in deltas:
        # One delta per tick: the next drain after the one that took this delta starts only
        # once that tick is over (a held delta reaches no wire, so the wire is no signal).
        consumer.on_delta(delta)
        await async_wait_until(lambda: consumer._queue.empty(), timeout=5, what="delta drained")
        seen = len(drains)
        await async_wait_until(lambda: len(drains) > seen, timeout=5, what="tick done")
    consumer.finish()
    await task


def _raw_runs(texts, secret: str, width: int = 8) -> list[str]:
    runs = {secret[i:i + width] for i in range(len(secret) - width + 1)}
    return sorted({r for r in runs for t in texts if r in t})


@pytest.mark.asyncio
@pytest.mark.parametrize("pad", [880, 885, 890, 893])
async def test_first_send_split_does_not_hard_cut_a_key_still_arriving(pad):
    adapter = _adapter()
    await _stream(adapter, "/" * pad + _KEY[:22], _KEY[22:] + " done")
    # A small pad lets the masked reply fit one message; the larger ones split.
    assert not _raw_runs(_on_screen(adapter).values(), _BODY)


@pytest.mark.asyncio
@pytest.mark.parametrize("deltas", [
    ("Starting the report.", " word" * 170 + " " + _AUTH, _TOKEN + " done"),  # seal path
    (" word" * 176 + " " + _AUTH, _TOKEN + " done"),                           # first-send path
    ("/" * 888 + _AUTH + _TOKEN[:15], _TOKEN[15:] + " done"),                 # hard cut in the token
], ids=["seal", "first-send", "first-send-hard-cut"])
async def test_bearer_token_after_its_scheme_word_is_never_split_off(deltas):
    adapter = _adapter()
    await _stream(adapter, *deltas)
    shown = _on_screen(adapter)
    assert len(shown) >= 2
    assert not _raw_runs(shown.values(), _TOKEN)


@pytest.mark.asyncio
@pytest.mark.parametrize("deltas", [
    ("Starting the report.", " word" * 150 + "\n" + _PEM_BEGIN + "\n" + _PEM_LINES[0] + "\n"
     + _PEM_LINES[1] + "\n", _PEM_LINES[2] + "\n" + _PEM_END + "\nDone."),
    (" word" * 175 + "\n" + _PEM_BEGIN + "\n" + _PEM_LINES[0] + "\n", _PEM_LINES[1] + "\n"
     + _PEM_LINES[2] + "\n" + _PEM_END + "\nDone."),
], ids=["seal", "first-send"])
async def test_pem_block_open_at_the_split_is_never_sealed_line_by_line(deltas):
    adapter = _adapter()
    await _stream(adapter, *deltas)
    shown = _on_screen(adapter)
    assert len(shown) >= 2
    for line in _PEM_LINES:
        assert not _raw_runs(shown.values(), line, width=16), line


@pytest.mark.asyncio
async def test_preview_frames_never_show_a_partial_or_sliding_key_window():
    adapter = _adapter()
    await _stream(adapter, "Your key is ", *[_KEY[i:i + 2] for i in range(0, len(_KEY), 2)],
                  " - keep it safe.")
    frames = [content for _op, _id, content in adapter.ops]
    assert frames
    assert not any(_KEY[:10] in f for f in frames)  # no raw prefix below the pattern floor
    # A mask shows the key's last 4 chars; frames of a growing key would slide that window.
    assert _raw_runs(frames, _BODY, width=4) == [_BODY[-4:]]


@pytest.mark.asyncio
async def test_stream_is_the_message_drafts_stay_prefix_stable_while_a_key_streams():
    frames = []

    async def send_draft(*, chat_id, draft_id, content, metadata=None):
        frames.append(content)
        return SimpleNamespace(success=True)

    adapter = _adapter(draft_stream_is_message=True, send_draft=send_draft,
                       supports_draft_streaming=lambda chat_type=None, metadata=None, **kw: True)
    await _stream(adapter, "Your key is ", *[_KEY[i:i + 3] for i in range(0, len(_KEY), 3)],
                  " - keep it safe.", transport="auto", chat_type="dm")
    assert len(frames) >= 2
    # The connector appends a prefix-extending frame and re-appends any other one whole.
    assert all(later.startswith(earlier) for earlier, later in pairwise(frames)), frames
    finals = [content for _op, _id, content in adapter.ops]
    assert not _raw_runs([*frames, *finals], _BODY, width=5)


@pytest.mark.asyncio
async def test_reply_and_finish_drained_in_one_tick_is_sent_redacted_whole():
    # No interim tick redacts the buffer first: the finalize frame is the first and only redaction.
    adapter = _adapter()
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(cursor=""))
    consumer.on_delta(f"Your key is {_KEY} - keep it safe.")
    consumer.finish()
    await consumer.run()
    shown = _on_screen(adapter)
    assert list(shown.values()) == [f"Your key is sk-pro...{_KEY[-4:]} - keep it safe."]


def _draft_adapter(frames: list):
    async def send_draft(*, chat_id, draft_id, content, metadata=None):
        frames.append(content)
        return SimpleNamespace(success=True)

    return _adapter(draft_stream_is_message=True, send_draft=send_draft,
                    supports_draft_streaming=lambda chat_type=None, metadata=None, **kw: True)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    "See https://github.com/NousResearch/hermes-agent/pull/123 for details",
    'Config: {"api_key": 1, "x": 2} ok',
    "Use https://bob:" + _TOKEN + "@example.com/v1 to log in",
], ids=["url", "json-key", "url-userinfo"])
async def test_draft_frames_stay_prefix_stable_while_a_url_streams_char_by_char(reply):
    # A half-typed URL can look like the start of a key or of user:pass@host; the settled
    # boundary must never move back over text already sent, or the connector re-appends it.
    from agent.redact import redact_for_egress

    frames = []
    adapter = _draft_adapter(frames)
    await _stream(adapter, *reply, transport="auto", chat_type="dm")
    assert len(frames) >= 2
    assert all(later.startswith(earlier) for earlier, later in pairwise(frames)), frames
    assert not _raw_runs(frames, _TOKEN)
    assert [content for _op, _id, content in adapter.ops] == [redact_for_egress(reply)]


@pytest.mark.asyncio
async def test_a_certificate_header_does_not_stall_the_preview():
    # Only private-key blocks are masked whole; a certificate header is ordinary text.
    adapter = _adapter()
    head = "Pin this:\n-----BEGIN CERTIFICATE-----\nMIIBszCCAVmgAwIBAgIU\n"
    await _stream(adapter, head, "more text " * 5, "and the end.")
    previews = [content for _op, _id, content in adapter.ops[:-1]]
    assert any("more text more text" in p for p in previews), previews


@pytest.mark.asyncio
async def test_pem_header_typed_char_by_char_holds_the_block():
    # Each header char is its own tick: a hold that releases on a partial "-----BEGIN" would
    # settle the header before the hold could see it, and the body would stream raw.
    adapter = _adapter()
    block = _PEM_BEGIN + "\n" + "\n".join(_PEM_LINES) + "\n" + _PEM_END
    await _stream(adapter, "Here: ", *block[:40], block[40:], " done.")
    for line in _PEM_LINES:
        assert not _raw_runs([content for _op, _id, content in adapter.ops], line, width=16), line


# A Bearer token whose body holds a credential keyword (pw, key, sig, auth...) looks like a
# ``NAME: value`` label of its own while it arrives.
_KW_TOKENS = ("abcdef" + "PW" + "ghijklmnopqrstuvwxyz012345", "Zx9" + "sig" + "Q2w8e7r6t5y4u3i2o1p0aSdFgHjK",
              "abcdefghijklmn" + "KEY" + "opqrstuvwxyz01234")


def _random_deltas(text: str, seed: int) -> list[str]:
    import random

    rnd, deltas, i = random.Random(seed), [], 0
    while i < len(text):
        k = rnd.choice((1, 1, 2, 3, 5, 7, 15))
        deltas.append(text[i:i + k])
        i += k
    return deltas


@pytest.mark.asyncio
@pytest.mark.parametrize(("token", "split", "draft"), [
    (_KW_TOKENS[0], "char", False), (_KW_TOKENS[0], "char", True),
    (_KW_TOKENS[1], "random", False), (_KW_TOKENS[2], "random", True),
], ids=["pw-char-edit", "pw-char-stream-is-message", "sig-random-edit", "key-random-stream-is-message"])
async def test_bearer_token_holding_a_keyword_is_never_shown_raw(token, split, draft):
    from agent.redact import redact_for_egress

    reply = "Use " + _AUTH + token + " now."
    assert token[:8] not in redact_for_egress(reply)  # the non-streamed send masks it
    frames = []
    adapter = _draft_adapter(frames) if draft else _adapter()
    deltas = list(reply) if split == "char" else _random_deltas(reply, seed=3)
    await _stream(adapter, *deltas, **({"transport": "auto", "chat_type": "dm"} if draft else {}))
    shown = [*frames, *(content for _op, _id, content in adapter.ops)]
    assert not _raw_runs(shown, token)
    assert all(later.startswith(earlier) for earlier, later in pairwise(frames)), frames


_FIRST, _SECOND = "Hx9" + "qLm4RtZ8kVb2", "Wp7" + "yNc3JsD6fGh1"


@pytest.mark.asyncio
@pytest.mark.parametrize("deltas", [
    list('Run: mysql --pass' + 'word "' + _FIRST + "\n" + _SECOND + '" -h db\nDone.'),
    ['Run: mysql --pass' + 'word "' + _FIRST + "\n", "ab\\", "c" + _SECOND + '" -h db\nDone.'],
    list('docker login -p "' + _FIRST + "\n" + _SECOND + '" registry.io\nDone.'),
], ids=["flag", "delta-ends-in-backslash", "docker-login-p"])
async def test_a_quoted_flag_value_spanning_lines_is_held_to_its_closing_quote(deltas):
    # The CLI redactor masks a quoted flag value across a newline; the open-value hold must too.
    adapter = _adapter()
    await _stream(adapter, *deltas)
    shown = [content for _op, _id, content in adapter.ops]
    assert not _raw_runs(shown, _FIRST) and not _raw_runs(shown, _SECOND)


@pytest.mark.asyncio
async def test_a_zhipu_key_id_is_not_shown_while_the_key_arrives():
    # ``<32 hex>.<secret>``: no continuation probe completes the hex id, so it needs its own hold.
    key = "43c5fa52f8b2b19b8f89a0f8a8e9b2d1" + "." + "Q7mZ2xW9rT4kL8vB"
    adapter = _adapter()
    await _stream(adapter, "Your zhipu key is ", *key, " - keep it safe.")
    shown = [content for _op, _id, content in adapter.ops]
    assert not _raw_runs(shown, key[:32])


@pytest.mark.asyncio
async def test_a_double_quoted_value_holding_an_apostrophe_is_held_to_its_closing_quote():
    # Only a double quote closes it: the apostrophe plus a later space must not release the hold.
    value = "alpha'" + "bravo charlie delta"
    adapter = _adapter()
    await _stream(adapter, *('Config: {"pass' + 'word": "' + value + '", "user": "bob"} done'))
    assert not _raw_runs([content for _op, _id, content in adapter.ops], value)
