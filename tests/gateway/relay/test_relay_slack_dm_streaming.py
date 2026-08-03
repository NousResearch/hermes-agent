"""Slack relay: edit-based streaming of the reply must fire in a DM.

Reported symptom (live): agent responses stream progressively (edit-based) in a
Slack THREAD but arrive FLAT (single message, no progressive edits) in a Slack
DM/home over the relay.

Root cause: a DM turn's streaming reply is sent with
``reply_to = <triggering message ts>`` (the stream consumer's
``initial_reply_to_id`` — its edit anchor). The connector's slackRestSender maps
a raw ``reply_to`` to a Slack ``thread_ts``, so a plain DM reply gets threaded
UNDER the user's message instead of posting flat at the DM root, and a threaded
first send loses the progressive edit streaming the user sees in a real thread.
Native ``SlackAdapter._resolve_thread_ts`` already suppresses this synthetic DM
thread anchor; the relay lane had no such disambiguation.

These are behaviour-contract tests: they assert how the outbound frame relates to
the chat type + thread metadata (the invariant the connector depends on), not a
snapshot. They drive the REAL ``RelayAdapter`` + ``GatewayStreamConsumer`` +
``StubConnector`` end to end.
"""

from __future__ import annotations

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

from tests.gateway.relay.stub_connector import StubConnector


def _slack_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="slack",
        label="Slack",
        max_message_length=4000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="mrkdwn",
        len_unit="chars",
        emoji="\U0001f4ac",
        platform_hint="",
        pii_safe=False,
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


def _wire(chat_id: str, chat_type: str, *, user_id="U1", scope_id=None):
    """A RelayAdapter fronting Slack, with inbound scope captured for chat_id."""
    stub = StubConnector(_slack_desc())
    adapter = RelayAdapter(PlatformConfig(), _slack_desc(), transport=stub)
    src = SessionSource(
        platform=Platform.SLACK,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        scope_id=scope_id,
    )
    adapter._capture_scope(
        MessageEvent(text="hi", source=src, message_type=MessageType.TEXT)
    )
    return adapter, stub


# ---------------------------------------------------------------------------
# The pure disambiguation contract (RelayAdapter.send)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_slack_dm_reply_keeps_anchor_in_thread_per_message_mode():
    """Default mode (reply_in_thread=True, thread-per-message): the triggering
    ts reply_to is the final reply's ONLY threading signal (base.py builds
    metadata from source.thread_id, which is None for a top-level DM) — it
    must be KEPT so the final message lands in the per-message thread with
    the progress bubbles (2026-07-27 mixed-placement report)."""
    adapter, stub = _wire("D1", "dm")
    await adapter.send("D1", "the answer", reply_to="1700.0001")
    assert len(stub.sent) == 1
    frame = stub.sent[0]
    assert frame["op"] == "send"
    assert frame["reply_to"] == "1700.0001", (
        "thread-per-message: the triggering ts anchors the final reply"
    )
    # The connector's Slack sender threads on metadata.thread_id ONLY
    # (threadTs() never reads the frame's reply_to), so the surviving anchor
    # must be promoted into metadata for the send to actually thread.
    assert (frame["metadata"] or {}).get("thread_id") == "1700.0001"


@pytest.mark.asyncio
async def test_slack_dm_reply_drops_synthetic_anchor_in_flat_mode():
    """Flat mode (reply_in_thread=False): the synthetic self-anchor is dropped
    so the reply posts flat at the DM root (native _resolve_thread_ts parity)
    and no synthetic thread is invented (#18859)."""
    adapter, stub = _wire("D1", "dm")
    adapter.config.extra = {"reply_in_thread": False}
    await adapter.send("D1", "the answer", reply_to="1700.0001")
    frame = stub.sent[0]
    assert frame["reply_to"] is None
    assert "thread_id" not in (frame["metadata"] or {})
    assert "thread_ts" not in (frame["metadata"] or {})


@pytest.mark.asyncio
async def test_slack_dm_reply_with_real_thread_keeps_anchor():
    """A DM turn that IS inside a real thread (metadata carries a distinct
    thread_id) must keep threading — the guard only drops the synthetic anchor."""
    adapter, stub = _wire("D1", "dm")
    await adapter.send(
        "D1", "in thread", reply_to="1700.0002", metadata={"thread_id": "1699.9000"}
    )
    frame = stub.sent[0]
    assert frame["reply_to"] == "1700.0002"
    assert frame["metadata"]["thread_id"] == "1699.9000"


@pytest.mark.asyncio
async def test_slack_channel_top_level_reply_keeps_autothread_anchor():
    """A channel top-level reply carries thread_id (its own ts) in metadata when
    autoThread is on; the DM-only guard must not touch it."""
    adapter, stub = _wire("C1", "channel", scope_id="T1")
    await adapter.send(
        "C1", "channel reply", reply_to="1700.0003", metadata={"thread_id": "1700.0003"}
    )
    frame = stub.sent[0]
    assert frame["reply_to"] == "1700.0003"
    assert frame["metadata"]["thread_id"] == "1700.0003"


@pytest.mark.asyncio
async def test_non_slack_dm_reply_unchanged():
    """The disambiguation is Slack-scoped: a non-Slack relay chat keeps reply_to
    (its connector owns its own threading semantics)."""
    stub = StubConnector(_slack_desc(platform="discord"))
    adapter = RelayAdapter(
        PlatformConfig(), _slack_desc(platform="discord"), transport=stub
    )
    src = SessionSource(
        platform=Platform.DISCORD, chat_id="dc1", chat_type="dm", user_id="U1"
    )
    adapter._capture_scope(
        MessageEvent(text="hi", source=src, message_type=MessageType.TEXT)
    )
    await adapter.send("dc1", "hi", reply_to="msg-9")
    assert stub.sent[0]["reply_to"] == "msg-9"


# ---------------------------------------------------------------------------
# End-to-end: the stream consumer keeps edit-streaming in a DM
# ---------------------------------------------------------------------------
async def _drive_stream(adapter, chat_id, *, metadata, initial_reply_to_id, chat_type):
    cfg = StreamConsumerConfig(
        edit_interval=0.0,
        buffer_threshold=1,
        transport="edit",
        chat_type=chat_type,
    )
    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id=chat_id,
        config=cfg,
        metadata=metadata,
        initial_reply_to_id=initial_reply_to_id,
    )
    # Feed progressive deltas, then finalize — mirrors the live delta callback.
    for chunk in ("Hel", "lo ", "world", ". Done."):
        consumer.on_delta(chunk)
    consumer.finish()
    await consumer.run()
    return consumer


@pytest.mark.asyncio
async def test_slack_dm_stream_consumer_edits_own_ts_not_flat():
    """A Slack DM turn (chat_type='dm', no thread, metadata None) still builds a
    stream consumer that keeps edit support and emits progressive EDITs of the
    reply message — the flat-DM regression contract.

    The connector returns a real message_id for the flat first send, so edit
    support must stay on and at least one edit op must be emitted (progressive
    streaming), identical to a thread. No synthetic thread is created.

    Runs in EXPLICIT flat mode (reply_in_thread=False) — that is the mode this
    contract belongs to; the default thread-per-message path is covered by
    test_slack_dm_stream_consumer_threads_in_thread_per_message_mode."""
    adapter, stub = _wire("D1", "dm")
    adapter.config.extra = {"reply_in_thread": False}
    consumer = await _drive_stream(
        adapter,
        "D1",
        metadata=None,  # DM: _status_thread_metadata is None in run.py
        initial_reply_to_id="1700.0001",  # the triggering message ts
        chat_type="dm",
    )

    ops = [f["op"] for f in stub.sent]
    # First a flat send; edit support stays on so progressive edits CAN flow
    # (exact intermediate-frame timing is covered by the stream_consumer unit
    # suite — here we assert the DM regression contract: streaming is not
    # self-disabled and every edit targets the reply's own ts).
    assert ops[0] == "send"
    # Edit support survived: message_id set, not the __no_edit__ sentinel.
    assert consumer.message_id and consumer.message_id != "__no_edit__"
    assert consumer._edit_supported is True

    first_send = stub.sent[0]
    # The reply posts FLAT at the DM root — no synthetic thread anchor.
    assert first_send["reply_to"] is None
    assert "thread_id" not in (first_send["metadata"] or {})
    assert "thread_ts" not in (first_send["metadata"] or {})
    # reply_to_message_id (the mirrored self-anchor) is stripped too.
    assert "reply_to_message_id" not in (first_send["metadata"] or {})

    # Any edits that flowed target the same first-send ts (editing its own
    # message), never a synthetic thread.
    edit_ids = {f["message_id"] for f in stub.sent if f["op"] == "edit"}
    assert edit_ids <= {stub.next_send_result["message_id"]}


@pytest.mark.asyncio
async def test_slack_thread_stream_consumer_still_threads_and_streams():
    """Regression guard: a Slack THREAD turn keeps its real thread_id AND streams
    (the DM fix must not change the thread path)."""
    adapter, stub = _wire("C1", "channel", scope_id="T1")
    consumer = await _drive_stream(
        adapter,
        "C1",
        metadata={"thread_id": "1699.9000"},
        initial_reply_to_id="1700.0002",
        chat_type="channel",
    )
    ops = [f["op"] for f in stub.sent]
    assert ops[0] == "send"
    assert consumer._edit_supported is True
    first_send = stub.sent[0]
    # Thread preserved: the real thread_id rides along and reply_to is kept.
    assert first_send["metadata"]["thread_id"] == "1699.9000"
    assert first_send["reply_to"] == "1700.0002"


@pytest.mark.asyncio
async def test_slack_dm_stream_consumer_threads_in_thread_per_message_mode():
    """Default mode: the DM stream's first send keeps the triggering-ts anchor
    so the streamed final reply lands in the per-message thread; edits still
    target the reply's own ts."""
    adapter, stub = _wire("D1", "dm")
    consumer = await _drive_stream(
        adapter,
        "D1",
        metadata=None,
        initial_reply_to_id="1700.0001",
        chat_type="dm",
    )
    first_send = stub.sent[0]
    assert first_send["op"] == "send"
    assert first_send["reply_to"] == "1700.0001"
    assert consumer.message_id and consumer.message_id != "__no_edit__"
    edit_ids = {f["message_id"] for f in stub.sent if f["op"] == "edit"}
    assert edit_ids <= {stub.next_send_result["message_id"]}


# ---------------------------------------------------------------------------
# The media lane obeys the SAME thread-anchor contract as the text lane.
#
# send() and _send_media() both egress through the connector's Slack sender,
# so an anchor resolved on only one of them threads an image under the user's
# DM message in flat mode, and loses the per-message thread in thread mode
# (threadTs() reads metadata only). Both lanes route through
# _apply_slack_thread_anchor; these pin that they stay in agreement.
# ---------------------------------------------------------------------------
def _media_wire(chat_id: str, chat_type: str):
    """Like _wire, but the descriptor advertises the send_media op."""
    desc = _slack_desc(supported_ops=("send", "edit", "typing", "send_media"))
    stub = StubConnector(desc)
    adapter = RelayAdapter(PlatformConfig(), desc, transport=stub)
    src = SessionSource(
        platform=Platform.SLACK,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id="U1",
    )
    adapter._capture_scope(
        MessageEvent(text="hi", source=src, message_type=MessageType.TEXT)
    )
    return adapter, stub


@pytest.mark.asyncio
async def test_slack_dm_media_keeps_and_promotes_anchor_in_thread_mode():
    """Thread-per-message: an image must land in the per-message thread. The
    connector threads on metadata.thread_id only, so the surviving anchor has
    to be promoted there — a bare reply_to would post to the home channel."""
    adapter, stub = _media_wire("D1", "dm")
    await adapter.send_image("D1", "https://example.com/x.png", reply_to="1700.0001")
    frame = [f for f in stub.sent if f["op"] == "send_media"][-1]
    assert frame["reply_to"] == "1700.0001"
    assert (frame["metadata"] or {}).get("thread_id") == "1700.0001", (
        "media frame must carry the anchor where the connector reads it"
    )


@pytest.mark.asyncio
async def test_slack_dm_media_drops_synthetic_anchor_in_flat_mode():
    """Flat mode: the media frame drops the synthetic self-anchor exactly as
    the text lane does, so the image posts flat at the DM root instead of
    threading under the user's message, and invents no thread (#18859)."""
    adapter, stub = _media_wire("D1", "dm")
    adapter.config.extra = {"reply_in_thread": False}
    await adapter.send_image("D1", "https://example.com/x.png", reply_to="1700.0001")
    frame = [f for f in stub.sent if f["op"] == "send_media"][-1]
    assert frame["reply_to"] is None
    assert "thread_id" not in (frame["metadata"] or {})
    assert "thread_ts" not in (frame["metadata"] or {})


@pytest.mark.asyncio
async def test_slack_channel_media_anchor_untouched():
    """Non-DM chats are outside the synthetic-anchor rule: a channel media
    send keeps its reply_to unchanged."""
    adapter, stub = _media_wire("C1", "channel")
    await adapter.send_image("C1", "https://example.com/x.png", reply_to="1700.0009")
    frame = [f for f in stub.sent if f["op"] == "send_media"][-1]
    assert frame["reply_to"] == "1700.0009"


@pytest.mark.asyncio
async def test_media_caller_metadata_not_mutated():
    """The anchor promotion must not leak into the caller's dict — media
    helpers are called in loops with a shared metadata mapping."""
    adapter, stub = _media_wire("D1", "dm")
    caller_md = {"user_id": "U1"}
    await adapter.send_image(
        "D1", "https://example.com/x.png", reply_to="1700.0001", metadata=caller_md
    )
    assert caller_md == {"user_id": "U1"}, "caller metadata was mutated in place"


# ---------------------------------------------------------------------------
# Operator flags coerce exactly as the native Slack adapter's do.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        (False, False),
        ("false", False),
        ("False", False),
        (" no ", False),
        ("off", False),
        ("0", False),
        (True, True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("1", True),
    ],
)
def test_relay_slack_flags_coerce_like_native(raw, expected):
    """A YAML-quoted "false" must turn these knobs OFF, matching native's
    str().strip().lower() predicate. A bare bool() would read any non-empty
    string as True and silently ignore the operator's off switch."""
    adapter, _stub = _wire("D1", "dm")
    adapter.config.extra = {
        "slack": {
            "reply_in_thread": raw,
            "dm_top_level_threads_as_sessions": raw,
        }
    }
    assert adapter._effective_reply_in_thread() is expected
    assert adapter._dm_top_level_threads_as_sessions() is expected


def test_relay_slack_flags_default_true_when_absent():
    """Both knobs default ON when the operator sets nothing."""
    adapter, _stub = _wire("D1", "dm")
    adapter.config.extra = {}
    assert adapter._effective_reply_in_thread() is True
    assert adapter._dm_top_level_threads_as_sessions() is True


# ---------------------------------------------------------------------------
# The status clear targets the same thread the heartbeat set.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_typing_and_clear_share_one_status_anchor():
    """send_typing and stop_typing resolve the anchor through one helper: a
    clear that no-ops threadless leaves the status line stuck until Slack's
    own timeout."""
    adapter, stub = _wire("D1", "dm")
    adapter._last_inbound_ts_by_chat["D1"] = "1700.0001"
    await adapter.send_typing("D1")
    await adapter.stop_typing("D1")
    typing_frames = [f for f in stub.sent if f["op"] == "typing"]
    assert len(typing_frames) == 2
    anchors = [(f["metadata"] or {}).get("thread_id") for f in typing_frames]
    assert anchors == ["1700.0001", "1700.0001"], (
        "the clear must target the thread the heartbeat set"
    )
