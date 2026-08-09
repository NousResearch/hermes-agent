import asyncio
import json
import threading

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(config: GatewayConfig) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


@pytest.mark.asyncio
async def test_preprocess_includes_slack_author_mention_for_shared_thread():
    """Shared Slack threads expose the current author's verifiable user ID
    next to the display name so 'mention me again' requests can bind the
    mention to the CURRENT speaker (#17916)."""
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.SLACK: PlatformConfig(enabled=True, token="fake"),
            },
        )
    )
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_name="team-channel",
        chat_type="group",
        user_id="U123",
        user_name="Alice",
        thread_id="171.000",
    )
    event = MessageEvent(text="mention me again", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "[Verified sender: Alice | Slack user <@U123>] mention me again"


# ---------------------------------------------------------------------------
# Cross-sender pending-aggregation guard (upstream issue #69961)
#
# ``merge_pending_message_event`` folds a follow-up event into the pending slot
# by mutating ``existing.text`` / ``existing.media_urls`` in place, without ever
# consulting ``event.source``.  In a SHARED (multi-user) session that means a
# second participant's content is absorbed into the first participant's event
# and later rendered under the FIRST sender's ``[Verified sender: ...]``
# envelope.
#
# The invariant these tests assert is deliberately behavioural, not structural:
#
#   Every fragment of content in a pending turn must originate from the sender
#   identified by that turn's ``source`` — and refusing to merge must never
#   cost a message.
# ---------------------------------------------------------------------------

from gateway.platforms.base import (  # noqa: E402
    BasePlatformAdapter,
    MessageType,
    PlatformConfig,
    merge_pending_message_event,
)
from gateway.session import build_session_key  # noqa: E402


ALICE_MARK = "ALICE-FRAGMENT"
BOB_MARK = "BOB-FRAGMENT"
CAROL_MARK = "CAROL-FRAGMENT"


def _group_source(
    *,
    user_id=None,
    user_id_alt=None,
    user_name=None,
    chat_type="group",
    thread_id=None,
    platform=Platform.TELEGRAM,
):
    return SessionSource(
        platform=platform,
        chat_id="-1002285219667",
        chat_name="Shared Group",
        chat_type=chat_type,
        user_id=user_id,
        user_id_alt=user_id_alt,
        user_name=user_name,
        thread_id=thread_id,
    )


def _alice_source(**kw):
    kw.setdefault("user_id", "alice-1")
    kw.setdefault("user_name", "Alice")
    return _group_source(**kw)


def _bob_source(**kw):
    kw.setdefault("user_id", "bob-2")
    kw.setdefault("user_name", "Bob")
    return _group_source(**kw)


def _carol_source(**kw):
    kw.setdefault("user_id", "carol-3")
    kw.setdefault("user_name", "Carol")
    return _group_source(**kw)


def _text_event(text, source, message_id="m"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
    )


def _photo_event(source, url, caption="", message_id="p"):
    return MessageEvent(
        text=caption,
        message_type=MessageType.PHOTO,
        source=source,
        message_id=message_id,
        media_urls=[url],
        media_types=["image/jpeg"],
    )


def _empty_media_photo_event(source, caption="", message_id="p"):
    """A PHOTO whose adapter could not retain the downloaded attachment."""
    return MessageEvent(
        text=caption,
        message_type=MessageType.PHOTO,
        source=source,
        message_id=message_id,
        media_urls=[],
        media_types=[],
    )


def _sender_identity(source):
    """Identity used to attribute a turn: stable alt id preferred."""
    if source is None:
        return None
    return getattr(source, "user_id_alt", None) or getattr(source, "user_id", None)


def _all_media(*events):
    urls = []
    for ev in events:
        if ev is None:
            continue
        urls.extend(getattr(ev, "media_urls", None) or [])
    return urls


def _fragments_of(text):
    """Return the per-sender markers present in a rendered turn."""
    text = text or ""
    return {mark for mark in (ALICE_MARK, BOB_MARK) if mark in text}


# --- helper / acceptance matrix -------------------------------------------


def test_merge_pending_rejects_cross_sender_text_followup():
    """Bob's text must not be absorbed into Alice's pending turn."""
    pending = {}
    alice = _text_event(f"hello {ALICE_MARK}", _alice_source(), message_id="a1")
    pending["shared"] = alice

    bob = _text_event(f"and also {BOB_MARK}", _bob_source(), message_id="b1")
    absorbed = merge_pending_message_event(pending, "shared", bob, merge_text=True)

    slot = pending["shared"]
    # Invariant: the pending turn's content belongs to exactly one sender —
    # the one named by its own source.
    assert _fragments_of(slot.text) == {ALICE_MARK}, (
        "cross-sender text was absorbed into the pending turn: "
        f"{slot.text!r} under source {_sender_identity(slot.source)!r}"
    )
    assert absorbed is False, "refusal must be reported to the caller"


def test_merge_pending_rejects_cross_sender_photo_burst():
    """Bob's photo must not join Alice's album."""
    pending = {}
    alice = _photo_event(_alice_source(), "/tmp/alice-1.jpg", message_id="a1")
    pending["shared"] = alice

    bob = _photo_event(_bob_source(), "/tmp/bob-1.jpg", message_id="b1")
    absorbed = merge_pending_message_event(pending, "shared", bob)

    slot = pending["shared"]
    assert slot.media_urls == ["/tmp/alice-1.jpg"], (
        "cross-sender media joined the album: "
        f"{slot.media_urls!r} under source {_sender_identity(slot.source)!r}"
    )
    assert absorbed is False, "refusal must be reported to the caller"


def test_merge_pending_rejects_cross_sender_photo_then_text():
    """A caption-less pending photo must not adopt another sender's text.

    This is the worst variant: today the incoming text does not merge — it
    REPLACES the empty pending text, so the turn is 100% Bob's words under
    Alice's verified envelope.
    """
    pending = {}
    alice = _photo_event(_alice_source(), "/tmp/alice-1.jpg", caption="", message_id="a1")
    pending["shared"] = alice

    bob = _text_event(f"what is this {BOB_MARK}", _bob_source(), message_id="b1")
    absorbed = merge_pending_message_event(pending, "shared", bob, merge_text=True)

    slot = pending["shared"]
    assert BOB_MARK not in (slot.text or ""), (
        "another sender's text was stamped onto the pending photo turn: "
        f"{slot.text!r} under source {_sender_identity(slot.source)!r}"
    )
    assert absorbed is False, "refusal must be reported to the caller"


def test_merge_pending_preserves_same_sender_photo_album():
    """Same-sender bursts must keep album semantics."""
    pending = {}
    source = _alice_source()
    pending["shared"] = _photo_event(source, "/tmp/a-1.jpg", message_id="a1")

    merge_pending_message_event(
        pending, "shared", _photo_event(_alice_source(), "/tmp/a-2.jpg", message_id="a2")
    )

    slot = pending["shared"]
    assert slot.media_urls == ["/tmp/a-1.jpg", "/tmp/a-2.jpg"]
    assert _sender_identity(slot.source) == "alice-1"


def test_merge_pending_preserves_same_sender_text_followup():
    """Same-sender rapid text follow-ups must still concatenate."""
    pending = {}
    pending["shared"] = _text_event(f"part one {ALICE_MARK}", _alice_source(), message_id="a1")

    merge_pending_message_event(
        pending,
        "shared",
        _text_event("part two", _alice_source(), message_id="a2"),
        merge_text=True,
    )

    slot = pending["shared"]
    assert "part one" in slot.text and "part two" in slot.text
    assert _fragments_of(slot.text) == {ALICE_MARK}


def test_merge_pending_falls_back_when_pending_sender_unknown():
    """No identity on the pending side ⇒ no conflict; refusal must not cost a message."""
    pending = {}
    pending["shared"] = _text_event("anonymous head", _group_source(), message_id="x1")

    merge_pending_message_event(
        pending,
        "shared",
        _text_event(f"bob follow-up {BOB_MARK}", _bob_source(), message_id="b1"),
        merge_text=True,
    )

    slot = pending["shared"]
    assert BOB_MARK in (slot.text or ""), "unknown-identity merge must not drop the follow-up"


def test_merge_pending_falls_back_when_incoming_sender_unknown():
    """No identity on the incoming side ⇒ no conflict; the message must survive."""
    pending = {}
    pending["shared"] = _text_event(f"alice head {ALICE_MARK}", _alice_source(), message_id="a1")

    merge_pending_message_event(
        pending,
        "shared",
        _text_event("anonymous follow-up", _group_source(), message_id="x1"),
        merge_text=True,
    )

    slot = pending["shared"]
    assert "anonymous follow-up" in (slot.text or ""), (
        "unknown-identity merge must not drop the follow-up"
    )


# --- identity resolution / contract ---------------------------------------


def test_merge_pending_matches_sender_on_user_id_alt():
    """One human whose per-message user_id differs but user_id_alt is stable
    (Signal UUID / Feishu union_id) is the SAME sender and must still merge."""
    pending = {}
    pending["shared"] = _photo_event(
        _group_source(user_id="dev-a", user_id_alt="uuid-alice", user_name="Alice"),
        "/tmp/a-1.jpg",
        message_id="a1",
    )

    merge_pending_message_event(
        pending,
        "shared",
        _photo_event(
            _group_source(user_id="dev-b", user_id_alt="uuid-alice", user_name="Alice"),
            "/tmp/a-2.jpg",
            message_id="a2",
        ),
    )

    slot = pending["shared"]
    assert slot.media_urls == ["/tmp/a-1.jpg", "/tmp/a-2.jpg"], (
        "user_id_alt must take precedence over user_id when attributing a sender"
    )


def test_merge_pending_rejects_equal_raw_id_when_stable_ids_conflict():
    """Contradictory stable IDs cannot be overridden by a reused raw ID."""
    pending = {
        "shared": _text_event(
            "alice secret",
            _group_source(
                user_id="shared-phone",
                user_id_alt="uuid-alice",
                user_name="Alice",
            ),
            message_id="a1",
        )
    }
    bob = _text_event(
        "bob instruction",
        _group_source(
            user_id="shared-phone",
            user_id_alt="uuid-bob",
            user_name="Bob",
        ),
        message_id="b1",
    )

    absorbed = merge_pending_message_event(
        pending, "shared", bob, merge_text=True
    )

    assert absorbed is False
    assert pending["shared"].text == "alice secret"
    assert pending["shared"].source.user_id_alt == "uuid-alice"


def test_merge_pending_matches_same_raw_sender_when_alt_is_asymmetric():
    """A conditionally populated alternate ID must not split one person's album."""
    pending = {
        "shared": _photo_event(
            _group_source(user_id="raw-9", user_id_alt="stable-9", user_name="Alice"),
            "/tmp/a-1.jpg",
            message_id="a1",
        )
    }

    absorbed = merge_pending_message_event(
        pending,
        "shared",
        _photo_event(
            _group_source(user_id="raw-9", user_name="Alice"),
            "/tmp/a-2.jpg",
            message_id="a2",
        ),
    )

    assert absorbed is True
    assert pending["shared"].media_urls == ["/tmp/a-1.jpg", "/tmp/a-2.jpg"]


@pytest.mark.parametrize(
    ("existing_token", "incoming_token"),
    [("same-token", "same-token"), ("alice-alt", "bob-raw")],
)
def test_merge_pending_rejects_identity_bearing_disjoint_id_namespaces(
    existing_token, incoming_token
):
    """Alt-only and raw-only identities cannot prove they name one sender."""
    pending = {
        "shared": _text_event(
            "alice secret",
            _group_source(user_id_alt=existing_token, user_name="Alice"),
            message_id="a1",
        )
    }
    bob = _text_event(
        "bob words",
        _group_source(user_id=incoming_token, user_name="Bob"),
        message_id="b1",
    )

    absorbed = merge_pending_message_event(
        pending, "shared", bob, merge_text=True
    )

    assert absorbed is False
    assert pending["shared"].text == "alice secret"
    assert pending["shared"].source.user_name == "Alice"


def test_merge_pending_returns_false_only_on_sender_conflict():
    """The return value is the caller's ownership signal: True = absorbed."""
    same_sender_pending = {"k": _text_event("head", _alice_source(), message_id="a1")}
    absorbed_same = merge_pending_message_event(
        same_sender_pending, "k", _text_event("tail", _alice_source(), message_id="a2"),
        merge_text=True,
    )

    unknown_pending = {"k": _text_event("head", _group_source(), message_id="x1")}
    absorbed_unknown = merge_pending_message_event(
        unknown_pending, "k", _text_event("tail", _bob_source(), message_id="b1"),
        merge_text=True,
    )

    fresh_slot = {}
    absorbed_fresh = merge_pending_message_event(
        fresh_slot, "k", _text_event("first", _alice_source(), message_id="a1")
    )

    conflict_pending = {"k": _text_event("head", _alice_source(), message_id="a1")}
    absorbed_conflict = merge_pending_message_event(
        conflict_pending, "k", _text_event("tail", _bob_source(), message_id="b1"),
        merge_text=True,
    )

    assert absorbed_same is True, "same-sender merge must report absorption"
    assert absorbed_unknown is True, "unknown identity must not be reported as a conflict"
    assert absorbed_fresh is True, "storing into an empty slot is absorption"
    assert absorbed_conflict is False, "only a real sender conflict may refuse"


# --- caller-level: refusal must queue, never drop -------------------------


class _PendingStubAdapter(BasePlatformAdapter):
    """Minimal adapter exposing a real ``_pending_messages`` slot."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        from gateway.platforms.base import SendResult

        return SendResult(success=True, message_id="msg-1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}


def _runner_with_adapter():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._queued_events = {}
    adapter = _PendingStubAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter.set_pending_event_queue_handler(
        lambda session_key, event: runner._queue_or_replace_pending_event(
            session_key, event, adapter
        )
    )
    return runner, adapter


def _queued_turns(runner, adapter, session_key):
    """Every pending turn for a session, head slot first."""
    turns = []
    head = adapter._pending_messages.get(session_key)
    if head is not None:
        turns.append(head)
    turns.extend(runner._queued_events.get(session_key, []))
    return turns


def test_queue_or_replace_pending_event_fifos_cross_sender_media():
    """A cross-sender media follow-up must become its own turn, not join the head."""
    runner, adapter = _runner_with_adapter()
    session_key = "telegram:group:shared"

    alice = _photo_event(_alice_source(), "/tmp/alice-1.jpg", caption=ALICE_MARK, message_id="a1")
    bob = _photo_event(_bob_source(), "/tmp/bob-1.jpg", caption=BOB_MARK, message_id="b1")

    runner._queue_or_replace_pending_event(session_key, alice)
    runner._queue_or_replace_pending_event(session_key, bob)

    turns = _queued_turns(runner, adapter, session_key)

    # Nothing may be lost.
    assert sorted(_all_media(*turns)) == ["/tmp/alice-1.jpg", "/tmp/bob-1.jpg"]

    # And each turn's content must belong to its own sender.
    for turn in turns:
        identity = _sender_identity(turn.source)
        expected = {ALICE_MARK} if identity == "alice-1" else {BOB_MARK}
        assert _fragments_of(turn.text) <= expected, (
            f"turn attributed to {identity!r} carries another sender's content: {turn.text!r}"
        )


@pytest.mark.parametrize("alice_tail_kind", ["photo", "text"])
def test_queue_or_replace_pending_event_preserves_a_b_a_arrival_order(
    alice_tail_kind,
):
    """A later Alice follow-up must not jump ahead of Bob's refused FIFO turn."""
    runner, adapter = _runner_with_adapter()
    session_key = "telegram:group:a-b-a-runner"
    alice_head = _photo_event(
        _alice_source(), "/tmp/a-1.jpg", message_id="a1"
    )
    bob = _photo_event(_bob_source(), "/tmp/b-1.jpg", message_id="b1")
    alice_tail = (
        _photo_event(_alice_source(), "/tmp/a-2.jpg", message_id="a2")
        if alice_tail_kind == "photo"
        else _text_event("alice tail", _alice_source(), message_id="a2")
    )

    for event in (alice_head, bob, alice_tail):
        runner._queue_or_replace_pending_event(session_key, event)

    assert [turn.message_id for turn in _queued_turns(runner, adapter, session_key)] == [
        "a1",
        "b1",
        "a2",
    ]


def test_queue_or_replace_pending_event_merges_contiguous_media_at_fifo_tail():
    """An adjacent Bob photo burst may merge, but only behind Alice's head."""
    runner, adapter = _runner_with_adapter()
    session_key = "telegram:group:tail-album"

    for event in (
        _photo_event(_alice_source(), "/tmp/a-1.jpg", message_id="a1"),
        _photo_event(_bob_source(), "/tmp/b-1.jpg", message_id="b1"),
        _photo_event(_bob_source(), "/tmp/b-2.jpg", message_id="b2"),
    ):
        runner._queue_or_replace_pending_event(session_key, event)

    turns = _queued_turns(runner, adapter, session_key)
    assert [turn.message_id for turn in turns] == ["a1", "b1"]
    assert turns[1].media_urls == ["/tmp/b-1.jpg", "/tmp/b-2.jpg"]


def test_queue_or_replace_pending_event_preserves_empty_media_photo_after_head():
    """A failed photo download must not replace the occupied head slot."""
    runner, adapter = _runner_with_adapter()
    session_key = "telegram:group:empty-photo-head"
    source = _alice_source()
    events = [
        _text_event("first", source, message_id="a1"),
        _empty_media_photo_event(source, caption="photo caption", message_id="a2"),
    ]

    for event in events:
        runner._queue_or_replace_pending_event(session_key, event)

    assert [turn.message_id for turn in _queued_turns(runner, adapter, session_key)] == [
        "a1",
        "a2",
    ]


def test_queue_or_replace_pending_event_preserves_empty_media_photo_after_tail():
    """A failed photo download must not disappear behind an occupied FIFO tail."""
    runner, adapter = _runner_with_adapter()
    session_key = "telegram:group:empty-photo-tail"
    events = [
        _text_event("alice head", _alice_source(), message_id="a1"),
        _text_event("bob tail", _bob_source(), message_id="b1"),
        _empty_media_photo_event(
            _bob_source(), caption="photo caption", message_id="b2"
        ),
    ]

    for event in events:
        runner._queue_or_replace_pending_event(session_key, event)

    assert [turn.message_id for turn in _queued_turns(runner, adapter, session_key)] == [
        "a1",
        "b1",
        "b2",
    ]


def test_queue_or_replace_pending_event_preserves_empty_media_photo_in_dm():
    """The tail replacement outcome must also remain lossless for one DM sender."""
    runner, adapter = _runner_with_adapter()
    source = _alice_source(chat_type="dm")
    session_key = build_session_key(source)
    events = [
        _text_event("first", source, message_id="m1"),
        _text_event("second", source, message_id="m2"),
        _empty_media_photo_event(source, caption="photo caption", message_id="m3"),
    ]

    for event in events:
        runner._queue_or_replace_pending_event(session_key, event)

    assert [turn.message_id for turn in _queued_turns(runner, adapter, session_key)] == [
        "m1",
        "m2",
        "m3",
    ]


def test_busy_photo_followup_from_other_sender_does_not_join_album():
    """Alice's album stays Alice's; Bob's photo becomes a separate turn."""
    runner, adapter = _runner_with_adapter()
    session_key = "telegram:group:album"

    runner._queue_or_replace_pending_event(
        session_key, _photo_event(_alice_source(), "/tmp/a-1.jpg", message_id="a1")
    )
    runner._queue_or_replace_pending_event(
        session_key, _photo_event(_alice_source(), "/tmp/a-2.jpg", message_id="a2")
    )
    runner._queue_or_replace_pending_event(
        session_key, _photo_event(_bob_source(), "/tmp/b-1.jpg", message_id="b1")
    )

    turns = _queued_turns(runner, adapter, session_key)

    # No message lost.
    assert sorted(_all_media(*turns)) == ["/tmp/a-1.jpg", "/tmp/a-2.jpg", "/tmp/b-1.jpg"]

    # Alice's album is intact and un-contaminated.
    alice_turns = [t for t in turns if _sender_identity(t.source) == "alice-1"]
    assert alice_turns, "Alice's turn disappeared"
    alice_media = _all_media(*alice_turns)
    assert sorted(alice_media) == ["/tmp/a-1.jpg", "/tmp/a-2.jpg"], (
        f"Alice's album carries foreign media: {alice_media!r}"
    )


@pytest.mark.asyncio
async def test_adapter_busy_refusal_reaches_runner_fifo_and_drains_separately():
    """The adapter busy path must hand a refused event to the real FIFO."""
    runner, adapter = _runner_with_adapter()
    adapter.config.extra["group_sessions_per_user"] = False

    async def _unused_handler(event):
        return None

    adapter.set_message_handler(_unused_handler)
    alice = _photo_event(_alice_source(), "/tmp/a-1.jpg", message_id="a1")
    bob = _photo_event(_bob_source(), "/tmp/b-1.jpg", message_id="b1")
    session_key = build_session_key(bob.source, group_sessions_per_user=False)

    adapter._pending_messages[session_key] = alice
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter.handle_message(bob)

    drained = adapter._pending_messages.pop(session_key)
    next_turn = runner._promote_queued_event(session_key, adapter, None)

    assert drained is alice
    assert next_turn is not None
    assert next_turn is bob
    assert _sender_identity(drained.source) == "alice-1"
    assert _sender_identity(next_turn.source) == "bob-2"
    assert _all_media(drained) == ["/tmp/a-1.jpg"]
    assert _all_media(next_turn) == ["/tmp/b-1.jpg"]


@pytest.mark.asyncio
@pytest.mark.parametrize("alice_tail_kind", ["photo", "text"])
async def test_adapter_busy_path_preserves_a_b_a_arrival_order(alice_tail_kind):
    """The real adapter busy path must respect an already-occupied runner tail."""
    runner, adapter = _runner_with_adapter()
    adapter.config.extra["group_sessions_per_user"] = False

    async def _unused_handler(event):
        return None

    adapter.set_message_handler(_unused_handler)
    source = _alice_source()
    session_key = build_session_key(source, group_sessions_per_user=False)
    adapter._active_sessions[session_key] = asyncio.Event()
    events = [
        _photo_event(source, "/tmp/a-1.jpg", message_id="a1"),
        _photo_event(_bob_source(), "/tmp/b-1.jpg", message_id="b1"),
        (
            _photo_event(_alice_source(), "/tmp/a-2.jpg", message_id="a2")
            if alice_tail_kind == "photo"
            else _text_event("alice tail", _alice_source(), message_id="a2")
        ),
    ]

    for event in events:
        await adapter.handle_message(event)

    assert [turn.message_id for turn in _queued_turns(runner, adapter, session_key)] == [
        "a1",
        "b1",
        "a2",
    ]


@pytest.mark.asyncio
async def test_adapter_busy_path_preserves_empty_media_photo_at_fifo_tail():
    """The real busy adapter path must retain PHOTO metadata after download failure."""
    runner, adapter = _runner_with_adapter()
    adapter.config.extra["group_sessions_per_user"] = False

    async def _unused_handler(event):
        return None

    adapter.set_message_handler(_unused_handler)
    source = _alice_source()
    session_key = build_session_key(source, group_sessions_per_user=False)
    adapter._active_sessions[session_key] = asyncio.Event()
    events = [
        _text_event("alice head", source, message_id="a1"),
        _text_event("bob tail", _bob_source(), message_id="b1"),
        _empty_media_photo_event(
            _bob_source(), caption="download failed", message_id="b2"
        ),
    ]

    for event in events:
        await adapter.handle_message(event)

    assert [turn.message_id for turn in _queued_turns(runner, adapter, session_key)] == [
        "a1",
        "b1",
        "b2",
    ]


@pytest.mark.asyncio
async def test_adapter_without_runner_keeps_legacy_same_sender_album_semantics():
    """Standalone adapters still merge same-sender photos without runner wiring."""
    adapter = _PendingStubAdapter()
    adapter.config.extra["group_sessions_per_user"] = False

    async def _unused_handler(event):
        return None

    adapter.set_message_handler(_unused_handler)
    source = _alice_source()
    session_key = build_session_key(source, group_sessions_per_user=False)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(
        _photo_event(source, "/tmp/a-1.jpg", message_id="a1")
    )
    await adapter.handle_message(
        _photo_event(_alice_source(), "/tmp/a-2.jpg", message_id="a2")
    )

    assert adapter._pending_messages[session_key].media_urls == [
        "/tmp/a-1.jpg",
        "/tmp/a-2.jpg",
    ]


def test_cross_sender_refusal_at_busy_cap_remains_reachable():
    """A refused merge must survive even when the ordinary busy queue is full."""
    runner, adapter = _runner_with_adapter()
    session_key = "telegram:group:overflow"
    adapter._pending_messages[session_key] = _photo_event(
        _alice_source(), "/tmp/a.jpg", message_id="a"
    )

    # Fill the canonical queue to the production cap: one head plus 31 FIFO
    # entries. The incoming media event is then refused by Alice's head, so the
    # refusal contract — not the ordinary busy-overflow policy — owns it.
    for index in range(runner._BUSY_QUEUE_MAX_PENDING - 1):
        runner._enqueue_fifo(
            session_key,
            _text_event(f"alice-{index}", _alice_source(), message_id=f"a-{index}"),
            adapter,
        )

    bob = _photo_event(_bob_source(), "/tmp/b.jpg", message_id="b")
    runner._queue_or_replace_pending_event(session_key, bob)

    assert any(turn is bob for turn in _queued_turns(runner, adapter, session_key))


@pytest.mark.parametrize(
    ("existing_token", "incoming_token"),
    [("same-token", "same-token"), ("alice-alt", "bob-raw")],
)
def test_busy_queue_fifos_identity_bearing_disjoint_id_namespaces(
    existing_token, incoming_token
):
    """Production queueing keeps alt-only and raw-only turns separately attributed."""
    runner, adapter = _runner_with_adapter()
    session_key = "telegram:group:disjoint-id-namespaces"
    alice = _text_event(
        "alice secret",
        _group_source(user_id_alt=existing_token, user_name="Alice"),
        message_id="a1",
    )
    bob = _text_event(
        "bob words",
        _group_source(user_id=incoming_token, user_name="Bob"),
        message_id="b1",
    )
    adapter._pending_messages[session_key] = alice

    runner._queue_or_replace_pending_event(
        session_key, bob, adapter, merge_text=True
    )

    turns = _queued_turns(runner, adapter, session_key)
    assert [turn.text for turn in turns] == ["alice secret", "bob words"]
    assert [turn.source.user_name for turn in turns] == ["Alice", "Bob"]


@pytest.mark.asyncio
async def test_three_sender_queue_debounce_never_loses_latest_sender():
    """A blocked Bob flush must not make Carol disappear behind Alice's head."""
    runner, adapter = _runner_with_adapter()
    session_key = "telegram:group:three-sender-debounce"
    alice = _text_event(ALICE_MARK, _alice_source(), message_id="a")
    bob = _text_event(BOB_MARK, _bob_source(), message_id="b")
    carol = _text_event(CAROL_MARK, _carol_source(), message_id="c")
    adapter._pending_messages[session_key] = alice

    await adapter._queue_text_debounce(session_key, bob)
    await adapter._queue_text_debounce(session_key, carol)

    state = adapter._text_debounce_store().get(session_key)
    reachable = _queued_turns(runner, adapter, session_key)
    if state is not None:
        reachable.append(state.event)
    assert any(turn is carol for turn in reachable)


@pytest.mark.asyncio
async def test_queue_text_debounce_rejects_cross_field_sender_id_collision():
    """An alternate ID must never match another sender's raw ID."""
    runner, adapter = _runner_with_adapter()
    adapter._busy_text_debounce_seconds = 60.0
    adapter._busy_text_hard_cap_seconds = 60.0
    session_key = "telegram:group:cross-field-id-collision"
    alice = _text_event(
        "alice secret",
        _group_source(
            user_id="raw-alice",
            user_id_alt="collision-token",
            user_name="Alice",
        ),
        message_id="a1",
    )
    bob = _text_event(
        "bob words",
        _group_source(user_id="collision-token", user_name="Bob"),
        message_id="b1",
    )

    await adapter._queue_text_debounce(session_key, alice)
    await adapter._queue_text_debounce(session_key, bob)
    await adapter._flush_text_debounce_now(session_key)

    turns = _queued_turns(runner, adapter, session_key)
    state = adapter._text_debounce_store().get(session_key)
    if state is not None:
        turns.append(state.event)
    assert [turn.text for turn in turns] == ["alice secret", "bob words"]
    assert [turn.source.user_name for turn in turns] == ["Alice", "Bob"]


@pytest.mark.asyncio
async def test_queue_text_debounce_merges_same_raw_sender_with_asymmetric_alt():
    """A conditionally populated alternate ID must not split one text burst."""
    runner, adapter = _runner_with_adapter()
    adapter._busy_text_debounce_seconds = 60.0
    adapter._busy_text_hard_cap_seconds = 60.0
    session_key = "telegram:group:asymmetric-alt"

    await adapter._queue_text_debounce(
        session_key,
        _text_event(
            "part one",
            _group_source(
                user_id="raw-9", user_id_alt="stable-9", user_name="Alice"
            ),
            message_id="a1",
        ),
    )
    await adapter._queue_text_debounce(
        session_key,
        _text_event(
            "part two",
            _group_source(user_id="raw-9", user_name="Alice"),
            message_id="a2",
        ),
    )
    await adapter._flush_text_debounce_now(session_key)

    turns = _queued_turns(runner, adapter, session_key)
    assert len(turns) == 1
    assert turns[0].text == "part one\npart two"
    assert turns[0].message_id == "a2"


@pytest.mark.asyncio
async def test_queue_text_debounce_preserves_identity_free_dm_chat_fallback():
    """A stable private chat still identifies a sender when both IDs are absent."""
    runner, adapter = _runner_with_adapter()
    adapter._busy_text_debounce_seconds = 60.0
    adapter._busy_text_hard_cap_seconds = 60.0
    session_key = "telegram:dm:identity-free"

    await adapter._queue_text_debounce(
        session_key,
        _text_event(
            "part one",
            _group_source(chat_type="dm"),
            message_id="a1",
        ),
    )
    await adapter._queue_text_debounce(
        session_key,
        _text_event(
            "part two",
            _group_source(chat_type="private"),
            message_id="a2",
        ),
    )
    await adapter._flush_text_debounce_now(session_key)

    turns = _queued_turns(runner, adapter, session_key)
    assert len(turns) == 1
    assert turns[0].text == "part one\npart two"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("existing_chat_id", "incoming_chat_id"),
    [("", ""), ("None", None)],
)
async def test_queue_text_debounce_rejects_missing_identity_free_dm_chat_id(
    existing_chat_id, incoming_chat_id
):
    """Missing chat identity must not turn the DM fallback into a wildcard."""
    runner, adapter = _runner_with_adapter()
    adapter._busy_text_debounce_seconds = 60.0
    adapter._busy_text_hard_cap_seconds = 60.0
    session_key = "telegram:dm:missing-identity"
    existing_source = _group_source(chat_type="dm")
    incoming_source = _group_source(chat_type="private")
    existing_source.chat_id = existing_chat_id
    incoming_source.chat_id = incoming_chat_id

    await adapter._queue_text_debounce(
        session_key,
        _text_event("part one", existing_source, message_id="a1"),
    )
    await adapter._queue_text_debounce(
        session_key,
        _text_event("part two", incoming_source, message_id="a2"),
    )
    await adapter._flush_text_debounce_now(session_key)

    turns = _queued_turns(runner, adapter, session_key)
    state = adapter._text_debounce_store().get(session_key)
    if state is not None:
        turns.append(state.event)
    assert [turn.text for turn in turns] == ["part one", "part two"]


@pytest.mark.asyncio
async def test_queue_text_debounce_preserves_a_b_a_arrival_order():
    """A debounced Alice tail must not merge into Alice's head past Bob."""
    runner, adapter = _runner_with_adapter()
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = 60.0
    adapter._busy_text_hard_cap_seconds = 60.0
    session_key = "telegram:group:a-b-a-debounce"

    for event in (
        _text_event("alice head", _alice_source(), message_id="a1"),
        _text_event("bob middle", _bob_source(), message_id="b1"),
        _text_event("alice tail", _alice_source(), message_id="a2"),
    ):
        await adapter._queue_text_debounce(session_key, event)
    await adapter._flush_text_debounce_now(session_key)

    assert [turn.message_id for turn in _queued_turns(runner, adapter, session_key)] == [
        "a1",
        "b1",
        "a2",
    ]


def test_get_pending_message_only_consumes_and_never_promotes_fifo():
    """Interrupt/reset discard callers must not refill the adapter slot."""
    runner, adapter = _runner_with_adapter()
    session_key = "telegram:group:discard"
    alice = _photo_event(_alice_source(), "/tmp/a.jpg", message_id="a")
    bob = _photo_event(_bob_source(), "/tmp/b.jpg", message_id="b")
    adapter._pending_messages[session_key] = alice
    adapter._queue_refused_pending_event(session_key, bob)

    assert adapter.get_pending_message(session_key) is alice
    assert session_key not in adapter._pending_messages
    assert runner._session_state(session_key).conversation.queued_events == [bob]


@pytest.mark.asyncio
async def test_stop_consumes_adapter_head_and_refused_runner_fifo():
    """Stopping a session must discard every queued turn, including refusals."""
    runner, adapter = _runner_with_adapter()
    session_key = "telegram:group:stop-discard"
    alice = _photo_event(_alice_source(), "/tmp/a.jpg", message_id="a")
    bob = _photo_event(_bob_source(), "/tmp/b.jpg", message_id="b")
    adapter._pending_messages[session_key] = alice
    runner._enqueue_fifo(session_key, bob, adapter)

    await runner._interrupt_and_clear_session(
        session_key,
        alice.source,
        interrupt_reason="test-stop",
        invalidation_reason="test-stop",
    )

    assert session_key not in adapter._pending_messages
    assert runner._queue_depth(session_key, adapter=adapter) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", ["stop", "new", "reset"])
async def test_control_command_discards_only_preexisting_debounce(cmd):
    """Control completion drops stale debounce but drains an in-command follow-up."""
    runner, adapter = _runner_with_adapter()
    adapter.config.extra["group_sessions_per_user"] = False
    adapter._busy_text_debounce_seconds = 60.0
    adapter._busy_text_hard_cap_seconds = 60.0
    session_key = build_session_key(_alice_source(), group_sessions_per_user=False)

    alice = _photo_event(_alice_source(), "/tmp/a.jpg", message_id="a")
    bob = _photo_event(_bob_source(), "/tmp/b.jpg", message_id="b")
    stale = _text_event("stale-before-control", _carol_source(), message_id="c")
    during = _text_event("follow-up-during-control", _bob_source(), message_id="d")
    command = _text_event(f"/{cmd}", _alice_source(), message_id=f"cmd-{cmd}")

    adapter._pending_messages[session_key] = alice
    runner._queue_or_replace_pending_event(session_key, bob)
    await adapter._queue_text_debounce(session_key, stale)
    adapter._active_sessions[session_key] = asyncio.Event()
    delivered = []

    async def _handler(event):
        if event is command:
            await runner._interrupt_and_clear_session(
                session_key,
                command.source,
                interrupt_reason=f"test-{cmd}",
                invalidation_reason=f"test-{cmd}",
            )
            await adapter._queue_text_debounce(session_key, during)
            return f"/{cmd} complete"
        delivered.append(event)
        return None

    adapter.set_message_handler(_handler)
    await adapter._dispatch_active_session_command(command, session_key, cmd)
    task = adapter._session_tasks.get(session_key)
    if task is not None:
        await asyncio.wait_for(task, timeout=1.0)

    assert stale not in delivered, f"/{cmd} replayed pre-control debounce work"
    assert delivered == [during], f"/{cmd} lost or duplicated the in-command follow-up"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "same_sender", [False, True], ids=["different-sender", "same-sender"]
)
@pytest.mark.parametrize("cmd", ["stop", "new", "reset"])
async def test_control_command_preserves_followup_arriving_during_topic_recovery(
    cmd, same_sender
):
    """The outer handle_message boundary must not misclassify a racing follow-up as stale."""
    runner, adapter = _runner_with_adapter()
    adapter.config.extra["group_sessions_per_user"] = False
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = 60.0
    adapter._busy_text_hard_cap_seconds = 60.0
    # Latest-main limits Telegram topic recovery to the private-DM lanes where
    # it is meaningful. Keep this race probe on that real recovery path rather
    # than forcing every group message through the shared executor.
    recovered_source = _alice_source(chat_type="dm", thread_id="topic-7")
    session_key = build_session_key(recovered_source, group_sessions_per_user=False)
    adapter._active_sessions[session_key] = asyncio.Event()

    command = _text_event(
        f"/{cmd}", _alice_source(chat_type="dm"), message_id=f"cmd-{cmd}"
    )
    stale_source = (
        _bob_source(chat_type="dm", thread_id="topic-7")
        if same_sender
        else _carol_source(chat_type="dm", thread_id="topic-7")
    )
    stale = _text_event("stale-before-control", stale_source, message_id="stale")
    during = _text_event(
        "follow-up-during-topic-recovery",
        _bob_source(chat_type="dm", thread_id="topic-7"),
        message_id="during",
    )
    recovery_entered = threading.Event()
    release_recovery = threading.Event()
    delivered = []

    def _recover(source):
        if source.user_id == command.source.user_id:
            recovery_entered.set()
            assert release_recovery.wait(timeout=2.0)
        return "topic-7" if source.user_id == command.source.user_id else source.thread_id

    async def _handler(event):
        if event is command:
            await runner._interrupt_and_clear_session(
                session_key,
                command.source,
                interrupt_reason=f"test-{cmd}",
                invalidation_reason=f"test-{cmd}",
            )
            return f"/{cmd} complete"
        delivered.append(event)
        return None

    adapter.set_topic_recovery_fn(_recover)
    adapter.set_message_handler(_handler)
    await adapter._queue_text_debounce(session_key, stale)
    command_task = asyncio.create_task(adapter.handle_message(command))
    try:
        assert await asyncio.to_thread(recovery_entered.wait, 1.0)
        await adapter.handle_message(during)
        state = adapter._text_debounce_store().get(session_key)
        assert state is not None
        if same_sender:
            assert state.event is stale
            assert state.event.text == (
                "stale-before-control\nfollow-up-during-topic-recovery"
            )
        else:
            assert state.event is during
    finally:
        release_recovery.set()
    await asyncio.wait_for(command_task, timeout=2.0)
    task = adapter._session_tasks.get(session_key)
    if task is not None:
        await asyncio.wait_for(task, timeout=1.0)

    assert [event.text for event in delivered] == [
        "follow-up-during-topic-recovery"
    ], f"/{cmd} replayed stale work or lost/duplicated the follow-up"
    assert delivered[0].message_id == "during", f"/{cmd} lost latest follow-up metadata"


def test_refused_runner_fifo_is_preserved_in_shutdown_forensics(tmp_path, monkeypatch):
    """A refused overflow turn must reach disk without stopping the gateway."""
    from gateway import shutdown_flush

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, adapter = _runner_with_adapter()
    session_key = "telegram:group:shutdown-refusal"
    alice = _photo_event(
        _alice_source(), "/tmp/a.jpg", caption=ALICE_MARK, message_id="a"
    )
    bob = _photo_event(
        _bob_source(user_id_alt="bob-stable", thread_id="topic-7"),
        "/tmp/b.jpg",
        caption="",
        message_id="b",
    )

    adapter._pending_messages[session_key] = alice
    runner._queue_or_replace_pending_event(session_key, bob)

    # Exercise the same stores used by adapter and runner graceful shutdown.
    shutdown_flush.flush_pending_to_file(
        dict(adapter._pending_messages), reason="test-adapter-shutdown"
    )
    shutdown_flush.flush_pending_to_file(
        dict(runner._pending_messages), reason="test-runner-shutdown"
    )
    flush_fifo = getattr(
        shutdown_flush,
        "flush_queued_events_to_file",
        lambda queued, *, reason: 0,
    )
    flush_fifo(dict(runner._queued_events), reason="test-runner-fifo-shutdown")

    payload_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "pending_messages").glob("*.json")
    )
    assert "/tmp/b.jpg" in payload_text, (
        "refused runner-FIFO media is absent from shutdown forensics"
    )
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "pending_messages").glob("*.json")
    ]
    bob_payload = next(
        payload for payload in payloads if "/tmp/b.jpg" in payload["data"].get("media_urls", [])
    )
    assert bob_payload["data"]["source"] == {
        "platform": "telegram",
        "chat_id": "-1002285219667",
        "chat_name": "Shared Group",
        "chat_type": "group",
        "user_id": "bob-2",
        "user_name": "Bob",
        "thread_id": "topic-7",
        "chat_topic": None,
        "user_id_alt": "bob-stable",
    }


@pytest.mark.asyncio
async def test_refused_events_live_only_in_runner_fifo_not_adapter_reset_state():
    """Adapter discard paths must not leave refusal work that can resurrect."""
    runner, adapter = _runner_with_adapter()
    session_key = "telegram:group:reset"
    bob = _photo_event(_bob_source(), "/tmp/b.jpg", message_id="b")

    adapter._queue_refused_pending_event(session_key, bob)
    await adapter.cancel_session_processing(session_key)

    assert not hasattr(adapter, "_refused_pending_events")
    assert runner._session_state(session_key).conversation.queued_events == []
    assert session_key not in adapter._pending_messages


# --- end-to-end security property -----------------------------------------


@pytest.mark.asyncio
async def test_verified_sender_envelope_matches_each_queued_turn():
    """The rendered envelope must name the author of every fragment it wraps."""
    runner = _make_runner(
        GatewayConfig(
            platforms={Platform.SLACK: PlatformConfig(enabled=True, token="fake")},
        )
    )

    alice_src = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_name="team-channel",
        chat_type="group",
        user_id="U-ALICE",
        user_name="Alice",
        thread_id="171.000",
    )
    bob_src = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_name="team-channel",
        chat_type="group",
        user_id="U-BOB",
        user_name="Bob",
        thread_id="171.000",
    )

    pending = {}
    pending["shared"] = _text_event(f"deploy now {ALICE_MARK}", alice_src, message_id="a1")
    merge_pending_message_event(
        pending,
        "shared",
        _text_event(f"and wipe the db {BOB_MARK}", bob_src, message_id="b1"),
        merge_text=True,
    )

    slot = pending["shared"]
    rendered = await runner._prepare_inbound_message_text(
        event=slot, source=slot.source, history=[]
    )

    envelope_owner = "Alice" if "Alice" in (rendered or "") else "Bob"
    foreign = BOB_MARK if envelope_owner == "Alice" else ALICE_MARK
    assert foreign not in (rendered or ""), (
        f"envelope names {envelope_owner} but the turn carries the other sender's "
        f"content: {rendered!r}"
    )


# --- regression ------------------------------------------------------------


def test_merge_pending_dm_session_behaviour_unchanged():
    """DMs are per-user by construction, so the guard is unreachable there and
    merge behaviour must be byte-for-byte what it was before."""
    alice_dm = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="555",
        chat_type="dm",
        user_id="alice-1",
        user_name="Alice",
    )
    bob_dm = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="666",
        chat_type="dm",
        user_id="bob-2",
        user_name="Bob",
    )

    # Two distinct senders can never share a DM session key.
    assert build_session_key(alice_dm) != build_session_key(bob_dm)

    pending = {}
    pending["dm"] = _text_event(f"part one {ALICE_MARK}", alice_dm, message_id="a1")
    merge_pending_message_event(
        pending, "dm", _text_event("part two", alice_dm, message_id="a2"), merge_text=True
    )
    merge_pending_message_event(
        pending,
        "dm",
        _photo_event(alice_dm, "/tmp/a-1.jpg", caption="look", message_id="a3"),
    )

    slot = pending["dm"]
    assert "part one" in slot.text and "part two" in slot.text and "look" in slot.text
    assert slot.media_urls == ["/tmp/a-1.jpg"]
    assert slot.message_type == MessageType.PHOTO
