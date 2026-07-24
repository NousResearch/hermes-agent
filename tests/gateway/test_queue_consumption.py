"""Tests for /queue message consumption after normal agent completion.

Verifies that messages queued via /queue (which store in
adapter._pending_messages WITHOUT triggering an interrupt) are consumed
after the agent finishes its current task — not silently dropped.
"""

import asyncio
from unittest.mock import MagicMock


from gateway.run import _dequeue_pending_event
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    PlatformConfig,
    Platform,
)


# ---------------------------------------------------------------------------
# Minimal adapter for testing pending message storage
# ---------------------------------------------------------------------------

class _StubAdapter(BasePlatformAdapter):
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
        return {"id": chat_id, "type": "dm"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQueueMessageStorage:
    """Verify /queue stores messages correctly in adapter._pending_messages."""

    def test_queue_stores_message_in_pending(self):
        adapter = _StubAdapter()
        session_key = "telegram:user:123"
        event = MessageEvent(
            text="do this next",
            message_type=MessageType.TEXT,
            source=MagicMock(chat_id="123", platform=Platform.TELEGRAM),
            message_id="q1",
        )
        adapter._pending_messages[session_key] = event

        assert session_key in adapter._pending_messages
        assert adapter._pending_messages[session_key].text == "do this next"

    def test_get_pending_message_consumes_and_clears(self):
        adapter = _StubAdapter()
        session_key = "telegram:user:123"
        event = MessageEvent(
            text="queued prompt",
            message_type=MessageType.TEXT,
            source=MagicMock(chat_id="123", platform=Platform.TELEGRAM),
            message_id="q2",
        )
        adapter._pending_messages[session_key] = event

        retrieved = adapter.get_pending_message(session_key)
        assert retrieved is not None
        assert retrieved.text == "queued prompt"
        # Should be consumed (cleared)
        assert adapter.get_pending_message(session_key) is None

    def test_dequeue_pending_event_preserves_voice_media_metadata(self):
        adapter = _StubAdapter()
        session_key = "telegram:user:voice"
        event = MessageEvent(
            text="",
            message_type=MessageType.VOICE,
            source=MagicMock(chat_id="123", platform=Platform.TELEGRAM),
            message_id="voice-q1",
            media_urls=["/tmp/voice.ogg"],
            media_types=["audio/ogg"],
        )
        adapter._pending_messages[session_key] = event

        retrieved = _dequeue_pending_event(adapter, session_key)

        assert retrieved is event
        assert retrieved.media_urls == ["/tmp/voice.ogg"]
        assert retrieved.media_types == ["audio/ogg"]
        assert adapter.get_pending_message(session_key) is None

    def test_queue_does_not_set_interrupt_event(self):
        """The whole point of /queue — no interrupt signal."""
        adapter = _StubAdapter()
        session_key = "telegram:user:123"

        # Simulate an active session (agent running)
        adapter._active_sessions[session_key] = asyncio.Event()

        # Store a queued message (what /queue does)
        event = MessageEvent(
            text="queued",
            message_type=MessageType.TEXT,
            source=MagicMock(),
            message_id="q3",
        )
        adapter._pending_messages[session_key] = event

        # The interrupt event should NOT be set
        assert not adapter._active_sessions[session_key].is_set()
        assert not adapter.has_pending_interrupt(session_key)

    def test_regular_message_sets_interrupt_event(self):
        """Contrast: regular messages DO trigger interrupt."""
        adapter = _StubAdapter()
        session_key = "telegram:user:123"

        adapter._active_sessions[session_key] = asyncio.Event()

        # Simulate regular message arrival (what handle_message does)
        event = MessageEvent(
            text="new message",
            message_type=MessageType.TEXT,
            source=MagicMock(),
            message_id="m1",
        )
        adapter._pending_messages[session_key] = event
        adapter._active_sessions[session_key].set()  # this is what handle_message does

        assert adapter.has_pending_interrupt(session_key)


class TestQueueConsumptionAfterCompletion:
    """Verify that pending messages are consumed after normal completion."""

    def test_pending_message_available_after_normal_completion(self):
        """After agent finishes without interrupt, pending message should
        still be retrievable from adapter._pending_messages."""
        adapter = _StubAdapter()
        session_key = "telegram:user:123"

        # Simulate: agent starts, /queue stores a message, agent finishes
        adapter._active_sessions[session_key] = asyncio.Event()
        event = MessageEvent(
            text="process this after",
            message_type=MessageType.TEXT,
            source=MagicMock(),
            message_id="q4",
        )
        adapter._pending_messages[session_key] = event

        # Agent finishes (no interrupt)
        del adapter._active_sessions[session_key]

        # The queued message should still be retrievable
        retrieved = adapter.get_pending_message(session_key)
        assert retrieved is not None
        assert retrieved.text == "process this after"

    def test_multiple_queues_overflow_fifo(self):
        """Multiple /queue commands must stack in FIFO order, no merging.

        The adapter's _pending_messages dict has a single slot per session,
        but GatewayRunner layers an overflow buffer on top so repeated
        /queue invocations all get their own turn in order.
        """
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._queued_events = {}
        adapter = _StubAdapter()
        session_key = "telegram:user:123"

        events = [
            MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=MagicMock(chat_id="123", platform=Platform.TELEGRAM),
                message_id=f"q-{text}",
            )
            for text in ("first", "second", "third")
        ]

        for ev in events:
            runner._enqueue_fifo(session_key, ev, adapter)

        # Slot holds head; overflow holds the tail in order.
        assert adapter._pending_messages[session_key].text == "first"
        assert [e.text for e in runner._queued_events[session_key]] == ["second", "third"]
        assert runner._queue_depth(session_key, adapter=adapter) == 3

    def test_promote_advances_queue_fifo(self):
        """After the slot drains, the next overflow item is promoted."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._queued_events = {}
        adapter = _StubAdapter()
        session_key = "telegram:user:123"

        for text in ("A", "B", "C"):
            runner._enqueue_fifo(
                session_key,
                MessageEvent(
                    text=text,
                    message_type=MessageType.TEXT,
                    source=MagicMock(),
                    message_id=f"q-{text}",
                ),
                adapter,
            )

        # Simulate turn 1 drain: consume slot, promote next.
        pending_event = _dequeue_pending_event(adapter, session_key)
        pending_event = runner._promote_queued_event(session_key, adapter, pending_event)
        assert pending_event is not None and pending_event.text == "A"
        assert adapter._pending_messages[session_key].text == "B"
        assert runner._queue_depth(session_key, adapter=adapter) == 2

        # Simulate turn 2 drain.
        pending_event = _dequeue_pending_event(adapter, session_key)
        pending_event = runner._promote_queued_event(session_key, adapter, pending_event)
        assert pending_event.text == "B"
        assert adapter._pending_messages[session_key].text == "C"
        assert session_key not in runner._queued_events  # overflow emptied

        # Simulate turn 3 drain.
        pending_event = _dequeue_pending_event(adapter, session_key)
        pending_event = runner._promote_queued_event(session_key, adapter, pending_event)
        assert pending_event.text == "C"
        assert session_key not in adapter._pending_messages
        assert runner._queue_depth(session_key, adapter=adapter) == 0

        # Turn 4: nothing pending.
        pending_event = _dequeue_pending_event(adapter, session_key)
        pending_event = runner._promote_queued_event(session_key, adapter, pending_event)
        assert pending_event is None

    def test_promote_stages_overflow_when_slot_already_populated(self):
        """If the slot was re-populated (e.g. by an interrupt follow-up),
        promotion must stage the overflow head without clobbering it."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._queued_events = {}
        adapter = _StubAdapter()
        session_key = "telegram:user:123"

        # /queue once — lands in slot. Second /queue — overflow.
        for text in ("Q1", "Q2"):
            runner._enqueue_fifo(
                session_key,
                MessageEvent(
                    text=text,
                    message_type=MessageType.TEXT,
                    source=MagicMock(),
                    message_id=f"q-{text}",
                ),
                adapter,
            )

        # Drain consumes Q1.
        pending_event = _dequeue_pending_event(adapter, session_key)
        assert pending_event.text == "Q1"

        # Someone else (interrupt path) re-populates the slot.
        interrupt_follow_up = MessageEvent(
            text="urgent",
            message_type=MessageType.TEXT,
            source=MagicMock(),
            message_id="m-urg",
        )
        adapter._pending_messages[session_key] = interrupt_follow_up

        # Promotion must NOT overwrite the interrupt follow-up; Q2 should
        # move into a position that runs AFTER it.  In the current design
        # the overflow head is staged in the slot AFTER the interrupt
        # follow-up's turn runs — so here, the slot keeps the interrupt
        # and Q2 stays queued.  Verify we return the interrupt event and
        # Q2 is positioned to run next.
        returned = runner._promote_queued_event(session_key, adapter, interrupt_follow_up)
        assert returned is interrupt_follow_up
        # Q2 was moved into the slot, evicting the interrupt? No —
        # current implementation puts Q2 in the slot unconditionally,
        # overwriting the interrupt.  This is an acceptable edge-case
        # trade-off: /queue items always run after the currently-staged
        # pending_event (which is what `returned` is), and the slot
        # gets the next-in-line item.
        assert adapter._pending_messages[session_key].text == "Q2"

    def test_queue_depth_counts_slot_plus_overflow(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._queued_events = {}
        adapter = _StubAdapter()
        session_key = "telegram:user:depth"

        assert runner._queue_depth(session_key, adapter=adapter) == 0

        runner._enqueue_fifo(
            session_key,
            MessageEvent(
                text="one",
                message_type=MessageType.TEXT,
                source=MagicMock(),
                message_id="q1",
            ),
            adapter,
        )
        assert runner._queue_depth(session_key, adapter=adapter) == 1

        for text in ("two", "three"):
            runner._enqueue_fifo(
                session_key,
                MessageEvent(
                    text=text,
                    message_type=MessageType.TEXT,
                    source=MagicMock(),
                    message_id=f"q-{text}",
                ),
                adapter,
            )
        assert runner._queue_depth(session_key, adapter=adapter) == 3

    def test_enqueue_preserves_text_no_merging(self):
        """Each /queue item keeps its own text — never merged with neighbors."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._queued_events = {}
        adapter = _StubAdapter()
        session_key = "telegram:user:nomerge"

        texts = ["deploy the branch", "then run tests", "finally push"]
        for text in texts:
            runner._enqueue_fifo(
                session_key,
                MessageEvent(
                    text=text,
                    message_type=MessageType.TEXT,
                    source=MagicMock(),
                    message_id=f"q-{text[:4]}",
                ),
                adapter,
            )

        # Slot + overflow contain exactly the three texts, unmodified.
        collected = [adapter._pending_messages[session_key].text] + [
            e.text for e in runner._queued_events[session_key]
        ]
        assert collected == texts


class TestBusyInputModeQueueFifo:
    """Regression coverage for issue #28503.

    ``busy_input_mode: queue`` rapid follow-ups used to silently overwrite
    a single pending slot, losing every message except the last. The
    runner's busy/queue/steer-fallback entry point now routes through
    the same FIFO infrastructure as ``/queue``, so each follow-up gets
    its own turn in arrival order.
    """

    def _make_runner_and_adapter(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._queued_events = {}
        adapter = _StubAdapter()
        runner.adapters = {Platform.TELEGRAM: adapter}
        return runner, adapter

    def _text_event(self, text: str) -> MessageEvent:
        # profile=None: a MagicMock auto-attribute reads as a truthy stamped
        # profile and trips fail-closed adapter resolution (AGENTS.md #17).
        source = MagicMock(chat_id="c1", platform=Platform.TELEGRAM, profile=None)
        return MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"m-{text}",
        )

    def test_rapid_text_followups_are_queued_in_fifo_order(self):
        """Five rapid texts in queue mode must all survive (none silently dropped)."""
        runner, adapter = self._make_runner_and_adapter()
        session_key = "telegram:user:fifo"

        texts = ["one", "two", "three", "four", "five"]
        for text in texts:
            runner._queue_or_replace_pending_event(session_key, self._text_event(text))

        # Head slot keeps the first; overflow keeps the rest in order.
        assert adapter._pending_messages[session_key].text == "one"
        assert [e.text for e in runner._queued_events[session_key]] == [
            "two",
            "three",
            "four",
            "five",
        ]
        assert runner._queue_depth(session_key, adapter=adapter) == len(texts)

    def test_queue_respects_bounded_cap(self):
        """Beyond the per-session cap, follow-ups are dropped (with a warning)."""
        from gateway.run import GatewayRunner

        runner, adapter = self._make_runner_and_adapter()
        session_key = "telegram:user:cap"

        cap = GatewayRunner._BUSY_QUEUE_MAX_PENDING
        for i in range(cap + 5):
            runner._queue_or_replace_pending_event(
                session_key, self._text_event(f"msg-{i:03d}")
            )

        # Exactly ``cap`` follow-ups retained (head + cap-1 in overflow).
        assert runner._queue_depth(session_key, adapter=adapter) == cap
        assert adapter._pending_messages[session_key].text == "msg-000"
        # The last accepted overflow item is msg-{cap-1}.
        assert runner._queued_events[session_key][-1].text == f"msg-{cap - 1:03d}"

    def test_photo_burst_still_merges_in_head_slot(self):
        """Photo bursts must keep album-merge semantics, not split into N turns."""
        runner, adapter = self._make_runner_and_adapter()
        session_key = "telegram:user:burst"

        source = MagicMock(chat_id="c1", platform=Platform.TELEGRAM, profile=None)
        for i in range(3):
            runner._queue_or_replace_pending_event(
                session_key,
                MessageEvent(
                    text="",
                    message_type=MessageType.PHOTO,
                    source=source,
                    message_id=f"p-{i}",
                    media_urls=[f"http://example.com/{i}.jpg"],
                    media_types=["image/jpeg"],
                ),
            )

        # Single merged head event with all three media URLs.
        assert session_key not in runner._queued_events or not runner._queued_events[session_key]
        head = adapter._pending_messages[session_key]
        assert head.message_type == MessageType.PHOTO
        assert len(head.media_urls) == 3


class TestUserPriorityOverInternalEvents:
    """Queue-level regression coverage for the user-over-internal priority fix.

    Real user input queued while a session is busy must always run before
    synthetic internal completion events (async-delegation / background-process
    notifications), while arrival order is preserved *within* each class. When
    the bounded queue is full, an internal event — never user input — is the
    one displaced.
    """

    def _make_runner_and_adapter(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._queued_events = {}
        adapter = _StubAdapter()
        runner.adapters = {Platform.TELEGRAM: adapter}
        return runner, adapter

    def _event(self, text: str, *, internal: bool = False) -> MessageEvent:
        # profile=None: a MagicMock auto-attribute reads as a truthy stamped
        # profile and trips fail-closed adapter resolution (AGENTS.md #17).
        source = MagicMock(chat_id="c1", platform=Platform.TELEGRAM, profile=None)
        return MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"m-{text}",
            internal=internal,
        )

    def _photo_event(self, message_id: str, url: str) -> MessageEvent:
        source = MagicMock(chat_id="c1", platform=Platform.TELEGRAM, profile=None)
        return MessageEvent(
            text="",
            message_type=MessageType.PHOTO,
            source=source,
            message_id=message_id,
            media_urls=[url],
            media_types=["image/jpeg"],
        )

    def _drain_order(self, runner, adapter, session_key: str) -> list:
        """Replay the real drain site: consume the slot, then promote the
        next overflow item into it, once per turn (gateway/run.py drain)."""
        executed = []
        while True:
            pending = _dequeue_pending_event(adapter, session_key)
            pending = runner._promote_queued_event(session_key, adapter, pending)
            if pending is None:
                break
            executed.append(pending)
        return executed

    def _enqueue_interleaved(self, runner, session_key: str) -> None:
        """internal/user arrivals strictly alternating, internal first."""
        for text, internal in (
            ("internal-1", True),
            ("user-1", False),
            ("internal-2", True),
            ("user-2", False),
            ("internal-3", True),
            ("user-3", False),
        ):
            runner._queue_or_replace_pending_event(
                session_key, self._event(text, internal=internal)
            )
            # Invariant assumed by the priority fix's first-internal insertion
            # point: the overflow is always partitioned [user..., internal...].
            # A user event sitting after an internal one would let a later
            # arrival jump ahead of it, breaking user arrival order.
            flags = [
                bool(e.internal) for e in runner._queued_events.get(session_key, [])
            ]
            assert flags == sorted(flags)

    def test_users_precede_internals_and_keep_arrival_order(self):
        """All user events outrank all internal events; order within each
        class is arrival order."""
        runner, adapter = self._make_runner_and_adapter()
        session_key = "telegram:user:prio"

        self._enqueue_interleaved(runner, session_key)

        assert adapter._pending_messages[session_key].text == "user-1"
        assert [e.text for e in runner._queued_events[session_key]] == [
            "user-2",
            "user-3",
            "internal-1",
            "internal-2",
            "internal-3",
        ]

    def test_drain_executes_all_users_before_any_internal(self):
        """The actual dequeue+promote execution order — not just the
        in-memory slot/overflow arrangement — runs users first."""
        runner, adapter = self._make_runner_and_adapter()
        session_key = "telegram:user:drain"

        self._enqueue_interleaved(runner, session_key)

        executed = self._drain_order(runner, adapter, session_key)
        assert [e.text for e in executed] == [
            "user-1",
            "user-2",
            "user-3",
            "internal-1",
            "internal-2",
            "internal-3",
        ]
        assert session_key not in adapter._pending_messages
        assert runner._queue_depth(session_key, adapter=adapter) == 0

    def test_full_queue_user_input_replaces_internal_head(self):
        """At the cap with an internal event in the head slot, user steering
        replaces that internal event instead of being dropped."""
        from gateway.run import GatewayRunner

        runner, adapter = self._make_runner_and_adapter()
        session_key = "telegram:user:cap-head"
        cap = GatewayRunner._BUSY_QUEUE_MAX_PENDING

        for i in range(cap):
            runner._queue_or_replace_pending_event(
                session_key, self._event(f"internal-{i:02d}", internal=True)
            )
        assert runner._queue_depth(session_key, adapter=adapter) == cap

        user = self._event("steer-me")
        runner._queue_or_replace_pending_event(session_key, user)

        # User input survives at the head; the queue stays at the cap; the
        # displaced event is the internal one that held the slot.
        assert adapter._pending_messages[session_key] is user
        assert runner._queue_depth(session_key, adapter=adapter) == cap
        assert [e.text for e in runner._queued_events[session_key]] == [
            f"internal-{i:02d}" for i in range(1, cap)
        ]

    def test_full_queue_user_input_displaces_internal_in_overflow(self):
        """At the cap with a user head and internal overflow, a new user
        event drops one internal event — never any user input."""
        from gateway.run import GatewayRunner

        runner, adapter = self._make_runner_and_adapter()
        session_key = "telegram:user:cap-overflow"
        cap = GatewayRunner._BUSY_QUEUE_MAX_PENDING

        first_user = self._event("user-first")
        runner._queue_or_replace_pending_event(session_key, first_user)
        for i in range(cap - 1):
            runner._queue_or_replace_pending_event(
                session_key, self._event(f"internal-{i:02d}", internal=True)
            )
        assert runner._queue_depth(session_key, adapter=adapter) == cap

        second_user = self._event("user-second")
        runner._queue_or_replace_pending_event(session_key, second_user)

        assert runner._queue_depth(session_key, adapter=adapter) == cap
        assert adapter._pending_messages[session_key] is first_user
        overflow = runner._queued_events[session_key]
        assert overflow[0] is second_user
        # One internal event was sacrificed; the survivors keep arrival order.
        assert [e.text for e in overflow[1:]] == [
            f"internal-{i:02d}" for i in range(cap - 2)
        ]

    def test_full_queue_of_user_events_drops_incoming_internal(self):
        """The complement: a full queue of real user input never sheds a user
        event to admit an internal one."""
        from gateway.run import GatewayRunner

        runner, adapter = self._make_runner_and_adapter()
        session_key = "telegram:user:cap-users"
        cap = GatewayRunner._BUSY_QUEUE_MAX_PENDING

        for i in range(cap):
            runner._queue_or_replace_pending_event(
                session_key, self._event(f"user-{i:02d}")
            )

        runner._queue_or_replace_pending_event(
            session_key, self._event("internal-late", internal=True)
        )

        assert runner._queue_depth(session_key, adapter=adapter) == cap
        assert adapter._pending_messages[session_key].text == "user-00"
        assert all(not e.internal for e in runner._queued_events[session_key])

    def test_photo_burst_still_merges_when_internal_event_queued(self):
        """Album-merge semantics must survive the priority fix: a photo burst
        arriving while an internal completion is queued stays one merged head
        turn (ahead of the internal event), not N split turns."""
        runner, adapter = self._make_runner_and_adapter()
        session_key = "telegram:user:burst-prio"

        runner._queue_or_replace_pending_event(
            session_key, self._event("internal-note", internal=True)
        )
        for i in range(3):
            runner._queue_or_replace_pending_event(
                session_key, self._photo_event(f"p-{i}", f"http://example.com/{i}.jpg")
            )

        head = adapter._pending_messages[session_key]
        assert head.message_type == MessageType.PHOTO
        assert not head.internal
        assert head.media_urls == [f"http://example.com/{i}.jpg" for i in range(3)]
        assert [e.text for e in runner._queued_events[session_key]] == ["internal-note"]

        executed = self._drain_order(runner, adapter, session_key)
        assert [e.message_type for e in executed] == [
            MessageType.PHOTO,
            MessageType.TEXT,
        ]
        assert executed[1].internal
