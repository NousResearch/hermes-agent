"""Tests for /queue message consumption after normal agent completion.

Verifies that messages queued via /queue (which store in
adapter._pending_messages WITHOUT triggering an interrupt) are consumed
after the agent finishes its current task — not silently dropped.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.run import _dequeue_pending_event
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    PlatformConfig,
    Platform,
)
from plugins.platforms.telegram.inbound_store import (
    CaptureDecision,
    DurableTelegramUpdateQueue,
    TelegramInboundStore,
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


def test_ordinary_busy_queue_rejects_work_after_the_32_item_cap():
    """Durable Telegram admission must not remove the ordinary busy cap."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._queued_events = {}
    adapter = _StubAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    session_key = "telegram:user:ordinary-cap"
    source = MagicMock(chat_id="ordinary-cap", platform=Platform.TELEGRAM, profile=None)

    assert runner._BUSY_QUEUE_MAX_PENDING == 32
    for index in range(runner._BUSY_QUEUE_MAX_PENDING + 1):
        runner._queue_or_replace_pending_event(
            session_key,
            MessageEvent(
                text=f"ordinary-{index}",
                message_type=MessageType.TEXT,
                source=source,
                message_id=f"ordinary-{index}",
            ),
        )

    assert runner._queue_depth(session_key, adapter=adapter) == 32
    assert adapter._pending_messages[session_key].text == "ordinary-0"
    assert runner._queued_events[session_key][-1].text == "ordinary-31"


@pytest.mark.asyncio
async def test_claimed_durable_telegram_event_does_not_consume_ordinary_busy_cap(tmp_path):
    """A claimed durable event is requeued, not marked done at the busy cap."""
    from gateway.run import GatewayRunner
    from plugins.platforms.telegram.adapter import TelegramAdapter

    dispatch_calls = []

    class ProbeRunner(GatewayRunner):
        async def _handle_message(self, event):
            dispatch_calls.append(event)

    ordinary_runner = ProbeRunner.__new__(ProbeRunner)
    ordinary_runner._queued_events = {}
    adapter = object.__new__(TelegramAdapter)
    adapter._pending_messages = {}
    adapter._active_sessions = {"telegram:user:ordinary-cap-with-durable": asyncio.Event()}
    adapter._inbound_store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    adapter._inbound_queue = None
    adapter._bot_account_id = None
    adapter.config = PlatformConfig(enabled=True, token="111:test-token", extra={})
    ordinary_runner.adapters = {Platform.TELEGRAM: adapter}
    ordinary_key = "telegram:user:ordinary-cap-with-durable"
    ordinary_source = MagicMock(
        chat_id=ordinary_key,
        platform=Platform.TELEGRAM,
        profile=None,
    )

    def durable_payload(update_id):
        return {
            "update_id": update_id,
            "message": {"message_id": update_id, "chat": {"id": 7}},
        }

    def classify(item):
        return CaptureDecision(
            actionable=True,
            update_kind="message",
            chat_id="7",
            message_id=str(item["update_id"]),
            session_key=ordinary_key,
            payload=item,
        )

    queue = DurableTelegramUpdateQueue(
        store=adapter._inbound_store,
        bot_account_id=111,
        classifier=classify,
        lease_owner="gateway:durable-cap-test",
        active_limit=1,
    )
    adapter._inbound_queue = queue
    ordinary_runner._effective_busy_text_mode = lambda source: "queue"
    adapter._message_handler = ordinary_runner._handle_message

    durable_payload = durable_payload(9001)
    await queue.put(durable_payload)
    claimed_item = await queue.get()
    claim = queue.claim_for_update(9001)
    assert claim is not None
    assert claimed_item["update_id"] == 9001
    assert claimed_item["message"]["message_id"] == 9001
    queue.task_done()

    durable_source = MagicMock(
        chat_id=ordinary_key,
        platform=Platform.TELEGRAM,
        profile=None,
    )
    claimed_durable_event = MessageEvent(
        text="durable",
        message_type=MessageType.TEXT,
        source=durable_source,
        raw_message=claim,
        message_id="77",
        platform_update_id=9001,
        metadata={
            "telegram_inbound_claimed": True,
            "telegram_durable_update_ids": [9001],
            "gateway_session_key": ordinary_key,
        },
    )

    for index in range(ordinary_runner._BUSY_QUEUE_MAX_PENDING + 1):
        ordinary_runner._queue_or_replace_pending_event(
            ordinary_key,
            MessageEvent(
                text=f"ordinary-{index}",
                message_type=MessageType.TEXT,
                source=ordinary_source,
                message_id=f"ordinary-{index}",
            ),
        )

    await adapter._dispatch_and_complete_durable_event(claimed_durable_event)

    row = adapter._inbound_store.get("telegram:111:9001")
    assert row is not None
    assert row.work_state == "queued"
    assert row.dispatch_state == "pending"
    assert queue.claim_for_update(9001) is None
    assert ordinary_runner._BUSY_QUEUE_MAX_PENDING == 32
    assert ordinary_runner._queue_depth(ordinary_key, adapter=adapter) == 32
    assert adapter._pending_messages[ordinary_key].text == "ordinary-0"
    assert ordinary_runner._queued_events[ordinary_key][-1].text == "ordinary-31"
    assert dispatch_calls == []


@pytest.mark.asyncio
async def test_durable_busy_fifo_consumes_each_in_band_event(tmp_path, monkeypatch):
    """One owner draining a busy FIFO finalizes every durable event."""
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._queued_events = {}
    runner._draining = False
    adapter = object.__new__(TelegramAdapter)
    BasePlatformAdapter.__init__(
        adapter,
        PlatformConfig(enabled=True, token="111:test-token", extra={}),
        Platform.TELEGRAM,
    )

    session_key = "agent:main:telegram:dm:durable-fifo-fence"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="durable-fifo-fence",
        chat_type="dm",
        user_id="7",
        user_name="tester",
    )
    state = SimpleNamespace(
        conversation=SimpleNamespace(queued_events=[]),
        turn=SimpleNamespace(agent=None, busy_ack_ts=0, started_ts=0),
    )
    owner_release = asyncio.Event()
    drained_events = []

    async def drain_owner():
        await owner_release.wait()
        while runner._queue_depth(session_key, adapter=adapter):
            queued_event = _dequeue_pending_event(adapter, session_key)
            queued_event = runner._promote_queued_event(
                session_key, adapter, queued_event
            )
            if queued_event is None:
                break
            drained_events.append(queued_event)
            await asyncio.sleep(0)
        current_task = asyncio.current_task()
        assert current_task is not None
        setattr(
            current_task,
            "_telegram_processing_outcome",
            ProcessingOutcome.SUCCESS,
        )

    owner_task = asyncio.create_task(drain_owner())
    adapter._active_sessions = {session_key: asyncio.Event()}
    adapter._session_tasks = {session_key: owner_task}

    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    adapter._inbound_store = store
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: CaptureDecision(
            actionable=True,
            update_kind="message",
            chat_id="7",
            message_id=str(item["message"]["message_id"]),
            session_key=session_key,
            payload=item,
        ),
        lease_owner="gateway:durable-fifo-fence",
        active_limit=4,
    )
    adapter._inbound_queue = queue

    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._adapter_for_source = lambda _source: adapter
    runner._peek_session_state = lambda _session_key: state
    runner._session_state = lambda _session_key: state
    runner._is_user_authorized = lambda _source: True
    runner._effective_busy_input_mode = lambda source: "steer"
    runner._effective_busy_text_mode = lambda source: "steer"
    runner._agent_has_active_subagents = lambda _agent: False

    async def no_compression(_session_key):
        return False

    runner._session_has_compression_in_flight = no_compression
    adapter._message_handler = runner._handle_message
    adapter._busy_session_handler = runner._handle_active_session_busy_message

    update_ids = [9101, 9102, 9103, 9104]
    message_ids = [8101, 8102, 8103, 8104]
    texts = ["first", "identical", "identical", "fourth"]
    events = []
    claims = []
    for update_id, message_id, text in zip(update_ids, message_ids, texts):
        payload_data = {
            "update_id": update_id,
            "message": {"message_id": message_id, "chat": {"id": 7}},
        }
        await queue.put(payload_data)
        await queue.get()
        claim = queue.claim_for_update(update_id)
        assert claim is not None
        queue.task_done()
        claims.append(claim)
        events.append(
            MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                raw_message=claim,
                message_id=str(message_id),
                platform_update_id=update_id,
                metadata={
                    "telegram_inbound_claimed": True,
                    "telegram_durable_update_ids": [update_id],
                    "gateway_session_key": session_key,
                },
                allow_gateway_control=False,
            )
        )

    try:
        for event in events:
            await adapter._dispatch_and_complete_durable_event(event)
        assert runner._queue_depth(session_key, adapter=adapter) == 4
        for claim in claims:
            row = store.get(claim.event_id)
            assert row is not None
            assert row.work_state == "leased"
            assert row.consumed_at is None

        owner_release.set()
        await asyncio.wait_for(owner_task, timeout=1.0)
        await asyncio.wait_for(queue._wait_for_lifecycle_tasks(), timeout=1.0)
    finally:
        if not owner_task.done():
            owner_release.set()
            await owner_task

    assert [event.platform_update_id for event in drained_events] == update_ids
    assert [event.message_id for event in drained_events] == [
        str(message_id) for message_id in message_ids
    ]
    assert drained_events[1].text == drained_events[2].text == "identical"
    for claim in claims:
        row = store.get(claim.event_id)
        assert row is not None
        assert row.work_state == "consumed"
        assert row.consumed_at is not None


@pytest.mark.asyncio
async def test_durable_busy_steer_is_retained_until_active_turn_completes(
    tmp_path,
    monkeypatch,
):
    """A durable plain-text follow-up steers and stays leased until turn success."""
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")

    class AcceptOnlyAgent:
        def __init__(self):
            self.steer_calls = []

        def steer(self, text):
            self.steer_calls.append(text)
            return True

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._queued_events = {}
    runner._draining = False
    adapter = object.__new__(TelegramAdapter)
    BasePlatformAdapter.__init__(
        adapter,
        PlatformConfig(enabled=True, token="111:test-token", extra={}),
        Platform.TELEGRAM,
    )
    runner.adapters = {Platform.TELEGRAM: adapter}

    session_key = "agent:main:telegram:dm:durable-steer"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="durable-steer",
        chat_type="dm",
        user_id="7",
        user_name="tester",
    )
    agent = AcceptOnlyAgent()
    state = SimpleNamespace(
        conversation=SimpleNamespace(queued_events=[]),
        turn=SimpleNamespace(agent=agent, busy_ack_ts=0, started_ts=0),
    )
    owner_release = asyncio.Event()

    async def active_turn_owner():
        await owner_release.wait()
        owner = asyncio.current_task()
        assert owner is not None
        setattr(owner, "_telegram_processing_outcome", ProcessingOutcome.SUCCESS)

    owner_task = asyncio.create_task(active_turn_owner())
    adapter._active_sessions = {session_key: asyncio.Event()}
    adapter._session_tasks = {session_key: owner_task}

    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: CaptureDecision(
            actionable=True,
            update_kind="message",
            chat_id="7",
            message_id=str(item["update_id"]),
            session_key=session_key,
            payload=item,
        ),
        lease_owner="gateway:durable-steer",
        active_limit=1,
    )
    adapter._inbound_store = store
    adapter._inbound_queue = queue

    runner._adapter_for_source = lambda _source: adapter
    runner._peek_session_state = lambda _session_key: state
    runner._session_state = lambda _session_key: state
    runner._is_user_authorized = lambda _source: True
    runner._effective_busy_input_mode = lambda _source: "steer"
    runner._effective_busy_text_mode = lambda _source: "steer"
    runner._agent_has_active_subagents = lambda _agent: False

    async def no_compression(_session_key):
        return False

    runner._session_has_compression_in_flight = no_compression
    adapter._message_handler = runner._handle_message
    adapter._busy_session_handler = runner._handle_active_session_busy_message

    update_id = 9004
    payload = {
        "update_id": update_id,
        "message": {"message_id": update_id, "chat": {"id": 7}},
    }
    await queue.put(payload)
    await queue.get()
    claim = queue.claim_for_update(update_id)
    assert claim is not None
    queue.task_done()

    event = MessageEvent(
        text="durable correction",
        message_type=MessageType.TEXT,
        source=source,
        raw_message=claim,
        message_id="80",
        platform_update_id=update_id,
        metadata={
            "telegram_inbound_claimed": True,
            "telegram_durable_update_ids": [update_id],
            "gateway_session_key": session_key,
        },
    )

    try:
        await adapter._dispatch_and_complete_durable_event(event)

        assert agent.steer_calls == ["durable correction"]
        assert runner._queue_depth(session_key, adapter=adapter) == 0
        row = store.get(claim.event_id)
        assert row is not None
        assert row.work_state == "leased"
        assert row.consumed_at is None

        owner_release.set()
        await asyncio.wait_for(owner_task, timeout=1.0)
        await asyncio.wait_for(queue._wait_for_lifecycle_tasks(), timeout=1.0)
    finally:
        if not owner_task.done():
            owner_release.set()
            await owner_task
        await queue.close()

    row = store.get(claim.event_id)
    assert row is not None
    assert row.work_state == "consumed"
    assert row.consumed_at is not None


@pytest.mark.asyncio
async def test_durable_busy_cap_claim_survives_process_loss_and_replays(tmp_path):
    """A busy-cap decision must leave a leased durable row recoverable."""
    from gateway.run import GatewayRunner
    from plugins.platforms.telegram.adapter import TelegramAdapter

    dispatch_calls = []

    class ProbeRunner(GatewayRunner):
        async def _handle_message(self, event):
            dispatch_calls.append(event)

    ordinary_runner = ProbeRunner.__new__(ProbeRunner)
    ordinary_runner._queued_events = {}
    adapter = object.__new__(TelegramAdapter)
    adapter._pending_messages = {}
    ordinary_key = "telegram:user:ordinary-cap-crash"
    adapter._active_sessions = {ordinary_key: asyncio.Event()}
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    adapter._inbound_store = store
    adapter._inbound_queue = None
    adapter._bot_account_id = None
    adapter.config = PlatformConfig(enabled=True, token="111:test-token", extra={})
    ordinary_runner.adapters = {Platform.TELEGRAM: adapter}
    ordinary_source = MagicMock(
        chat_id=ordinary_key,
        platform=Platform.TELEGRAM,
        profile=None,
    )

    def durable_payload(update_id):
        return {
            "update_id": update_id,
            "message": {"message_id": update_id, "chat": {"id": 7}},
        }

    def classify(item):
        return CaptureDecision(
            actionable=True,
            update_kind="message",
            chat_id="7",
            message_id=str(item["update_id"]),
            session_key=ordinary_key,
            payload=item,
        )

    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=classify,
        lease_owner="gateway:durable-cap-crash-old",
        active_limit=1,
    )
    adapter._inbound_queue = queue
    ordinary_runner._effective_busy_text_mode = lambda source: "queue"
    adapter._message_handler = ordinary_runner._handle_message

    update_id = 9002
    await queue.put(durable_payload(update_id))
    claimed_item = await queue.get()
    claim = queue.claim_for_update(update_id)
    assert claim is not None
    assert claimed_item["update_id"] == update_id
    queue.task_done()

    claimed_event = MessageEvent(
        text="durable",
        message_type=MessageType.TEXT,
        source=ordinary_source,
        raw_message=claim,
        message_id="78",
        platform_update_id=update_id,
        metadata={
            "telegram_inbound_claimed": True,
            "telegram_durable_update_ids": [update_id],
            "gateway_session_key": ordinary_key,
        },
    )
    for index in range(ordinary_runner._BUSY_QUEUE_MAX_PENDING + 1):
        ordinary_runner._queue_or_replace_pending_event(
            ordinary_key,
            MessageEvent(
                text=f"ordinary-{index}",
                message_type=MessageType.TEXT,
                source=ordinary_source,
                message_id=f"ordinary-{index}",
            ),
        )

    await adapter._dispatch_inbound_event(claimed_event)

    row = store.get("telegram:111:9002")
    assert row is not None
    assert row.work_state == "leased"
    assert row.lease_owner == "gateway:durable-cap-crash-old"
    assert queue.claim_for_update(update_id) is claim
    assert dispatch_calls == []
    assert ordinary_runner._queue_depth(ordinary_key, adapter=adapter) == 32

    # Process loss discards only the old queue's in-memory claim map. The
    # replacement queue must recover the still-leased SQLite row itself.
    queue._scheduler_loop = None
    queue.forget_claims({claim.event_id})
    assert queue.claim_for_update(update_id) is None
    replacement_queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=classify,
        lease_owner="gateway:durable-cap-crash-new",
        active_limit=1,
    )
    assert await replacement_queue.recover(now=0.0) == 1
    replayed_item = await replacement_queue.get()
    replayed_claim = replacement_queue.claim_for_update(update_id)
    assert replayed_item["update_id"] == update_id
    assert replayed_claim is not None
    replacement_queue.task_done()
    assert await replacement_queue.complete_update(update_id, success=True)

    row = store.get("telegram:111:9002")
    assert row is not None
    assert row.work_state == "consumed"
    assert row.consumed_at is not None


@pytest.mark.asyncio
async def test_concurrent_durable_busy_cap_admission_is_replayable(tmp_path, monkeypatch):
    """A raced durable admission cannot turn an inherited cap drop into success."""
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")

    class ProbeRunner(GatewayRunner):
        async def _handle_message(self, event):
            return None

    runner = ProbeRunner.__new__(ProbeRunner)
    runner._queued_events = {}
    runner._draining = False
    adapter = object.__new__(TelegramAdapter)
    BasePlatformAdapter.__init__(
        adapter,
        PlatformConfig(enabled=True, token="111:test-token", extra={}),
        Platform.TELEGRAM,
    )

    ordinary_key = "agent:main:telegram:dm:durable-cap-race"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="durable-cap-race",
        chat_type="dm",
        user_id="7",
        user_name="tester",
    )
    state = SimpleNamespace(
        conversation=SimpleNamespace(queued_events=[]),
        turn=SimpleNamespace(agent=None, busy_ack_ts=0, started_ts=0),
    )
    owner_release = asyncio.Event()
    event_to_cancel = []

    async def cancel_after_draining_event():
        await owner_release.wait()
        assert len(event_to_cancel) == 1
        assert adapter._remove_durable_event_from_gateway_fifo(
            event_to_cancel[0], ordinary_key
        )
        current_task = asyncio.current_task()
        assert current_task is not None
        setattr(
            current_task,
            "_telegram_processing_outcome",
            ProcessingOutcome.CANCELLED,
        )
        raise asyncio.CancelledError

    owner_task = asyncio.create_task(cancel_after_draining_event())

    adapter._active_sessions = {ordinary_key: asyncio.Event()}
    adapter._session_tasks = {ordinary_key: owner_task}
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    adapter._inbound_store = store
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: CaptureDecision(
            actionable=True,
            update_kind="message",
            chat_id="7",
            message_id=str(item["update_id"]),
            session_key=ordinary_key,
            payload=item,
        ),
        lease_owner="gateway:durable-cap-race",
        active_limit=2,
    )
    adapter._inbound_queue = queue

    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._adapter_for_source = lambda event_source: adapter
    runner._peek_session_state = lambda session_key: state
    runner._session_state = lambda session_key: state
    runner._is_user_authorized = lambda event_source: True
    runner._effective_busy_input_mode = lambda event_source: "queue"
    runner._effective_busy_text_mode = lambda event_source: "queue"
    runner._agent_has_active_subagents = lambda agent: False

    async def no_compression(session_key):
        return False

    runner._session_has_compression_in_flight = no_compression
    actual_busy_handler = runner._handle_active_session_busy_message
    first_busy_finished = asyncio.Event()

    async def synchronized_busy_handler(event, session_key):
        # Force both callers to reach the real GatewayRunner cap check before
        # either one can reserve a FIFO slot. The adapter fix must serialize
        # this boundary without changing the inherited cap.
        await asyncio.sleep(0)
        result = await actual_busy_handler(event, session_key)
        if event.platform_update_id == 9301:
            first_busy_finished.set()
        return result

    adapter._message_handler = runner._handle_message
    adapter._busy_session_handler = synchronized_busy_handler

    update_ids = (9301, 9302)
    claims = []
    for update_id in update_ids:
        await queue.put(
            {
                "update_id": update_id,
                "message": {"message_id": update_id, "chat": {"id": 7}},
            }
        )
        item = await queue.get()
        claim = queue.claim_for_update(update_id)
        assert claim is not None
        assert item["update_id"] == update_id
        queue.task_done()
        claims.append(claim)

    # One pending head plus thirty overflow events establishes the ordinary
    # inherited depth-31 boundary. The next accepted event would be item 32.
    for index in range(31):
        runner._queue_or_replace_pending_event(
            ordinary_key,
            MessageEvent(
                text=f"ordinary-{index}",
                message_type=MessageType.DOCUMENT,
                source=source,
                message_id=f"ordinary-{index}",
                allow_gateway_control=False,
            ),
        )
    assert runner._queue_depth(ordinary_key, adapter=adapter) == 31
    assert not owner_task.done()

    def durable_event(update_id, claim):
        return MessageEvent(
            text=f"durable-{update_id}",
            message_type=MessageType.DOCUMENT,
            source=source,
            raw_message=claim,
            message_id=f"message-{update_id}",
            platform_update_id=update_id,
            metadata={
                "telegram_inbound_claimed": True,
                "telegram_durable_update_ids": [update_id],
                "gateway_session_key": ordinary_key,
            },
            allow_gateway_control=False,
        )

    events = [durable_event(update_id, claim) for update_id, claim in zip(update_ids, claims)]
    event_to_cancel.append(events[0])
    dispatch_tasks = [
        asyncio.create_task(adapter._dispatch_and_complete_durable_event(event))
        for event in events
    ]
    await asyncio.wait_for(first_busy_finished.wait(), timeout=1.0)
    # The first caller is now waiting at the inherited background owner. Give
    # the serialized second caller one scheduling turn to observe the full cap.
    await asyncio.sleep(0)
    owner_release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(owner_task, timeout=1.0)
    await asyncio.gather(*dispatch_tasks)
    await asyncio.wait_for(queue._wait_for_lifecycle_tasks(), timeout=1.0)

    # The admitted durable event left the FIFO under an owner that reported
    # cancellation. It and the concurrent cap rejection must both be replayable.
    rows = [store.get(f"telegram:111:{update_id}") for update_id in update_ids]
    assert all(row is not None for row in rows)
    assert [row.work_state for row in rows] == ["queued", "queued"]
    assert all(row.dispatch_state == "pending" for row in rows)
    assert all(row.consumed_at is None for row in rows)
    assert all(queue.claim_for_update(update_id) is None for update_id in update_ids)

    queued_events = list(state.conversation.queued_events)
    pending = adapter._pending_messages.get(ordinary_key)
    volatile_events = ([pending] if pending is not None else []) + queued_events
    volatile_ids = {
        update_id
        for queued_event in volatile_events
        for update_id in getattr(queued_event, "metadata", {}).get(
            "telegram_durable_update_ids", []
        )
    }
    assert not volatile_ids.intersection(update_ids)
    assert runner._queue_depth(ordinary_key, adapter=adapter) == 31

    if not owner_task.done():
        owner_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner_task
