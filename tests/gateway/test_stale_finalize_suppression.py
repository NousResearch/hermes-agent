"""Regression coverage for #71643 — stale streamed finalize suppression.

A *successful* Telegram finalize edit can carry only the last streamed
preview snapshot: deltas generated between the last preview edit and stream
completion never reach any Bot API call, yet ``final_response_sent`` /
``final_content_delivered`` are set from the call's success and suppress the
gateway's normal final send. The missing tail is then lost with no retry.

These tests exercise the real gateway boundary (``GatewayRunner._run_agent``
with a live ``GatewayStreamConsumer``), per the review guidance on #71643:

1. fake agent emits a visible prefix through ``stream_delta_callback``;
2. the consumer successfully finalizes that prefix;
3. the agent returns a longer ``final_response`` containing a missing tail;
4. the result must NOT silently suppress — the complete final response must
   reach the platform (reconciliation edit or normal final send);
5. control: when the streamed text exactly equals the final text, the
   suppression still occurs (no duplicate delivery).

Plus unit coverage for ``GatewayStreamConsumer.delivered_final_matches``.
"""

import asyncio
import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# ---------------------------------------------------------------------------
# Boundary-test fakes
# ---------------------------------------------------------------------------


class FinalizeCaptureAdapter(BasePlatformAdapter):
    """Adapter that records every send/edit with its finalize flag."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self._next_id = 0

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    def _mint_id(self) -> str:
        self._next_id += 1
        return f"m-{self._next_id}"

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SendResult(success=True, message_id=self._mint_id())

    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None
    ) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "finalize": finalize,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


STREAMED_PREFIX = "The photo shows a dog on a beach"
MISSING_TAIL = " with a red frisbee in its mouth, mid-leap over the surf."
FULL_RESPONSE = STREAMED_PREFIX + MISSING_TAIL


class StalePrefixAgent:
    """Streams only a prefix; the completed response carries a longer tail.

    Models the #71643 incident shape: the tail generated between the last
    preview edit and stream completion never reaches the stream callback, so
    the consumer's successful finalize edit carries stale preview text while
    ``final_response`` holds the complete answer.
    """

    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback(STREAMED_PREFIX)
        return {
            "final_response": FULL_RESPONSE,
            "response_previewed": False,
            "messages": [],
            "api_calls": 1,
        }


class CompleteStreamAgent:
    """Control: the streamed text exactly equals the final response."""

    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback(FULL_RESPONSE)
        return {
            "final_response": FULL_RESPONSE,
            "response_previewed": False,
            "messages": [],
            "api_calls": 1,
        }


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
        streaming=StreamingConfig.from_dict(
            {"enabled": True, "edit_interval": 0.01, "buffer_threshold": 1}
        ),
    )
    return runner


async def _run_streaming_turn(monkeypatch, tmp_path, agent_cls, session_id):
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.dump(
            {
                "display": {"tool_progress": "off", "interim_assistant_messages": False},
                "streaming": {
                    "enabled": True,
                    "edit_interval": 0.01,
                    "buffer_threshold": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = FinalizeCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
    )
    result = await runner._run_agent(
        message="describe this photo",
        context_prompt="",
        history=[],
        source=source,
        session_id=session_id,
        session_key="agent:main:telegram:group:-1001",
    )
    return adapter, result


# ---------------------------------------------------------------------------
# Gateway-boundary regression (#71643)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_finalize_does_not_suppress_complete_response(
    monkeypatch, tmp_path
):
    """The complete response must reach the platform even when the finalize
    edit succeeded with only the stale preview snapshot."""
    adapter, result = await _run_streaming_turn(
        monkeypatch, tmp_path, StalePrefixAgent, "sess-71643-stale-finalize"
    )

    assert result["final_response"] == FULL_RESPONSE
    # The missing tail must appear in at least one platform call — either the
    # reconciliation edit or the normal final send. On the buggy path it
    # appears in NO call at all (message loss).
    all_payloads = [c["content"] for c in adapter.sent] + [
        e["content"] for e in adapter.edits
    ]
    assert any(FULL_RESPONSE in payload for payload in all_payloads), (
        f"complete response never reached the platform; payloads: {all_payloads!r}"
    )
    # The recovery must not duplicate: when the gateway suppressed its normal
    # final send (already_sent), the complete response must have been the
    # payload of the message that finalized on screen — either an in-place
    # reconciliation/finalize edit, or (consumer-declared final contract,
    # 2026-08-16) the primary send itself when the authoritative final was
    # adopted before the first flush. Both shapes are single-message.
    if result.get("already_sent"):
        _edit_carried = any(
            e["content"] == FULL_RESPONSE and e["finalize"] for e in adapter.edits
        )
        _send_carried = any(c["content"] == FULL_RESPONSE for c in adapter.sent)
        assert _edit_carried or _send_carried, (
            "already_sent=True but neither an edit nor the primary send "
            "carried the complete response"
        )


@pytest.mark.asyncio
async def test_equal_text_control_still_suppresses_duplicate_send(
    monkeypatch, tmp_path
):
    """When the streamed text equals the final response, suppression must
    keep working — no duplicate full-response send."""
    adapter, result = await _run_streaming_turn(
        monkeypatch, tmp_path, CompleteStreamAgent, "sess-71643-control-equal"
    )

    assert result["final_response"] == FULL_RESPONSE
    assert result.get("already_sent") is True
    # Exactly one platform message holds the answer: the streamed message
    # (created by one send, then edited). No duplicate full send.
    full_sends = [c for c in adapter.sent if FULL_RESPONSE in c["content"]]
    assert len(full_sends) <= 1, f"duplicate final delivery: {full_sends!r}"


class _PayloadLessSplitConsumer(GatewayStreamConsumer):
    """Force the #78541 shape after a normal stream drain.

    Claims final delivery via the multi-message split path but leaves no
    recorded payload — the pre-fix gateway treated matcher ``None`` as
    legacy trust and swallowed the complete ``final_response``.
    """

    async def run(self):
        await super().run()
        self._final_response_sent = True
        self._final_content_delivered = True
        self._turn_split_delivery = True
        self._delivered_final_text = None
        self._stream_ledger = ""


@pytest.mark.asyncio
async def test_payload_less_split_does_not_suppress_complete_response(
    monkeypatch, tmp_path
):
    """#78541 — payload-less split-delivery flags must not swallow the reply."""
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.dump(
            {
                "display": {"tool_progress": "off", "interim_assistant_messages": False},
                "streaming": {
                    "enabled": True,
                    "edit_interval": 0.01,
                    "buffer_threshold": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = StalePrefixAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    stream_consumer_mod = importlib.import_module("gateway.stream_consumer")
    # run.py imports GatewayStreamConsumer locally inside _run_agent — patch
    # the defining module so the local import picks up the sabotage subclass.
    monkeypatch.setattr(
        stream_consumer_mod, "GatewayStreamConsumer", _PayloadLessSplitConsumer
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    adapter = FinalizeCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1004492624436",
        chat_type="group",
        thread_id="1",
    )
    result = await runner._run_agent(
        message="describe this photo",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-78541-payload-less-split",
        session_key="agent:main:telegram:group:-1004492624436:1",
    )

    assert result["final_response"] == FULL_RESPONSE
    all_payloads = [c["content"] for c in adapter.sent] + [
        e["content"] for e in adapter.edits
    ]
    # The contract is "the complete reply is not swallowed", which has two
    # legitimate shapes: _run_agent puts the full text on the wire itself
    # (reconcile edit), or it declines to claim delivery so the caller's normal
    # final send delivers it. A multi-message split takes the second shape --
    # the reconcile edit is deliberately skipped there because it would repeat
    # every sealed head chunk inside the tail message (#78541). Asserting only
    # the first shape would pin the recovery route rather than the guarantee.
    delivered_here = any(FULL_RESPONSE in payload for payload in all_payloads)
    assert delivered_here or not result.get("already_sent"), (
        "complete response was neither delivered nor left to the normal final "
        f"send; already_sent={result.get('already_sent')!r} payloads={all_payloads!r}"
    )


# ---------------------------------------------------------------------------
# Consumer unit coverage: delivered_final_matches tri-state
# ---------------------------------------------------------------------------


def _consumer():
    adapter = FinalizeCaptureAdapter()
    return GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(cursor=" ▉")
    )


class TestDeliveredFinalMatches:
    def test_no_record_returns_none(self):
        consumer = _consumer()
        assert consumer.delivered_final_matches("anything") is None

    def test_matching_record_returns_true(self):
        consumer = _consumer()
        consumer._record_turn_final_payload(FULL_RESPONSE)
        assert consumer.delivered_final_matches(FULL_RESPONSE) is True

    def test_stale_prefix_record_returns_false(self):
        consumer = _consumer()
        consumer._record_turn_final_payload(STREAMED_PREFIX)
        assert consumer.delivered_final_matches(FULL_RESPONSE) is False

    def test_payload_less_split_delivery_returns_false(self):
        """#78541 — payload-less split must not inherit legacy trust."""
        consumer = _consumer()
        consumer._turn_split_delivery = True
        consumer._delivered_final_text = None
        assert consumer.delivered_final_matches(FULL_RESPONSE) is False

    def test_split_delivery_with_matching_ledger_returns_true(self):
        """Complete overflow split that recorded its ledger still suppresses."""
        consumer = _consumer()
        consumer._turn_split_delivery = True
        consumer._stream_ledger = FULL_RESPONSE
        consumer._record_turn_final_payload(STREAMED_PREFIX)  # tail only; ledger wins
        assert consumer.delivered_final_matches(FULL_RESPONSE) is True

    def test_split_delivery_with_stale_ledger_returns_false(self):
        consumer = _consumer()
        consumer._turn_split_delivery = True
        consumer._stream_ledger = STREAMED_PREFIX
        consumer._record_turn_final_payload(STREAMED_PREFIX)
        assert consumer.delivered_final_matches(FULL_RESPONSE) is False

    def test_empty_final_text_returns_none(self):
        consumer = _consumer()
        consumer._record_turn_final_payload(STREAMED_PREFIX)
        assert consumer.delivered_final_matches("") is None

    def test_segment_delivered_text_still_matches(self):
        consumer = _consumer()
        consumer._record_turn_final_payload(STREAMED_PREFIX)
        # A prior segment delivered the exact final text.
        consumer._delivered_segment_texts.append(FULL_RESPONSE)
        assert consumer.delivered_final_matches(FULL_RESPONSE) is True

    def test_reset_segment_state_clears_record(self):
        consumer = _consumer()
        consumer._record_turn_final_payload(STREAMED_PREFIX)
        consumer._reset_segment_state()
        assert consumer._delivered_final_text is None
        assert consumer._turn_split_delivery is False
        assert consumer._stream_ledger == ""


# ---------------------------------------------------------------------------
# End-to-end split delivery: drive the real overflow-split loop (no hand-set
# private flags) and assert the no-duplicate / no-swallow contract on both
# sides.  These are the shapes that #78541's boundary fix has to not regress:
# a genuine complete split must still suppress, and an incomplete one must not.
# ---------------------------------------------------------------------------


class _SplittingAdapter(FinalizeCaptureAdapter):
    """Small message cap so real prose trips the consumer's overflow split.

    Also records deletions and can be told to fail edits, so the fresh-final
    and flood-control paths can be driven without patching internals.
    """

    MAX_MESSAGE_LENGTH = 220

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform)
        self.deleted = []
        self.fail_edits = False

    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None
    ) -> SendResult:
        if self.fail_edits:
            return SendResult(
                success=False, error="Flood control exceeded. Retry in 12 seconds"
            )
        return await super().edit_message(
            chat_id, message_id, content, finalize=finalize, metadata=metadata
        )

    async def delete_message(self, chat_id, message_id) -> bool:
        self.deleted.append(message_id)
        return True

    def truncate_message(self, text, limit, len_fn=len):
        chunks, rest = [], text
        while len_fn(rest) > limit:
            cut = rest.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = limit
            chunks.append(rest[:cut])
            rest = rest[cut:].lstrip("\n")
        chunks.append(rest)
        return chunks


def _split_consumer():
    adapter = _SplittingAdapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat-split",
        StreamConsumerConfig(
            edit_interval=0.0, buffer_threshold=1, cursor="",
            fresh_final_after_seconds=0.0,
        ),
    )
    return adapter, consumer


async def _drain_split_turn(consumer, lines):
    task = asyncio.create_task(consumer.run())
    for line in lines:
        consumer.on_delta(line + "\n")
        await asyncio.sleep(0.005)
    consumer.finish()
    await asyncio.wait_for(task, timeout=10)


@pytest.mark.asyncio
async def test_complete_overflow_split_still_suppresses_duplicate():
    """A fully delivered multi-message reply must NOT be re-sent (#45517)."""
    adapter, consumer = _split_consumer()
    lines = [f"line {i} " + "x" * 60 for i in range(12)]
    await _drain_split_turn(consumer, lines)

    complete = "\n".join(lines)
    assert consumer._turn_split_delivery is True, "overflow split never triggered"
    # The user received the whole answer across several messages, so the
    # gateway must keep suppressing its own final send.
    assert consumer.delivered_final_matches(complete) is True


@pytest.mark.asyncio
async def test_split_delivery_missing_tail_does_not_suppress():
    """#78541 — when the completed response exceeds what the split delivered,
    the matcher must report a mismatch so the gateway still sends it."""
    adapter, consumer = _split_consumer()
    lines = [f"line {i} " + "x" * 60 for i in range(12)]
    await _drain_split_turn(consumer, lines)

    never_streamed = "\n".join(lines) + "\n\n" + "tail the user never saw " * 8
    assert consumer._turn_split_delivery is True
    assert consumer.delivered_final_matches(never_streamed) is False


@pytest.mark.asyncio
async def test_split_delivery_keeps_sealed_heads_on_fresh_final():
    """The fresh-final route must not delete sealed head messages.

    ``_try_fresh_final`` replaces every tracked preview with one fresh message,
    which only holds the whole answer on a single-message turn.  After a split
    the sealed heads carry text that the fresh message does not, so deleting
    them would drop delivered content (#78541).
    """
    adapter, consumer = _split_consumer()
    head_id = await consumer._send_new_chunk("HEAD text. " * 12, None, final=False)
    consumer._turn_split_delivery = True

    assert head_id in consumer._preview_message_ids
    assert await consumer._try_fresh_final("TAIL text. " * 12) is False
    assert head_id not in adapter.deleted, "sealed head chunk was deleted"


@pytest.mark.asyncio
async def test_failed_final_edit_after_split_records_visible_payload():
    """A flood-controlled cosmetic final edit must not cause a duplicate.

    The complete answer is already on screen; only the cursor-strip edit
    failed.  Recording the visible payload keeps the gateway suppressing its
    own send instead of posting the whole answer twice (#36965 / #25349).
    """
    adapter = _SplittingAdapter()
    cursor = " \u2589"
    consumer = GatewayStreamConsumer(
        adapter,
        "chat-split",
        StreamConsumerConfig(
            edit_interval=0.0, buffer_threshold=1, cursor=cursor,
            fresh_final_after_seconds=0.0,
        ),
    )
    full = "The complete answer. " * 8
    # Streaming already put the whole answer on screen, cursor and all.
    await consumer._send_or_edit(full + cursor)
    assert consumer._last_sent_text.endswith(cursor)
    consumer._turn_split_delivery = True
    consumer._stream_ledger = full
    adapter.fail_edits = True

    # The cosmetic cursor-strip edit is rate-limited and fails.
    assert await consumer._send_or_edit(full, finalize=True) is False
    assert consumer._final_content_delivered is True
    assert consumer.delivered_final_matches(full) is True


@pytest.mark.asyncio
async def test_empty_fallback_final_after_split_records_only_what_survives():
    """A recovery that DELETES the sealed heads must not claim them as delivered.

    ``_send_empty_fallback_final`` replaces the active segment: it sends the
    completed text as a fresh message and deletes every tracked segment
    preview -- including the sealed head chunks of an overflow split.  Only the
    new message is left on screen, so the recorded payload must be that message
    verbatim, NOT the stream ledger.  Recording the ledger would claim delivery
    for text this path just removed, and the gateway would suppress its own
    send and leave the user with a fraction of the answer (#78541).
    """
    adapter = _SplittingAdapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat-split",
        StreamConsumerConfig(
            edit_interval=0.0, buffer_threshold=1, cursor="",
            fresh_final_after_seconds=0.0,
        ),
    )
    head = "HEAD text. " * 40
    tail = "TAIL text. " * 6
    complete = head + tail

    # Sealed head chunk is on screen and tracked as a segment preview.
    head_id = await consumer._send_new_chunk(head, None, final=False)
    consumer._turn_split_delivery = True
    consumer._stream_ledger = complete
    assert head_id in consumer._segment_preview_message_ids

    # The recovery commits only ``tail`` and deletes the sealed head.
    assert await consumer._send_empty_fallback_final(tail) == "delivered"
    assert head_id in adapter.deleted, "expected the recovery to delete the head"

    # The head is gone from the chat, so the complete answer was NOT delivered:
    # the gateway must be told this is a mismatch and send it.
    assert consumer.delivered_final_matches(complete) is False
