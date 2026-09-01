"""Regression coverage for #95382 / #98552 — silent partial delivery.

#95382 (Discord): the WebSocket drops after the first streaming edit (which
carried only a prefix). The consumer's delivery flags could suppress the
gateway's normal final send even though no recorded payload proved the
COMPLETE ``final_response`` ever reached the platform; and when the normal
final send then failed on the dead transport, the failure was recorded with a
non-retryable error string, so the delivery-obligation ledger's reconnect
sweep never replayed it — the turn's output was silently lost until a full
process restart.

#98552 (Telegram): a finalize path that sets ``final_content_delivered=True``
without recording what was actually delivered produced the same false
positive on a 624-char message truncated at 333 chars.

Class contract under test:

1. ``delivered_final_matches`` judges a payload-less delivery flag against
   the FINAL content (via ``has_delivered_text``) instead of returning the
   legacy-trust ``None`` — only the explicitly-marked ambiguous-timeout path
   keeps legacy trust.
2. Every flag-setting site records its delivered payload (fresh-final and
   the optimistic native finalize were the record-less holdouts).
3. Discord transport-shaped send failures are classified as
   ``send_path_degraded`` (retryable) so the ledger reconnect sweep can
   replay the stranded final response.

Boundary tests drive the REAL ``GatewayRunner._run_agent`` with a live
``GatewayStreamConsumer`` (pattern from test_stale_finalize_suppression.py).
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


STREAMED_PREFIX = "Deploy summary: 713 items published (578 as of 08-26"
MISSING_TAIL = ", another 135 over the past 4 days). All checks green."
FULL_RESPONSE = STREAMED_PREFIX + MISSING_TAIL


# ---------------------------------------------------------------------------
# Unit coverage — delivered_final_matches tri-state tightening
# ---------------------------------------------------------------------------


def _make_consumer(adapter=None, **overrides):
    adapter = adapter or SimpleNamespace(
        MAX_MESSAGE_LENGTH=4096,
        splits_long_messages=True,
    )
    consumer = GatewayStreamConsumer.__new__(GatewayStreamConsumer)
    consumer.adapter = adapter
    consumer.chat_id = "c1"
    consumer.cfg = StreamConsumerConfig(cursor="▉")
    consumer._final_response_sent = True
    consumer._final_content_delivered = True
    consumer._delivered_final_text = None
    consumer._turn_split_delivery = False
    consumer._delivery_ambiguous = False
    consumer._delivered_commentary_texts = []
    consumer._delivered_segment_texts = []
    consumer._last_sent_text = ""
    consumer._accumulated = ""
    consumer._stream_ledger = ""
    consumer._initial_reply_to_id = None
    consumer.metadata = None
    for key, value in overrides.items():
        setattr(consumer, key, value)
    return consumer


class TestDeliveredFinalMatchesRecordless:
    def test_recordless_flag_with_partial_visible_is_mismatch(self):
        """#95382 core: flag set, no record, visible text is only a prefix —
        the matcher must return False (recover), not None (legacy trust)."""
        consumer = _make_consumer(_last_sent_text=STREAMED_PREFIX + "▉")
        assert consumer.delivered_final_matches(FULL_RESPONSE) is False

    def test_recordless_flag_with_no_visible_text_is_mismatch(self):
        """Flag set but nothing visibly delivered at all — mismatch."""
        consumer = _make_consumer()
        assert consumer.delivered_final_matches(FULL_RESPONSE) is False

    def test_recordless_flag_with_equal_visible_text_matches(self):
        """Duplicate-suppression control: the visible text IS the final
        answer — suppression must be retained (True)."""
        consumer = _make_consumer(_last_sent_text=FULL_RESPONSE + "▉")
        assert consumer.delivered_final_matches(FULL_RESPONSE) is True

    def test_ambiguous_timeout_keeps_legacy_trust(self):
        """The explicitly-marked ambiguous full-final timeout is the ONE
        record-less case that keeps legacy trust (None) — re-sending there
        risks a duplicate, not a recovery."""
        consumer = _make_consumer(_delivery_ambiguous=True)
        assert consumer.delivered_final_matches(FULL_RESPONSE) is None

    def test_recorded_payload_still_wins_over_visible(self):
        consumer = _make_consumer(
            _delivered_final_text=FULL_RESPONSE,
            _last_sent_text="something else entirely",
        )
        assert consumer.delivered_final_matches(FULL_RESPONSE) is True

    def test_payloadless_split_still_refuses_trust(self):
        """#78541 behavior preserved by the tightening."""
        consumer = _make_consumer(_turn_split_delivery=True)
        assert consumer.delivered_final_matches(FULL_RESPONSE) is False

    def test_delivered_segment_text_matches(self):
        """A segment-finalized delivery of the final text still suppresses."""
        consumer = _make_consumer(
            _delivered_segment_texts=[FULL_RESPONSE],
        )
        assert consumer.delivered_final_matches(FULL_RESPONSE) is True


class TestFlagSettingSitesRecordPayload:
    @pytest.mark.asyncio
    async def test_fresh_final_records_delivered_payload(self):
        """_try_fresh_final must record what it sent (#95382 holdout)."""

        class FreshAdapter:
            MAX_MESSAGE_LENGTH = 4096
            splits_long_messages = True

            def __init__(self):
                self.sent = []

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                self.sent.append(content)
                return SendResult(success=True, message_id="m-1")

        adapter = FreshAdapter()
        consumer = _make_consumer(adapter)
        consumer._final_response_sent = False
        consumer._final_content_delivered = False
        consumer._preview_message_ids = set()
        consumer._message_id = "m-0"
        consumer._message_created_ts = None
        consumer.metadata = None
        consumer._already_sent = False

        ok = await consumer._try_fresh_final(STREAMED_PREFIX, is_turn_final=True)
        assert ok is True
        assert consumer._final_response_sent is True
        # The recorded payload lets the gateway detect a stale fresh-final.
        assert consumer._delivered_final_text is not None
        assert STREAMED_PREFIX in consumer._delivered_final_text
        assert consumer.delivered_final_matches(FULL_RESPONSE) is False
        assert consumer.delivered_final_matches(STREAMED_PREFIX) is True


# ---------------------------------------------------------------------------
# Gateway-boundary regression — record-less flags must not swallow the reply
# ---------------------------------------------------------------------------


class CaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.DISCORD):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self._next_id = 0
        self.fail_edits = False

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    def _mint_id(self) -> str:
        self._next_id += 1
        return f"m-{self._next_id}"

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id=self._mint_id())

    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None
    ) -> SendResult:
        if self.fail_edits:
            return SendResult(success=False, error="websocket closed")
        self.edits.append(
            {"message_id": message_id, "content": content, "finalize": finalize}
        )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class PrefixOnlyAgent:
    """Streams only a prefix; the completed response has a longer tail."""

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


class _RecordlessFlagConsumer(GatewayStreamConsumer):
    """Sabotage subclass: models the #95382/#98552 incident state.

    After a normal drain, claim final delivery via the flags but scrub the
    recorded payload — the pre-fix gateway read matcher ``None`` as legacy
    trust and suppressed the corrective send even though only the prefix was
    ever visible.
    """

    async def run(self):
        await super().run()
        self._final_response_sent = True
        self._final_content_delivered = True
        self._turn_split_delivery = False
        self._delivered_final_text = None
        # Only the prefix was ever on screen.
        self._last_sent_text = STREAMED_PREFIX
        self._delivered_segment_texts = []
        self._delivered_commentary_texts = []


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


async def _run_turn(monkeypatch, tmp_path, *, consumer_cls=None, session_id):
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
    fake_run_agent.AIAgent = PrefixOnlyAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    if consumer_cls is not None:
        stream_consumer_mod = importlib.import_module("gateway.stream_consumer")
        monkeypatch.setattr(
            stream_consumer_mod, "GatewayStreamConsumer", consumer_cls
        )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="1534932197436424204", chat_type="group"
    )
    result = await runner._run_agent(
        message="deploy status?",
        context_prompt="",
        history=[],
        source=source,
        session_id=session_id,
        session_key=f"agent:main:discord:group:{session_id}",
    )
    return adapter, result


@pytest.mark.asyncio
async def test_recordless_delivery_flag_does_not_suppress_complete_response(
    monkeypatch, tmp_path
):
    """#95382 boundary: flags claim delivery, nothing recorded, only the
    prefix visible — the complete response must NOT be suppressed."""
    adapter, result = await _run_turn(
        monkeypatch,
        tmp_path,
        consumer_cls=_RecordlessFlagConsumer,
        session_id="sess-95382-recordless",
    )
    assert result["final_response"] == FULL_RESPONSE
    # Pre-fix behavior: already_sent=True and the tail appears in NO platform
    # call (silent partial delivery). Post-fix: either the gateway performed
    # the reconciliation edit itself (full text on the wire), or it declined
    # to claim delivery so the caller's normal final send delivers it.
    all_payloads = [c["content"] for c in adapter.sent] + [
        e["content"] for e in adapter.edits
    ]
    delivered_here = any(FULL_RESPONSE in p for p in all_payloads)
    assert delivered_here or not result.get("already_sent"), (
        "silent partial delivery: gateway claimed delivery but the complete "
        f"response never reached the platform; payloads={all_payloads!r}"
    )


@pytest.mark.asyncio
async def test_normal_streaming_turn_still_suppresses_exactly_once(
    monkeypatch, tmp_path
):
    """Control: an honest streaming turn (finalize edit carries the full
    response) must still suppress the duplicate normal send."""
    adapter, result = await _run_turn(
        monkeypatch, tmp_path, session_id="sess-95382-control"
    )
    assert result["final_response"] == FULL_RESPONSE
    all_payloads = [c["content"] for c in adapter.sent] + [
        e["content"] for e in adapter.edits
    ]
    assert any(FULL_RESPONSE in p for p in all_payloads)
    full_sends = [c for c in adapter.sent if FULL_RESPONSE in c["content"]]
    assert len(full_sends) <= 1, f"duplicate final delivery: {full_sends!r}"


@pytest.mark.asyncio
async def test_recordless_flag_with_dead_transport_leaves_normal_send(
    monkeypatch, tmp_path
):
    """#95382 incident shape: the reconciliation edit ALSO fails (dead
    transport). The gateway must NOT claim already_sent — the normal final
    send (and, on failure there, the delivery ledger) owns recovery."""

    class _DeadEditRecordlessConsumer(_RecordlessFlagConsumer):
        async def run(self):
            await super().run()
            # Transport dies after the stream drained: every further edit
            # fails, like a dropped Discord WebSocket.
            self.adapter.fail_edits = True

    adapter, result = await _run_turn(
        monkeypatch,
        tmp_path,
        consumer_cls=_DeadEditRecordlessConsumer,
        session_id="sess-95382-dead-transport",
    )
    assert result["final_response"] == FULL_RESPONSE
    assert not result.get("already_sent"), (
        "gateway claimed delivery although neither the stream nor the "
        "reconciliation edit put the complete response on the wire"
    )


# ---------------------------------------------------------------------------
# Discord transport classification + ledger reconnect replay (#95382 lane 2)
# ---------------------------------------------------------------------------


class TestDiscordTransportClassification:
    def _adapter_module(self):
        import plugins.platforms.discord.adapter as mod

        return mod

    def test_connection_error_is_transport(self):
        mod = self._adapter_module()
        assert mod._is_discord_transport_error(ConnectionError("websocket closed"))
        assert mod._is_discord_transport_error(
            RuntimeError("Session is closed")
        )
        assert mod._is_discord_transport_error(OSError(104, "Connection reset"))

    def test_http_and_timeout_errors_are_not_transport(self):
        mod = self._adapter_module()
        assert not mod._is_discord_transport_error(
            RuntimeError("error code: 50013: Missing Permissions")
        )
        assert not mod._is_discord_transport_error(asyncio.TimeoutError())

    @pytest.mark.asyncio
    async def test_send_without_client_reports_send_path_degraded(self):
        mod = self._adapter_module()
        adapter = mod.DiscordAdapter.__new__(mod.DiscordAdapter)
        adapter._client = None
        result = await mod.DiscordAdapter.send(adapter, "c1", "hello")
        assert result.success is False
        assert result.error == "send_path_degraded"
        assert result.retryable is True


class TestLedgerReplaysDegradedDiscordSend:
    def test_reconnect_sweep_claims_degraded_discord_row(self, tmp_path, monkeypatch):
        """End-to-end ledger check: a final response rejected with
        ``send_path_degraded`` on Discord is claimed by the runtime
        reconnect sweep; a generic 'Not connected' row (pre-fix error
        string) is stranded. This is the exact silent-loss mechanism from
        the #95382 field logs."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        import gateway.delivery_ledger as dl

        importlib.reload(dl)

        oid_degraded = dl.compute_obligation_id("sess-a", "msg-1", FULL_RESPONSE)
        dl.record_obligation(
            obligation_id=oid_degraded,
            session_key="agent:main:discord:group:c1",
            platform="discord",
            chat_id="c1",
            thread_id=None,
            content=FULL_RESPONSE,
        )
        dl.mark_attempting(oid_degraded)
        dl.mark_failed(oid_degraded, "send_path_degraded")

        oid_generic = dl.compute_obligation_id("sess-b", "msg-2", FULL_RESPONSE)
        dl.record_obligation(
            obligation_id=oid_generic,
            session_key="agent:main:discord:group:c2",
            platform="discord",
            chat_id="c2",
            thread_id=None,
            content=FULL_RESPONSE,
        )
        dl.mark_attempting(oid_generic)
        dl.mark_failed(oid_generic, "Not connected")

        claimed = dl.sweep_failed_for_runtime("discord")
        claimed_ids = {row["obligation_id"] for row in claimed}
        assert oid_degraded in claimed_ids, (
            "send_path_degraded Discord row must be replayable after reconnect"
        )
        assert oid_generic not in claimed_ids, (
            "non-transport errors must not be blindly replayed"
        )
