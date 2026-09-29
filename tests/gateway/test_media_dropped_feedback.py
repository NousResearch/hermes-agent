"""A MEDIA attachment the gateway skips must re-enter the session, not vanish into the log (#75065).

``filter_media_delivery_paths`` strips a rejected ``MEDIA:`` directive, never uploads the file, and
records only a host-side ``WARNING``. On a sandboxed terminal backend the agent cannot read that
log, so it reports an attachment the user never received and never learns to correct the path.

These tests pin the two halves of the fix:
- the rejection is collected (``dropped``) and formatted into one ``[IMPORTANT: ...]`` line;
- the foreground gateway delivery paths inject that line as a bounded same-session follow-up turn.
"""

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter, SendResult, append_media_dropped_notice, format_media_dropped_notice,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_goals import GatewayGoalsMixin
from gateway.run_notifications import GatewayNotificationsMixin
from gateway.session import SessionSource

MISSING_REASON = "not found on this host"


def _source() -> SessionSource:
    return SessionSource(platform=Platform.DISCORD, chat_id="D1", chat_type="dm", thread_id=None,
                         message_id="m1")


def _event(metadata=None) -> MessageEvent:
    return MessageEvent(text="hi", message_type=MessageType.TEXT, source=_source(),
                        message_id="m1", metadata=metadata or {})


def _strict_roots(tmp_path, monkeypatch):
    root = tmp_path / "media-cache"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
    return root


class TestFormatMediaDroppedNotice:
    def test_one_line_names_path_and_reason(self):
        notice = format_media_dropped_notice(
            [{"path": "/workspace/out/video.mp4", "reason": MISSING_REASON}])
        assert notice == (
            "[IMPORTANT: 1 MEDIA attachment(s) were skipped: "
            "/workspace/out/video.mp4 - " + MISSING_REASON + "]")
        assert "\n" not in notice

    def test_dedupes_paths_and_counts_them(self):
        notice = format_media_dropped_notice([
            {"path": "/a.mp4", "reason": MISSING_REASON},
            {"path": "/a.mp4", "reason": MISSING_REASON},
            {"path": "/b.mp4", "reason": "denied by the delivery policy"},
        ])
        assert notice.startswith("[IMPORTANT: 2 MEDIA attachment(s) were skipped: ")
        assert "/a.mp4 - " + MISSING_REASON in notice
        assert "/b.mp4 - denied by the delivery policy" in notice

    def test_empty_when_nothing_was_dropped(self):
        assert format_media_dropped_notice([]) == ""
        assert format_media_dropped_notice(None) == ""

    def test_control_chars_cannot_forge_a_second_line(self):
        notice = format_media_dropped_notice([{"path": "/a\nFAKE LOG LINE.mp4", "reason": ""}])
        assert "\n" not in notice

    def test_long_path_is_not_truncated(self):
        long_path = "/workspace/" + "nested/" * 60 + "video.mp4"
        notice = format_media_dropped_notice([{"path": long_path, "reason": MISSING_REASON}])
        assert long_path in notice


class TestBackgroundTaskNotice:
    """The detached background-task lane has no turn to re-enter (run_turn.py)."""

    def test_appends_notice_below_existing_text(self):
        out = append_media_dropped_notice(
            "Rendered the clip.", [{"path": "/workspace/clip.mp4", "reason": MISSING_REASON}])
        assert out.startswith("Rendered the clip.\n\n[IMPORTANT: 1 MEDIA attachment(s) were skipped:")
        assert "/workspace/clip.mp4 - " + MISSING_REASON in out

    def test_empty_text_becomes_the_notice(self):
        """A media-only response must still carry the drop, not fall to \"(No response generated)\"."""
        out = append_media_dropped_notice(
            "", [{"path": "/workspace/clip.mp4", "reason": MISSING_REASON}])
        assert out.startswith("[IMPORTANT: 1 MEDIA attachment(s) were skipped:")

    def test_untouched_when_nothing_was_dropped(self):
        assert append_media_dropped_notice("Rendered the clip.", []) == "Rendered the clip."
        assert append_media_dropped_notice("", []) == ""


class TestFilterRecordsRejections:
    def test_filter_appends_dropped_with_reason(self, tmp_path, monkeypatch):
        root = _strict_roots(tmp_path, monkeypatch)
        safe = root / "ok.png"
        safe.write_bytes(b"\x89PNG")
        outside = tmp_path / "outside.png"
        outside.write_bytes(b"\x89PNG")
        dropped: list = []

        kept = BasePlatformAdapter.filter_media_delivery_paths(
            [(str(outside), False), (str(safe), False)], dropped=dropped)

        assert [p for p, _ in kept] == [str(safe.resolve())]
        assert dropped == [{"path": str(outside), "reason": "denied by the delivery policy"}]

    def test_missing_file_reported_as_not_found(self, tmp_path, monkeypatch):
        _strict_roots(tmp_path, monkeypatch)
        ghost = tmp_path / "never-written.mp4"
        dropped: list = []
        BasePlatformAdapter.filter_media_delivery_paths([(str(ghost), False)], dropped=dropped)
        assert len(dropped) == 1 and dropped[0]["reason"] == MISSING_REASON

    def test_no_dropped_list_is_still_backward_compatible(self, tmp_path, monkeypatch):
        _strict_roots(tmp_path, monkeypatch)
        missing = tmp_path / "gone.mp4"
        assert BasePlatformAdapter.filter_media_delivery_paths([(str(missing), False)]) == []


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.DISCORD)

    async def connect(self) -> bool:  # pragma: no cover - not exercised
        return True

    async def disconnect(self) -> None:  # pragma: no cover - not exercised
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - not exercised
        return {"name": chat_id, "type": "dm"}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        from gateway.platforms.base import SendResult
        return SendResult(success=True, message_id="m1")


class TestExtractionCollectsRejections:
    @pytest.mark.asyncio
    async def test_extract_response_content_collects_dropped_media(self, tmp_path, monkeypatch):
        _strict_roots(tmp_path, monkeypatch)
        missing = tmp_path / "never-written.mp4"
        adapter = _Adapter()

        extracted = await adapter._extract_response_content(
            f"Here you go.\nMEDIA:{missing}", _event(), "agent:main:discord:dm:D1",
            is_ephemeral_response=False)

        assert extracted.media_files == []
        assert len(extracted.media_dropped) == 1
        assert "never-written.mp4" in extracted.media_dropped[0]["path"]
        assert extracted.media_dropped[0]["reason"] == MISSING_REASON
        assert "Here you go." in extracted.text_content

    @pytest.mark.asyncio
    async def test_rejected_bare_path_is_reported_like_a_media_tag(self, tmp_path, monkeypatch):
        """extract_local_files strips a bare path from the text, so a policy rejection of it must
        reach the agent too, not only the host log."""
        _strict_roots(tmp_path, monkeypatch)
        outside = tmp_path / "report.pdf"
        outside.write_bytes(b"%PDF")
        adapter = _Adapter()

        extracted = await adapter._extract_response_content(
            f"Saved it to {outside}", _event(), "agent:main:discord:dm:D1", is_ephemeral_response=False)

        assert extracted.local_files == []
        assert extracted.media_dropped == [
            {"path": str(outside), "reason": "denied by the delivery policy"}]

    @pytest.mark.asyncio
    async def test_deliverable_media_reports_nothing_dropped(self, tmp_path, monkeypatch):
        root = _strict_roots(tmp_path, monkeypatch)
        good = root / "chart.png"
        good.write_bytes(b"\x89PNG")
        adapter = _Adapter()

        extracted = await adapter._extract_response_content(
            f"Chart.\nMEDIA:{good}", _event(), "agent:main:discord:dm:D1",
            is_ephemeral_response=False)

        assert extracted.media_dropped == []
        assert [p for p, _ in extracted.media_files] == [str(good.resolve())]


class _Runner(GatewayGoalsMixin, GatewayNotificationsMixin):
    """Real notification mixin method under test; collaborators captured, not a live runner."""

    def __init__(self):
        self.captured: list = []

    def _thread_metadata_for_source(self, source, anchor=None):
        return {}

    def _reply_anchor_for_event(self, event):
        return None

    def _session_key_for_source(self, source):
        return "agent:main:discord:dm:D1"

    def _enqueue_fifo(self, session_key, queued_event, adapter):
        self.captured.append((session_key, queued_event, adapter))


def _media_adapter():
    return SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_multiple_images=AsyncMock(),
        send_document=AsyncMock(),
        send_image_file=AsyncMock(),
        send_video=AsyncMock(),
        send_voice=AsyncMock(),
    )


class TestSameSessionFeedback:
    def test_queue_media_delivery_feedback_enqueues_one_internal_turn(self):
        runner = _Runner()
        runner._queue_media_delivery_feedback(
            _event(), "agent:main:discord:dm:D1", None,
            [{"path": "/workspace/x.mp4", "reason": MISSING_REASON}])

        assert len(runner.captured) == 1
        session_key, feedback, _adapter = runner.captured[0]
        assert session_key == "agent:main:discord:dm:D1"
        assert feedback.internal is True
        assert feedback.message_id is None
        # Not a reply to the message that produced the bad path (#52694 anchor rule).
        assert feedback.source.message_id is None
        assert feedback.source.chat_id == "D1"
        assert feedback.metadata.get("media_delivery_feedback") is True
        assert "[IMPORTANT: 1 MEDIA attachment(s) were skipped: /workspace/x.mp4 - " \
               + MISSING_REASON + "]" in feedback.text

    def test_feedback_never_chains_a_second_notice(self):
        """The feedback turn re-emitting a bad path must not queue another turn forever."""
        runner = _Runner()
        runner._queue_media_delivery_feedback(
            _event(metadata={"media_delivery_feedback": True}), "agent:main:discord:dm:D1", None,
            [{"path": "/workspace/x.mp4", "reason": MISSING_REASON}])
        assert runner.captured == []

    def test_nothing_dropped_queues_nothing(self):
        runner = _Runner()
        runner._queue_media_delivery_feedback(
            _event(), "agent:main:discord:dm:D1", None, [])
        assert runner.captured == []

    def test_adapter_no_op_without_a_runner(self):
        """Standalone adapters (no gateway runner) must not raise on a rejected path."""
        adapter = _Adapter()
        adapter._emit_media_dropped_feedback(
            _event(), "agent:main:discord:dm:D1",
            [{"path": "/workspace/x.mp4", "reason": MISSING_REASON}])

    def test_adapter_hands_rejections_to_the_runner(self):
        adapter = _Adapter()
        captured: list = []
        adapter.gateway_runner = SimpleNamespace(
            _queue_media_delivery_feedback=lambda *args: captured.append(args))
        dropped = [{"path": "/workspace/x.mp4", "reason": MISSING_REASON}]

        adapter._emit_media_dropped_feedback(_event(), "agent:main:discord:dm:D1", dropped)

        assert len(captured) == 1
        event_arg, session_key_arg, adapter_arg, dropped_arg = captured[0]
        assert session_key_arg == "agent:main:discord:dm:D1"
        assert adapter_arg is adapter
        assert dropped_arg == dropped

    @pytest.mark.asyncio
    async def test_foreground_turn_delivers_text_and_hands_off_the_rejection(self, tmp_path, monkeypatch):
        """Non-streaming turn end to end: the reply is delivered without the MEDIA line and the
        runner receives the rejection after delivery (the ``_process_message_background`` wiring)."""
        _strict_roots(tmp_path, monkeypatch)
        missing = tmp_path / "never-written.mp4"
        adapter = _Adapter()
        sent: list = []
        captured: list = []

        async def _send(chat_id, content, reply_to=None, metadata=None):
            from gateway.platforms.base import SendResult
            sent.append(content)
            return SendResult(success=True, message_id="m2")

        adapter.send = _send
        adapter.gateway_runner = SimpleNamespace(
            _queue_media_delivery_feedback=lambda *args: captured.append(args))

        async def handler(_event):
            return f"Here you go.\nMEDIA:{missing}"

        adapter.set_message_handler(handler)
        event = _event()
        await adapter._process_message_background(event, "agent:main:discord:dm:D1")

        assert sent == ["Here you go."]
        assert len(captured) == 1
        event_arg, session_key_arg, adapter_arg, dropped_arg = captured[0]
        assert event_arg is event and adapter_arg is adapter
        assert session_key_arg == "agent:main:discord:dm:D1"
        assert [d["reason"] for d in dropped_arg] == [MISSING_REASON]
        assert "never-written.mp4" in dropped_arg[0]["path"]

    @pytest.mark.asyncio
    async def test_post_stream_rejected_media_reenters_the_session(self, tmp_path, monkeypatch):
        """Streamed reply: the visible text is untouched and the rejection comes back to the agent."""
        _strict_roots(tmp_path, monkeypatch)
        runner = _Runner()
        adapter = _media_adapter()
        missing = tmp_path / "never-written.mp4"

        await GatewayNotificationsMixin._deliver_media_from_response(
            runner, f"Rendered.\nMEDIA:{missing}", _event(), adapter)

        assert len(runner.captured) == 1
        feedback = runner.captured[0][1]
        assert "never-written.mp4" in feedback.text
        assert feedback.metadata.get("media_delivery_feedback") is True
        adapter.send_document.assert_not_awaited()
        adapter.send_video.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_post_stream_safe_media_delivers_without_feedback(self, tmp_path, monkeypatch):
        root = _strict_roots(tmp_path, monkeypatch)
        good = root / "clip.mp4"
        good.write_bytes(b"mp4")
        runner = _Runner()
        adapter = _media_adapter()

        await GatewayNotificationsMixin._deliver_media_from_response(
            runner, f"Done.\nMEDIA:{good}", _event(), adapter)

        assert runner.captured == []
        adapter.send_video.assert_awaited_once()

class _PlainAdapter(BasePlatformAdapter):
    """Real adapter (real ``_pending_messages`` slot) that records what reaches the chat."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent: list = []

    async def connect(self) -> bool:  # pragma: no cover - not exercised
        return True

    async def disconnect(self) -> None:  # pragma: no cover - not exercised
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id=f"s{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - not exercised
        return {"id": chat_id}


class _AlwaysBadMediaAgent:
    """Every turn — human or feedback — re-emits the same undeliverable MEDIA path."""

    calls: list = []
    bad_path = ""

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        type(self).calls.append(message)
        return {"final_response": f"Rendered.\nMEDIA:{type(self).bad_path}", "messages": [], "api_calls": 1}


def _real_runner(adapter, monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _AlwaysBadMediaAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
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
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False, group_sessions_per_user=False, stt_enabled=False)
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _tg_source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm")


class TestFeedbackIsSingleHopThroughTheRealDrain:
    """Drives ``GatewayRunner._run_agent``'s real in-band FIFO drain (``_enqueue_fifo`` →
    ``_pending_messages`` → queued-lane ``_deliver_queued_first_response``) with a model that
    re-emits the rejected path on EVERY turn. Only human turns may earn a notice."""

    @pytest.mark.asyncio
    async def test_repeated_rejected_output_yields_one_notice_per_human_turn(self, tmp_path, monkeypatch):
        _strict_roots(tmp_path, monkeypatch)
        _AlwaysBadMediaAgent.calls = []
        _AlwaysBadMediaAgent.bad_path = str(tmp_path / "never-written.mp4")
        adapter = _PlainAdapter()
        runner = _real_runner(adapter, monkeypatch, tmp_path)
        session_key = runner._session_key_for_source(_tg_source())
        enqueued: list = []
        real_enqueue = runner._enqueue_fifo

        def _spy(key, queued_event, adapter_):
            enqueued.append(queued_event)
            real_enqueue(key, queued_event, adapter_)

        runner._enqueue_fifo = _spy
        # A second human message is already queued behind the first, so every response in the
        # chain is delivered through the queued-follow-up lane.
        adapter._pending_messages[session_key] = MessageEvent(
            text="and another", message_type=MessageType.TEXT, source=_tg_source(), message_id="q1")

        result = await runner._run_agent(
            message="make the clip", context_prompt="", history=[], source=_tg_source(),
            session_id="sess-media-loop", session_key=session_key)

        # Two human turns, then exactly the two notices they earned — and the chain stops even
        # though both notice turns re-emitted the same rejected path.
        assert len(_AlwaysBadMediaAgent.calls) == 4
        assert _AlwaysBadMediaAgent.calls[:2] == ["make the clip", "and another"]
        assert all("MEDIA attachment(s) were skipped" in m for m in _AlwaysBadMediaAgent.calls[2:])
        # Without the queued-lane provenance a notice turn queues a third notice, which is left in
        # the slot when the chain hits _MAX_INTERRUPT_DEPTH, for the adapter to drain: self-feeding.
        assert len(enqueued) == 2
        assert all(e.metadata.get("media_delivery_feedback") is True for e in enqueued)
        assert session_key not in adapter._pending_messages
        # The chain's outer final (delivered by the adapter against the OPENING human event)
        # carries the terminal notice turn's provenance, so it cannot queue a third notice.
        assert result["queued_terminal_media_delivery_feedback"] is True

    @pytest.mark.asyncio
    async def test_outer_final_of_a_drained_notice_does_not_requeue(self, tmp_path, monkeypatch):
        """The outer delivery of the chain's terminal notice turn, with the provenance the handler
        copies onto the opening event, queues nothing."""
        _strict_roots(tmp_path, monkeypatch)
        adapter = _PlainAdapter()
        runner = _real_runner(adapter, monkeypatch, tmp_path)
        enqueued: list = []
        runner._enqueue_fifo = lambda *args: enqueued.append(args)
        opening = MessageEvent(text="make the clip", source=_tg_source(), message_id="h1",
                               metadata={"media_delivery_feedback": True})

        await runner._deliver_media_from_response(
            f"Rendered.\nMEDIA:{tmp_path / 'never-written.mp4'}", opening, adapter)

        assert enqueued == []

    @pytest.mark.asyncio
    async def test_queued_lane_carries_feedback_provenance(self, tmp_path, monkeypatch):
        _strict_roots(tmp_path, monkeypatch)
        adapter = _PlainAdapter()
        runner = _real_runner(adapter, monkeypatch, tmp_path)
        enqueued: list = []
        runner._enqueue_fifo = lambda *args: enqueued.append(args)
        response = f"Rendered.\nMEDIA:{tmp_path / 'never-written.mp4'}"

        await runner._deliver_queued_first_response(
            response, _tg_source(), adapter, event_metadata={"media_delivery_feedback": True})
        assert enqueued == []
        assert adapter.sent == ["Rendered."]

        await runner._deliver_queued_first_response(response, _tg_source(), adapter)
        assert len(enqueued) == 1
