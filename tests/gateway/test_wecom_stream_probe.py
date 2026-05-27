"""End-to-end-ish probes for WeCom gateway native streaming setup."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import SendResult
from gateway.platforms.wecom import WeComAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class RecordingWeComAdapter(WeComAdapter):
    """WeCom adapter with an in-memory websocket frame recorder."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True))
        self.frames = []
        self.normal_sends = []
        self._ws = SimpleNamespace(closed=False)

    async def _send_json(self, payload):
        self.frames.append(payload)

    async def send_typing(self, chat_id, metadata=None):
        return SendResult(success=True)

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.normal_sends.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id=f"normal-{len(self.normal_sends)}")


class TestWeComNativeStreamingProbe:
    def test_runner_builds_wecom_metadata_even_without_thread_id(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        source = SessionSource(
            platform=Platform.WECOM,
            chat_id="chat-1",
            user_id="user-1",
            chat_type="dm",
            thread_id=None,
        )

        assert runner._thread_metadata_for_source(source, "msg-1") == {
            "reply_to_message_id": "msg-1"
        }

    @pytest.mark.asyncio
    async def test_wecom_stream_consumer_emits_partial_and_finish_frames(self):
        """Probe the exact transport frames expected by the real gateway path."""
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = RecordingWeComAdapter()
        adapter._reply_req_ids["msg-1"] = "req-1"
        cfg = StreamConsumerConfig(
            transport="draft",
            chat_type="dm",
            edit_interval=0.01,
            buffer_threshold=5,
            cursor="",
        )
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat-1",
            config=cfg,
            metadata={"reply_to_message_id": "msg-1"},
            initial_reply_to_id="msg-1",
        )

        consumer.on_delta("正在")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.on_delta("流式输出")
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        stream_frames = [f for f in adapter.frames if f.get("cmd") == "aibot_respond_msg"]
        assert len(stream_frames) >= 2
        assert all(f["headers"]["req_id"] == "req-1" for f in stream_frames)
        payloads = [f["body"]["stream"] for f in stream_frames]
        assert payloads[0]["finish"] is False
        assert payloads[-1]["finish"] is True
        assert payloads[-1]["content"] == "正在流式输出"
        assert {p["id"] for p in payloads} == {payloads[0]["id"]}
        assert consumer.final_response_sent is True


    @pytest.mark.asyncio
    async def test_wecom_native_stream_uses_stream_limit_for_long_content(self):
        """Long native streams should not fall into generic markdown chunk sends."""
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = RecordingWeComAdapter()
        adapter._reply_req_ids["msg-1"] = "req-1"
        long_text = "长" * (adapter.MAX_MESSAGE_LENGTH + 500)
        cfg = StreamConsumerConfig(
            transport="draft",
            chat_type="dm",
            edit_interval=0.01,
            buffer_threshold=5,
            cursor="",
        )
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat-1",
            config=cfg,
            metadata={"reply_to_message_id": "msg-1"},
            initial_reply_to_id="msg-1",
        )

        consumer.on_delta(long_text)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        stream_frames = [f for f in adapter.frames if f.get("cmd") == "aibot_respond_msg"]
        assert adapter.normal_sends == []
        assert len(stream_frames) >= 2
        assert stream_frames[0]["body"]["stream"]["finish"] is False
        assert stream_frames[-1]["body"]["stream"]["finish"] is True
        assert stream_frames[-1]["body"]["stream"]["content"] == long_text
        assert consumer.final_response_sent is True

    @pytest.mark.asyncio
    async def test_gateway_wecom_stream_setup_uses_native_draft_metadata(self, monkeypatch):
        """Regression: gateway used to construct no metadata in WeCom DMs.

        The absence of reply_to_message_id meant send_draft() could not resolve
        the callback req_id, so no native stream frames were produced and only
        the normal final send reached the user.
        """
        import gateway.run as run_mod
        from gateway.stream_consumer import GatewayStreamConsumer

        adapter = RecordingWeComAdapter()
        adapter._reply_req_ids["msg-1"] = "req-1"
        source = SessionSource(
            platform=Platform.WECOM,
            chat_id="chat-1",
            user_id="user-1",
            chat_type="dm",
            thread_id=None,
        )
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={Platform.WECOM: PlatformConfig(enabled=True)},
            streaming=StreamingConfig(
                enabled=True,
                transport="edit",
                edit_interval=0.01,
                buffer_threshold=5,
                cursor=" ▉",
            ),
        )
        runner.adapters = {Platform.WECOM: adapter}
        runner._provider_routing = {}
        runner._reasoning_config = None
        runner._service_tier = ""
        runner._agent_cache_lock = None
        runner._agent_cache = None
        runner._prefill_messages = []
        runner._fallback_model = None
        runner._session_db = None
        runner._running_agents = {}
        runner._draining = False
        runner.hooks = SimpleNamespace(loaded_hooks=False, emit=AsyncMock())
        runner._running_generation_by_session = {}
        runner._session_reasoning_overrides = {}
        runner._session_model_overrides = {}
        runner._session_provider_overrides = {}
        runner._session_workdirs = {}
        runner._session_reasoning = {}
        runner._session_service_tier = {}
        runner._session_toolsets = {}
        runner._session_ephemeral_system_prompt = {}
        runner._session_approvals = {}
        runner._session_model_aliases = {}
        runner._session_db_lock = None
        runner._ephemeral_system_prompt = ""
        runner._pending_session_prompts = {}
        runner._memory_manager = None
        runner._user_profile_manager = None
        runner._context_engine = None

        class FakeAgent:
            def __init__(self, *args, **kwargs):
                self.stream_delta_callback = None
                self.interim_assistant_callback = None
                self.tool_progress_callback = None
                self.step_callback = None
                self.status_callback = None
                self.reasoning_config = None
                self.service_tier = ""
                self.request_overrides = {}
                self.context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=0)
                self.session_prompt_tokens = 0
                self.session_completion_tokens = 0

            def run_conversation(self, *args, **kwargs):
                import time

                assert self.stream_delta_callback is not None
                self.stream_delta_callback("正在")
                time.sleep(0.05)
                self.stream_delta_callback("流式输出")
                time.sleep(0.05)
                return {
                    "final_response": "正在流式输出",
                    "messages": [],
                    "api_calls": 1,
                    "completed": True,
                    "interrupted": False,
                    "partial": False,
                    "tools": [],
                }

            def get_activity_summary(self):
                return {"seconds_since_activity": 0}

        import run_agent as run_agent_mod

        monkeypatch.setattr(run_agent_mod, "AIAgent", FakeAgent)
        monkeypatch.setattr(
            run_mod,
            "_load_gateway_config",
            lambda: {
                "display": {"interim_assistant_messages": False},
                "streaming": {"enabled": True, "transport": "edit"},
            },
        )
        monkeypatch.setattr(GatewayStreamConsumer, "_draft_id_counter", 0)
        monkeypatch.setattr(runner, "_get_proxy_url", lambda: None)
        monkeypatch.setattr(runner, "_resolve_turn_agent_config", lambda *a, **k: {"model": "test", "runtime": {}})
        monkeypatch.setattr(runner, "_extract_cache_busting_config", lambda *a, **k: {})
        monkeypatch.setattr(runner, "_agent_config_signature", lambda *a, **k: "sig")
        monkeypatch.setattr(runner, "_load_service_tier", lambda: None)
        monkeypatch.setattr(runner, "_resolve_session_reasoning_config", lambda *a, **k: None)
        monkeypatch.setattr(runner, "_is_session_run_current", lambda *a, **k: True)
        monkeypatch.setattr(runner, "_release_running_agent_state", lambda *a, **k: None)
        monkeypatch.setattr(runner, "_enforce_agent_cache_cap", lambda *a, **k: None)
        monkeypatch.setattr(runner, "_init_cached_agent_for_turn", lambda *a, **k: None)
        monkeypatch.setattr(runner, "_session_saved_history", lambda *a, **k: ([], 0), raising=False)

        result = await runner._run_agent(
            message="probe",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-1",
            session_key="agent:main:wecom:dm:chat-1",
            event_message_id="msg-1",
        )

        stream_frames = [f for f in adapter.frames if f.get("cmd") == "aibot_respond_msg"]
        assert result["already_sent"] is True
        assert len(stream_frames) >= 2
        assert stream_frames[0]["headers"]["req_id"] == "req-1"
        assert stream_frames[0]["body"]["stream"]["finish"] is False
        assert stream_frames[-1]["body"]["stream"]["finish"] is True
        assert stream_frames[-1]["body"]["stream"]["content"] == "正在流式输出"
