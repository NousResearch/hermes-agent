"""Regression: a terminal/tool start that triggers a progress-bubble overflow
split must still get its completion-duration suffix on the SAME row, not a
standalone ``✅ terminal completed in Xs`` fallback line below it.

Root cause: ``_roll_progress_overflow_if_needed`` keeps only the last group
as the mutable bubble and used to BLANKET-clear the pending tool-line index
map. That orphaned a start row that survived into the kept bubble, so its
later ``__tool_duration__`` couldn't find it and fell back to a standalone
completion line — the "second line instead of the same one" symptom.
"""

import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
import run_agent
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.session import SessionSource, build_session_key


class TinyLimitTelegramAdapter:
    """Editable, code-block-capable, with a small message limit so a single
    extra tool row forces an overflow split."""

    name = "fake-telegram"
    MAX_MESSAGE_LENGTH = 80
    REQUIRES_EDIT_FINALIZE = False
    SUPPORTS_MESSAGE_EDITING = True
    supports_code_blocks = True

    def __init__(self):
        self.sent = []
        self.edits = []
        self.typing = []

    def message_len_fn(self, text):
        return len(str(text))

    async def send(self, chat_id, content, reply_to=None, metadata=None, **kwargs):
        message_id = f"msg-{len(self.sent) + 1}"
        self.sent.append({"content": content})
        return SendResult(success=True, message_id=message_id)

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        self.edits.append({"message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        self.typing.append({"chat_id": chat_id})

    def has_pending_interrupt(self, session_key):
        return False

    def get_pending_message(self, session_key):
        return None

    def register_post_delivery_callback(self, *args, **kwargs):
        return None


class BaseFakeTimingAgent:
    def __init__(self, *args, **kwargs):
        self.model = kwargs.get("model") or (args[0] if args else "fake-model")
        self.session_id = kwargs.get("session_id", "session-1")
        self.tools = []
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=0)
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.step_callback = None
        self.stream_delta_callback = None
        self.interim_assistant_callback = None
        self.status_callback = None
        self.notice_callback = None
        self.notice_clear_callback = None
        self.background_review_callback = None
        self.clarify_callback = None
        self.reasoning_config = None
        self.service_tier = None
        self.request_overrides = {}
        self.is_interrupted = False

    def get_activity_summary(self):
        return {
            "seconds_since_activity": 0.0,
            "last_activity_desc": "fake tool progress",
            "current_tool": None,
            "api_call_count": 1,
            "max_iterations": 90,
        }

    def interrupt(self, message=None):
        self.is_interrupted = True

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **kwargs):
        raise NotImplementedError


class FakeOverflowAgent(BaseFakeTimingAgent):
    """A prior tool fills the bubble, then a terminal start tips it over the
    80-char limit (overflow split), then the terminal completes."""

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **kwargs):
        cb = self.tool_progress_callback
        assert cb is not None

        cb("tool.started", "execute_code", preview="x = 1", args={"code": "x = 1"})
        cb("tool.completed", "execute_code", duration=0.1)
        # Terminal renders as a multi-line code block; appending it overflows
        # the 80-char bubble, splitting it and resetting pending indexes.
        cb("tool.started", "terminal", preview="acli jira workitem view ASPD-30171",
           args={"command": "acli jira workitem view ASPD-30171"})
        cb("tool.completed", "terminal", duration=1.4)
        time.sleep(2.0)
        return {
            "final_response": "done",
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "done"},
            ],
            "api_calls": 1,
            "completed": True,
        }


def _user_config():
    return {
        "display": {
            "interim_assistant_messages": False,
            "platforms": {
                "telegram": {
                    "streaming": False,
                    "tool_progress": "all",
                    "tool_completion_durations": True,
                }
            },
        },
        "agent": {"disabled_toolsets": []},
    }


async def _run(monkeypatch, user_config, agent_cls):
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: user_config)
    monkeypatch.setattr(
        gateway_run, "_reload_runtime_env_preserving_config_authority", lambda: None
    )
    monkeypatch.setattr(run_agent, "AIAgent", agent_cls)

    adapter = TinyLimitTelegramAdapter()
    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.hooks = SimpleNamespace(loaded_hooks=False, emit=AsyncMock())
    runner.session_store = SimpleNamespace(_entries={})
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._provider_routing = {}
    runner._reasoning_config = None
    runner._service_tier = None
    runner._session_db = None
    runner._fallback_model = None
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._voice_mode = {}

    runner._get_proxy_url = lambda: None
    runner._thread_metadata_for_source = lambda source, event_message_id=None: None
    runner._resolve_session_agent_runtime = lambda **kwargs: ("fake-model", {})
    runner._resolve_session_reasoning_config = lambda **kwargs: None
    runner._load_service_tier = lambda: None
    runner._resolve_turn_agent_config = lambda message, model, runtime: {
        "model": model,
        "runtime": runtime,
    }
    runner._agent_config_signature = lambda *args, **kwargs: "fake-signature"
    runner._extract_cache_busting_config = lambda config: {}
    runner._enforce_agent_cache_cap = lambda: None
    runner._init_cached_agent_for_turn = lambda agent, depth: None
    runner._is_session_run_current = lambda session_key, generation: True
    runner._release_running_agent_state = lambda *args, **kwargs: None
    runner._update_runtime_status = lambda *args, **kwargs: None
    runner._is_intentional_model_switch = lambda *args, **kwargs: False
    runner._evict_cached_agent = lambda *args, **kwargs: None
    runner._promote_queued_event = lambda session_key, adapter, pending_event: pending_event
    runner._is_telegram_topic_lane = lambda source: False
    runner._deliver_platform_notice = AsyncMock()

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
        user_name="tester",
    )
    session_key = build_session_key(source)

    result = await runner._run_agent(
        message="run fake tools",
        context_prompt="",
        history=[],
        source=source,
        session_id="session-1",
        session_key=session_key,
        run_generation=1,
    )
    rendered = "\n".join(
        [item["content"] for item in adapter.sent]
        + [item["content"] for item in adapter.edits]
    )
    return result, rendered


@pytest.mark.asyncio
async def test_terminal_duration_stays_inline_across_overflow_split(monkeypatch):
    result, rendered = await _run(monkeypatch, _user_config(), FakeOverflowAgent)
    print("\n===== RENDERED =====\n" + rendered + "\n===== END =====")

    assert result["final_response"] == "done"
    # The terminal's duration must land on its own start row...
    assert "💻 terminal · 1.4s" in rendered
    # ...not as a standalone fallback completion line below it.
    assert "✅ terminal completed in 1.4s" not in rendered
