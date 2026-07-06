import json
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


class FakeEditableTelegramAdapter:
    name = "fake-telegram"
    MAX_MESSAGE_LENGTH = 4000
    REQUIRES_EDIT_FINALIZE = False
    SUPPORTS_MESSAGE_EDITING = True

    def __init__(self):
        self.sent = []
        self.edits = []
        self.typing = []
        self._pending_messages = {}

    def message_len_fn(self, text):
        return len(str(text))

    async def send(self, chat_id, content, reply_to=None, metadata=None, **kwargs):
        message_id = f"msg-{len(self.sent) + 1}"
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
                **kwargs,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                **kwargs,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

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

    def emit_tool_progress(self, cb):
        raise NotImplementedError

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **kwargs):
        cb = self.tool_progress_callback
        assert cb is not None
        self.emit_tool_progress(cb)
        # Simulate the post-tool model turn so the throttled gateway progress
        # editor has time to publish the completion-duration update.
        time.sleep(1.7)

        return {
            "final_response": "done",
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "done"},
            ],
            "api_calls": 1,
            "completed": True,
        }


class FakeTimingAgent(BaseFakeTimingAgent):
    def emit_tool_progress(self, cb):
        cb("tool.started", "write_file", preview="/tmp/demo.txt", args={"path": "/tmp/demo.txt"})
        cb("tool.completed", "write_file", duration=0.5)
        cb("tool.started", "terminal", preview="date", args={"command": "date"})
        cb("tool.completed", "terminal", duration=0.01)


class FakeRepeatedTimingAgent(BaseFakeTimingAgent):
    def emit_tool_progress(self, cb):
        args = {"code": "print('same')"}
        cb("tool.started", "execute_code", preview="print('same')", args=args)
        cb("tool.completed", "execute_code", duration=0.2)
        cb("tool.started", "execute_code", preview="print('same')", args=args)
        cb("tool.completed", "execute_code", duration=0.3)


class FakeTodoTimingAgent(BaseFakeTimingAgent):
    def emit_tool_progress(self, cb):
        args = {
            "todos": [
                {"id": "one", "content": "Audit config", "status": "in_progress"},
                {"id": "two", "content": "Summarize risk", "status": "pending"},
            ]
        }
        cb("tool.started", "todo", preview=None, args=args)
        # Completion carries the tool result with per-item wall-clock timing
        # (the todo tool stamps started_at/ended_at and exposes elapsed_seconds).
        # The "one" task finished after ~2m14s; "two" never started → no time.
        result = json.dumps({
            "todos": [
                {"id": "one", "content": "Audit config", "status": "completed", "elapsed_seconds": 134.0},
                {"id": "two", "content": "Summarize risk", "status": "pending", "elapsed_seconds": None},
            ],
            "summary": {
                "total": 2, "pending": 1, "in_progress": 0,
                "completed": 1, "cancelled": 0, "total_elapsed_seconds": 134.0,
            },
        })
        cb("tool.completed", "todo", duration=0.004, result=result)


class FakeParallelSameToolAgent(BaseFakeTimingAgent):
    """Mirrors agent/tool_executor.py ordering: ALL starts first (in parsed
    order), THEN all completions (in the same parsed order). Two same-name
    terminal calls with distinct previews + durations must each get their own
    duration appended to the correct row (FIFO), not swapped."""

    def emit_tool_progress(self, cb):
        cb("tool.started", "terminal", preview="sleep 10", args={"command": "sleep 10"})
        cb("tool.started", "terminal", preview="date", args={"command": "date"})
        cb("tool.completed", "terminal", duration=10.0)   # completes "sleep 10"
        cb("tool.completed", "terminal", duration=0.01)   # completes "date"


class FakeNewModeRepeatedAgent(BaseFakeTimingAgent):
    """Repeated same-name calls under tool_progress='new' must still each get
    their own row when durations are enabled (no 'new' suppression)."""

    def emit_tool_progress(self, cb):
        cb("tool.started", "execute_code", preview="print(1)", args={"code": "print(1)"})
        cb("tool.completed", "execute_code", duration=0.2)
        cb("tool.started", "execute_code", preview="print(2)", args={"code": "print(2)"})
        cb("tool.completed", "execute_code", duration=0.3)


def test_format_tool_completion_progress_line():
    assert gateway_run._format_tool_completion_progress_line("write_file", 0.5) == (
        "✅ write_file completed in 0.5s"
    )
    assert gateway_run._format_tool_completion_progress_line("terminal", 0.01) == (
        "✅ terminal completed in 10ms"
    )
    assert gateway_run._format_tool_completion_progress_line(
        "delegate_task", 125.2, is_error=True
    ) == "❌ delegate_task failed after 2m 05s"


def _base_user_config(platform_display):
    return {
        "display": {
            "interim_assistant_messages": False,
            "platforms": {
                "telegram": {
                    "streaming": False,
                    **platform_display,
                }
            },
        },
        "agent": {"disabled_toolsets": []},
    }


async def _run_gateway_with_fake_agent(monkeypatch, user_config, agent_cls):
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: user_config)
    monkeypatch.setattr(gateway_run, "_reload_runtime_env_preserving_config_authority", lambda: None)
    monkeypatch.setattr(run_agent, "AIAgent", agent_cls)

    completion_lines = []
    original_completion_line = gateway_run._format_tool_completion_progress_line

    def spy_completion_line(*args, **kwargs):
        line = original_completion_line(*args, **kwargs)
        completion_lines.append(line)
        return line

    monkeypatch.setattr(gateway_run, "_format_tool_completion_progress_line", spy_completion_line)

    adapter = FakeEditableTelegramAdapter()
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
    return result, rendered, completion_lines


@pytest.mark.asyncio
async def test_telegram_completion_durations_work_when_tool_progress_is_off(monkeypatch):
    user_config = _base_user_config({"tool_completion_durations": True})

    result, rendered, completion_lines = await _run_gateway_with_fake_agent(
        monkeypatch,
        user_config,
        FakeTimingAgent,
    )

    assert result["final_response"] == "done"
    assert "✅ write_file completed in 0.5s" in completion_lines
    assert "✅ terminal completed in 10ms" in completion_lines
    assert "✅ write_file completed in 0.5s" in rendered
    assert "✅ terminal completed in 10ms" in rendered
    # Duration-only mode should not re-enable noisy tool-start rows.
    assert "/tmp/demo.txt" not in rendered


@pytest.mark.asyncio
async def test_completion_rows_are_not_overwritten_by_repeated_start_dedup(monkeypatch):
    user_config = _base_user_config(
        {
            "tool_progress": "all",
            "tool_completion_durations": True,
        }
    )

    result, rendered, completion_lines = await _run_gateway_with_fake_agent(
        monkeypatch,
        user_config,
        FakeRepeatedTimingAgent,
    )

    assert result["final_response"] == "done"
    assert completion_lines == []
    assert "execute_code: \"print('same')\" · 0.2s" in rendered
    assert "execute_code: \"print('same')\" · 0.3s" in rendered
    assert "✅ execute_code completed in 0.2s" not in rendered
    assert "✅ execute_code completed in 0.3s" not in rendered


@pytest.mark.asyncio
async def test_todo_plan_card_rerenders_with_per_item_durations(monkeypatch):
    user_config = _base_user_config(
        {
            "tool_progress": "all",
            "tool_completion_durations": True,
        }
    )

    result, rendered, completion_lines = await _run_gateway_with_fake_agent(
        monkeypatch,
        user_config,
        FakeTodoTimingAgent,
    )

    assert result["final_response"] == "done"
    assert completion_lines == []
    # The todo card is RE-RENDERED at completion with per-item wall-clock
    # durations (the start card is replaced in place via __todo_complete__).
    # The old behavior appended the tool's own write latency ("· 4ms") to the
    # start card, which measured the in-memory list write, not task time — that
    # suffix must be gone, and no standalone "✅ todo completed" fallback row.
    assert "· 4ms" not in rendered
    assert "✅ todo completed in 4ms" not in rendered
    # Completed task shows its wall-clock span; pending task shows none.
    assert "✅ completed (2m 14s) - Audit config" in rendered
    assert "⏳ pending - Summarize risk" in rendered
    # The pending row never gets a duration paren.
    pending_rows = [
        line for line in rendered.splitlines() if "Summarize risk" in line
    ]
    assert pending_rows and all("(" not in row for row in pending_rows)


@pytest.mark.asyncio
async def test_todo_completion_without_result_keeps_start_card(monkeypatch):
    """If a todo completion arrives with no result payload, keep the start card
    as-is. The completion must NOT emit the 'Reading task list' sentinel or a
    bogus empty card."""
    user_config = _base_user_config(
        {
            "tool_progress": "all",
            "tool_completion_durations": True,
        }
    )

    class FakeTodoNoResultAgent(BaseFakeTimingAgent):
        def emit_tool_progress(self, cb):
            args = {"todos": [
                {"id": "one", "content": "Audit config", "status": "in_progress"},
            ]}
            cb("tool.started", "todo", preview=None, args=args)
            cb("tool.completed", "todo", duration=0.004)  # no result kwarg

    result, rendered, completion_lines = await _run_gateway_with_fake_agent(
        monkeypatch,
        user_config,
        FakeTodoNoResultAgent,
    )

    assert result["final_response"] == "done"
    # Start card preserved; no sentinel, no duration suffix, no fallback row.
    assert "🔄 in progress - Audit config" in rendered
    assert "Reading task list" not in rendered
    assert "· 4ms" not in rendered
    assert "✅ todo completed" not in rendered


@pytest.mark.asyncio
async def test_parallel_same_tool_durations_match_fifo_not_swapped(monkeypatch):
    """Two same-name terminal calls (parsed-order starts, then parsed-order
    completions) must each get their own duration on the correct row."""
    user_config = _base_user_config(
        {
            "tool_progress": "all",
            "tool_completion_durations": True,
        }
    )

    result, rendered, completion_lines = await _run_gateway_with_fake_agent(
        monkeypatch,
        user_config,
        FakeParallelSameToolAgent,
    )

    assert result["final_response"] == "done"
    assert completion_lines == []
    # FIFO match: "sleep 10" started first → gets the first completion (10s);
    # "date" started second → gets the second completion (10ms).
    assert 'terminal: "sleep 10" · 10s' in rendered
    assert 'terminal: "date" · 10ms' in rendered
    # The pre-fix LIFO bug would swap these.
    assert 'terminal: "sleep 10" · 10ms' not in rendered
    assert 'terminal: "date" · 10s' not in rendered


@pytest.mark.asyncio
async def test_new_mode_keeps_repeated_rows_when_durations_enabled(monkeypatch):
    """Under tool_progress='new', repeated same-tool calls are normally
    collapsed, but with durations enabled each call keeps its own row so its
    timing lands in place (never a standalone fallback row)."""
    user_config = _base_user_config(
        {
            "tool_progress": "new",
            "tool_completion_durations": True,
        }
    )

    result, rendered, completion_lines = await _run_gateway_with_fake_agent(
        monkeypatch,
        user_config,
        FakeNewModeRepeatedAgent,
    )

    assert result["final_response"] == "done"
    assert completion_lines == []
    assert 'execute_code: "print(1)" · 0.2s' in rendered
    assert 'execute_code: "print(2)" · 0.3s' in rendered
    # No fallback standalone completion rows — both durations landed in place.
    assert "✅ execute_code completed" not in rendered
