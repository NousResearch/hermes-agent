"""Regression: completion-duration suffixes must survive a ``__reset__``.

A ``__tool_duration__`` event mutates the tool's progress row in place
(appending e.g. ``· 0.5s``), but the platform edit that publishes it is
throttled (``_PROGRESS_EDIT_INTERVAL``).  When interim assistant content
lands — or the final reply is delivered as a new message — the gateway
enqueues a ``__reset__`` that abandons the current progress bubble.

The main-loop ``__reset__`` handler used to clear ``progress_lines`` WITHOUT
flushing, stranding the stale start row on the platform with no duration
suffix.  This is the exact symptom seen for ``delegate_task`` (whose
completion straddled an interim "Preflight" message) and for the LAST tools
before a reply (``write_file`` / ``terminal``).  The drain-path handler
always flushed; this test locks the two paths to the same behavior.
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


class FakeCodeBlockTelegramAdapter:
    """Faithful Telegram fake: supports message editing AND code blocks."""

    name = "fake-telegram"
    MAX_MESSAGE_LENGTH = 4000
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


class FakeInterimResetAgent(BaseFakeTimingAgent):
    """Mirrors the reported screenshot.

    ``delegate_task`` completes, THEN an interim assistant message lands
    (fires ``__reset__`` via the stream consumer's ``on_new_message``),
    THEN ``write_file`` + ``terminal`` run as the last tools.  Without the
    flush-before-reset fix, ``delegate_task`` loses its ``· 2m 05s`` suffix.
    """

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **kwargs):
        cb = self.tool_progress_callback
        assert cb is not None
        interim = self.interim_assistant_callback

        cb("tool.started", "delegate_task", preview=None,
           args={"tasks": [{"goal": "review diff", "profile": "verify-php-standards"}]})
        cb("tool.completed", "delegate_task", duration=125.2)

        # Interim commentary → stream consumer sends a message → __reset__.
        if interim:
            interim("Preflight: PASS, no blockers. Pushing.")
            time.sleep(0.3)

        cb("tool.started", "write_file", preview="/tmp/x.out", args={"path": "/tmp/x.out"})
        cb("tool.completed", "write_file", duration=0.5)
        cb("tool.started", "terminal", preview="ls", args={"command": "cd /repo && ls"})
        cb("tool.completed", "terminal", duration=0.01)
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


def _user_config(interim):
    return {
        "display": {
            "interim_assistant_messages": interim,
            "platforms": {
                "telegram": {
                    "streaming": False,
                    "tool_progress": "all",
                    "delegate_task_args": True,
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

    adapter = FakeCodeBlockTelegramAdapter()
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
    return result, rendered, adapter


@pytest.mark.asyncio
async def test_duration_suffix_survives_reset_from_interim_content(monkeypatch):
    """delegate_task completion straddles an interim message; its duration
    must still be published (flushed before __reset__ abandons the bubble)."""
    result, rendered, _adapter = await _run(
        monkeypatch, _user_config(interim=True), FakeInterimResetAgent
    )

    assert result["final_response"] == "done"
    # The tool whose completion straddled the interim __reset__ keeps its suffix.
    # NOTE: the delegate_task card now renders as the roster-style header
    # ("🔀 Delegate task — N tasks") instead of the old "delegate_task
    # parameters" JSON card; the duration suffix still attaches to that line.
    assert "🔀 Delegate task — 1 tasks · 2m 05s" in rendered
    # The last tools before the reply keep their suffixes too.
    assert 'write_file: "/tmp/x.out" · 0.5s' in rendered


class FakeResetMidToolAgent(BaseFakeTimingAgent):
    """A __reset__ lands BETWEEN a terminal's start and completed.

    Interim commentary arrives while the terminal command is still running.
    The stream consumer sends it asynchronously, firing __reset__, which used
    to clear the pending-tool-line index — so the terminal completion could
    not find its start row and rendered as a SEPARATE fallback line
    (``✅ terminal completed in 1.4s``) instead of an inline ``· 1.4s``
    suffix on the ``💻 terminal`` block.
    """

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **kwargs):
        cb = self.tool_progress_callback
        assert cb is not None
        interim = self.interim_assistant_callback

        # Terminal starts, then commentary arrives mid-run (fires __reset__),
        # then the terminal completes.
        cb("tool.started", "terminal", preview="acli jira workitem view ASPD-30171",
           args={"command": "acli jira workitem view ASPD-30171"})
        if interim:
            interim("Let me check the ticket details.")
            time.sleep(0.4)  # let the async __reset__ land before completion
        cb("tool.completed", "terminal", duration=1.4)
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


@pytest.mark.asyncio
async def test_terminal_duration_inline_when_reset_lands_mid_tool(monkeypatch):
    """A __reset__ between terminal start and completion must NOT produce a
    standalone '✅ terminal completed in 1.4s' fallback line — the suffix
    belongs inline on the terminal block."""
    result, rendered, adapter = await _run(
        monkeypatch, _user_config(interim=True), FakeResetMidToolAgent
    )

    assert result["final_response"] == "done"
    # Inline suffix on the terminal block, not a separate fallback row.
    assert "💻 terminal · 1.4s" in rendered
    assert "✅ terminal completed in 1.4s" not in rendered
