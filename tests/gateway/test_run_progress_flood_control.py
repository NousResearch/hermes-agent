"""Tests for flood-control handling in the gateway's tool-progress loop.

When a progress-bubble edit is rejected with Telegram flood control, the
gateway used to answer it with a fresh ``adapter.send()`` into the same
chat.  That extra API call is charged against the same penalty the edit
just hit, so Telegram escalates the retry-after: a ~50s back-off compounds
into multi-thousand-second bans during which nothing reaches the chat while
the agent keeps completing turns (#76494).

The contract these tests pin: a flood-controlled edit backs off and keeps
editing — it never answers with another message.  Failures that are
genuinely not recoverable still fall back to sending, because that is the
only way the user sees progress once editing is broken.
"""

import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource

# The progress loop throttles edits to _PROGRESS_EDIT_INTERVAL (1.5s) and it
# is a local in the production function, so it cannot be monkeypatched.
#
# Timing matters here in a way that is easy to get wrong: an event that lands
# while the throttle is still closed is drained into the pending buffer and
# the loop then goes back to waiting on the queue — no edit is attempted until
# something else wakes it.  So the second event must arrive *after* the
# throttle has already expired, which is why the gap below exceeds 1.5s
# rather than being the "obvious" short sleep.
_THROTTLE_EXPIRY_GAP = 2.0
_EDIT_SETTLE_WAIT = 0.8


class _EditFailingAdapter(BasePlatformAdapter):
    """Records sends/edits and fails every edit with a caller-supplied error."""

    def __init__(self, edit_error: str, retryable: bool = False):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self._edit_error = edit_error
        self._edit_retryable = retryable
        self.sent = []
        self.edits = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id=f"msg-{len(self.sent)}")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append({"message_id": message_id, "content": content})
        return SendResult(
            success=False,
            error=self._edit_error,
            retryable=self._edit_retryable,
        )

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class _TwoToolAgent:
    """Emits one tool event to open the bubble, then a second to force an edit."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        # First event: no bubble exists yet, so the loop sends one and
        # remembers its message_id.
        self.tool_progress_callback("tool.started", "web_search", "opening bubble", {})
        time.sleep(_THROTTLE_EXPIRY_GAP)
        # Second event: a bubble now exists and the throttle has expired, so
        # the loop edits it immediately — and the adapter rejects that edit.
        self.tool_progress_callback("tool.started", "web_search", "forces an edit", {})
        time.sleep(_EDIT_SETTLE_WAIT)
        return {"final_response": "done", "messages": [], "api_calls": 1}


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
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


async def _run_with_edit_error(monkeypatch, tmp_path, session_id, edit_error, retryable=False):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _TwoToolAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = _EditFailingAdapter(edit_error, retryable=retryable)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "fake"},
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=source,
        session_id=session_id,
        session_key="agent:main:telegram:group:-1001:17585",
    )
    return adapter, result


@pytest.mark.asyncio
async def test_flood_controlled_edit_does_not_resend_into_the_same_chat(monkeypatch, tmp_path):
    """The reported escalation: a flood-controlled edit must not answer with a send.

    Telegram charges the penalty per API call against the chat, so the
    fallback send is what turns a ~50s back-off into an hours-long ban.
    """
    adapter, _ = await _run_with_edit_error(
        monkeypatch,
        tmp_path,
        "sess-flood",
        "Too Many Requests: retry after 50",
    )

    assert adapter.edits, "test bug: the loop never attempted an edit, so nothing was exercised"
    assert len(adapter.sent) == 1, (
        "flood-controlled edit triggered an extra send into the same chat "
        f"({len(adapter.sent)} sends) — this is the burst that escalates the "
        "retry-after into a multi-hour ban"
    )


@pytest.mark.asyncio
async def test_flood_wording_without_retry_after_is_also_treated_as_flood(monkeypatch, tmp_path):
    """Both spellings the adapters surface ('flood', 'retry after') take the branch."""
    adapter, _ = await _run_with_edit_error(
        monkeypatch,
        tmp_path,
        "sess-flood-wording",
        "Flood control exceeded",
    )

    assert adapter.edits, "test bug: the loop never attempted an edit"
    assert len(adapter.sent) == 1, (
        "a 'Flood control exceeded' edit failure still fell through to a send"
    )


@pytest.mark.asyncio
async def test_flood_controlled_edit_keeps_editing_enabled(monkeypatch, tmp_path):
    """Flood is transient, so can_edit must survive it and later ticks still edit.

    Disabling edits would silently downgrade the whole turn to one message
    per tool call, which is the pre-v0.9 behavior users opted out of.
    """
    adapter, _ = await _run_with_edit_error(
        monkeypatch,
        tmp_path,
        "sess-flood-keeps-editing",
        "Too Many Requests: retry after 50",
    )

    assert len(adapter.edits) >= 1, "test bug: no edit was attempted at all"
    assert len(adapter.sent) == 1, (
        "editing was disabled by flood control — subsequent progress fell back "
        "to sending new messages instead of retrying the edit"
    )


@pytest.mark.asyncio
async def test_non_flood_edit_failure_still_falls_back_to_send(monkeypatch, tmp_path):
    """The legitimate fallback survives: when editing is truly broken, send.

    This is the branch the flood fix must not take away — without it a
    chat whose bubble was deleted would show no progress at all.
    """
    adapter, _ = await _run_with_edit_error(
        monkeypatch,
        tmp_path,
        "sess-not-flood",
        "Bad Request: message to edit not found",
    )

    assert adapter.edits, "test bug: the loop never attempted an edit"
    assert len(adapter.sent) >= 2, (
        "a non-recoverable edit failure no longer falls back to sending — "
        "progress would silently stop reaching the chat"
    )
