"""Regression: standing /goal must not spin on provider failure or survive /stop.

Observed failure mode (gateway messaging platforms):
1. A session has an active standing /goal (Ralph loop).
2. The configured model starts returning non-retryable provider failures
   (e.g. HTTP 404 model-not-found after a catalog removal).
3. Each failed turn still yields a short final_response / failed=True.
4. The goal judge fail-opens to ``continue`` ("no evidence of progress").
5. Gateway enqueues another synthetic ``[Continuing toward your standing goal]``
   user message → tight loop until turn budget (default 20) is exhausted.
6. Gateway ``/stop`` only interrupted the agent and did NOT pause the goal,
   so the loop resumed between turns / after restart. CLI Ctrl+C already
   auto-pauses goals; gateway did not.

Follow-ups from PR review:
- Sibling-thread /stop must pause goals on EACH interrupted sibling session.
- Interrupt must preserve non-goal pending follow-ups (only drain goal conts).
- Prefer structured failed/failure_reason over free-text markers alone.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE, GoalManager


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _source(uid="user-1", chat_id="chat-1", thread_id=None):
    return SessionSource(
        platform=Platform.TELEGRAM if thread_id is None else Platform.DISCORD,
        chat_id=chat_id,
        user_id=uid,
        thread_id=thread_id,
        chat_type="forum" if thread_id else "dm",
    )


def _reload(sid: str) -> GoalManager:
    return GoalManager(session_id=sid, default_max_turns=20)


class _FakeAdapter:
    def __init__(self):
        self._pending_messages = {}
        self.callbacks = {}
        self._active_sessions = {}
        self.sent = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SimpleNamespace(success=True)

    def register_post_delivery_callback(self, session_key, callback, *, generation=None):
        self.callbacks[session_key] = (generation, callback)

    def get_pending_message(self, session_key):
        return self._pending_messages.pop(session_key, None)

    async def interrupt_session_activity(self, session_key, chat_id, metadata=None):
        return None


class _StoreEntry:
    def __init__(self, session_key, session_id):
        self.session_key = session_key
        self.session_id = session_id


class _FakeStore:
    """SessionStore-shaped stub with optional multi-key entries map."""

    def __init__(self, session_key=None, session_id=None, entries=None):
        if entries is not None:
            self._entries = dict(entries)
        elif session_key is not None:
            self._entries = {session_key: _StoreEntry(session_key, session_id)}
        else:
            self._entries = {}
        self._key = session_key
        self._sid = session_id

    def get_or_create_session(self, source):
        # Prefer exact key if known; otherwise first entry / stored default.
        if self._key and self._key in self._entries:
            return self._entries[self._key]
        if self._entries:
            return next(iter(self._entries.values()))
        return _StoreEntry(self._key or "unknown", self._sid or "")


async def _inline_exec(func, *args):
    return func(*args)


def _wire_post_turn(runner, adapter, *, session_key="telegram:chat-1", max_turns=20):
    runner.adapters = {Platform.TELEGRAM: adapter, Platform.DISCORD: adapter}
    runner.config = SimpleNamespace()
    runner._goal_max_turns_from_config = lambda: max_turns
    runner._session_key_for_source = lambda source: session_key
    runner._adapter_for_source = lambda source: adapter
    runner._enqueue_fifo = MagicMock()
    runner._peek_session_state = lambda key: SimpleNamespace(
        conversation=SimpleNamespace(queued_events=[])
    )
    runner._defer_goal_status_notice_after_delivery = AsyncMock()
    runner._warm_goals_session_db = AsyncMock()
    runner._run_in_executor_with_context = _inline_exec


# ---------------------------------------------------------------------------
# Structured + text failure detectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"failed": True, "failure_reason": "model_not_found"},
        {"failed": True, "failure_reason": "auth"},
        {"failed": True, "failure_reason": "billing"},
        {"failed": True, "failure_reason": "rate_limit"},
        {
            "failed": True,
            "failure_reason": None,
            "error": "HTTP 404: Model 'x' not found",
        },
        {
            "failed": False,
            "final_response": (
                "⚠️ The model provider failed after retries. "
                "I kept raw provider details out of chat."
            ),
        },
    ],
)
def test_structured_provider_failure_detector_positive(kwargs):
    assert GatewayRunner._is_goal_blocking_agent_failure(**kwargs) is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"failed": False, "final_response": "All checks passed; goal done."},
        {"failed": True, "failure_reason": "context_overflow"},
        {"failed": True, "failure_reason": "payload_too_large"},
        {
            "failed": False,
            "final_response": (
                "When debugging APIs, remember that clients often treat missing routes "
                "similarly to missing objects. For example, some docs say that "
                "receiving a not-found status simply means try another path. "
                "Here is a longer explanation of DNS resolution and caching layers "
                "so the body is clearly ordinary assistant prose rather than a "
                "provider failure envelope returned to chat."
            ),
        },
        {"failed": False, "final_response": "The model 'priya' voice sounded good."},
    ],
)
def test_structured_provider_failure_detector_negative(kwargs):
    assert GatewayRunner._is_goal_blocking_agent_failure(**kwargs) is False


# ---------------------------------------------------------------------------
# Post-turn continuation short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_model_not_found_pauses_goal_and_skips_judge(hermes_home):
    sid = f"sid-provider-{uuid.uuid4().hex}"
    GoalManager(session_id=sid, default_max_turns=20).set(
        "Ship a working integration end-to-end"
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = _FakeAdapter()
    _wire_post_turn(runner, adapter)

    with patch("hermes_cli.goals.judge_goal") as judge_mock:
        judge_mock.side_effect = AssertionError("judge must not run on provider failure")
        await runner._post_turn_goal_continuation(
            session_entry=SimpleNamespace(session_id=sid),
            source=_source(),
            final_response="The request failed: Model 'x' not found",
            failed=True,
            failure_reason="model_not_found",
            error="Model 'x' not found",
        )

    state = _reload(sid).state
    assert state.status == "paused"
    assert "provider" in (state.paused_reason or "").lower()
    runner._enqueue_fifo.assert_not_called()


@pytest.mark.asyncio
async def test_text_envelope_still_pauses_when_failed_flag_missing(hermes_home):
    sid = f"sid-env-{uuid.uuid4().hex}"
    GoalManager(session_id=sid, default_max_turns=20).set("Do the thing")

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = _FakeAdapter()
    _wire_post_turn(runner, adapter)

    failure = (
        "⚠️ The model provider failed after retries. I kept raw provider details "
        "out of chat; check gateway logs for diagnostics."
    )
    with patch("hermes_cli.goals.judge_goal") as judge_mock:
        judge_mock.side_effect = AssertionError("judge must not run")
        await runner._post_turn_goal_continuation(
            session_entry=SimpleNamespace(session_id=sid),
            source=_source(),
            final_response=failure,
            failed=False,
        )

    assert _reload(sid).state.status == "paused"
    runner._enqueue_fifo.assert_not_called()


@pytest.mark.asyncio
async def test_context_overflow_does_not_auto_pause_goal(hermes_home):
    sid = f"sid-ctx-{uuid.uuid4().hex}"
    GoalManager(session_id=sid, default_max_turns=5).set("Finish the task")

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = _FakeAdapter()
    _wire_post_turn(runner, adapter, max_turns=5)

    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("continue", "not done yet", False, None, False),
    ):
        await runner._post_turn_goal_continuation(
            session_entry=SimpleNamespace(session_id=sid),
            source=_source(),
            final_response="⚠️ Session too large for the model's context window.",
            failed=True,
            failure_reason="context_overflow",
        )

    # Not auto-paused by provider short-circuit; judge may continue.
    assert _reload(sid).state.status == "active"
    runner._enqueue_fifo.assert_called_once()


@pytest.mark.asyncio
async def test_normal_response_still_evaluates_goal(hermes_home):
    sid = f"sid-ok-{uuid.uuid4().hex}"
    GoalManager(session_id=sid, default_max_turns=5).set("Finish the task")

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = _FakeAdapter()
    _wire_post_turn(runner, adapter, max_turns=5)

    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("done", "verified", False, None, False),
    ):
        await runner._post_turn_goal_continuation(
            session_entry=SimpleNamespace(session_id=sid),
            source=_source(),
            final_response="Done. Verification: all checks passed with concrete output.",
            failed=False,
        )

    assert _reload(sid).state.status == "done"
    runner._enqueue_fifo.assert_not_called()


# ---------------------------------------------------------------------------
# /stop: pause + pending preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_preserves_non_goal_pending_followup(hermes_home):
    sid = f"sid-pend-{uuid.uuid4().hex}"
    GoalManager(session_id=sid, default_max_turns=10).set("Do the thing")

    session_key = "agent:main:telegram:dm:chat-1"
    adapter = _FakeAdapter()
    user_followup = MessageEvent(
        text="please also summarize costs",
        message_type=MessageType.TEXT,
        source=_source(),
    )
    adapter._pending_messages[session_key] = user_followup

    class _Agent:
        def interrupt(self, reason=None):
            return None

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running_agents = {session_key: _Agent()}
    runner.session_store = _FakeStore(session_key, sid)
    runner._is_user_authorized = lambda source: True
    runner._sibling_thread_run_keys = lambda source, key: []
    runner._goal_max_turns_from_config = lambda: 10
    runner._session_key_for_source = lambda source: session_key
    runner._adapter_for_source = lambda source: adapter
    runner._thread_metadata_for_source = lambda source, reply_to_message_id=None: None
    runner._invalidate_session_run_generation = lambda *a, **k: None
    runner._release_running_agent_state = lambda *a, **k: None
    runner._evict_cached_agent = lambda *a, **k: None
    runner._peek_session_state = lambda key: SimpleNamespace(
        turn=SimpleNamespace(agent=_Agent()),
        persistent=SimpleNamespace(pending_command_text=None),
        conversation=SimpleNamespace(queued_events=[]),
    )
    runner.adapters = {Platform.TELEGRAM: adapter}

    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=_source())
    await runner._handle_stop_command(event)

    assert _reload(sid).state.status == "paused"
    # Real user follow-up must survive interrupt+goal drain.
    restored = adapter._pending_messages.get(session_key)
    assert restored is not None
    assert restored.text == "please also summarize costs"


@pytest.mark.asyncio
async def test_stop_with_no_agent_pauses_active_goal_and_drains_queue(hermes_home):
    sid = f"sid-stop-{uuid.uuid4().hex}"
    GoalManager(session_id=sid, default_max_turns=20).set(
        "Keep working until integration is complete"
    )

    session_key = "agent:main:telegram:dm:chat-1"
    adapter = _FakeAdapter()
    adapter._pending_messages[session_key] = MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal="keep going"),
        message_type=MessageType.TEXT,
        source=_source(),
    )

    q_state = SimpleNamespace(
        conversation=SimpleNamespace(
            queued_events=[
                MessageEvent(
                    text=CONTINUATION_PROMPT_TEMPLATE.format(goal="keep going"),
                    message_type=MessageType.TEXT,
                    source=_source(),
                ),
                MessageEvent(
                    text="real user queued follow-up",
                    message_type=MessageType.TEXT,
                    source=_source(),
                ),
            ]
        )
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running_agents = {}
    runner.session_store = _FakeStore(session_key, sid)
    runner._is_user_authorized = lambda source: True
    runner._sibling_thread_run_keys = lambda source, key: []
    runner._goal_max_turns_from_config = lambda: 20
    runner._session_key_for_source = lambda source: session_key
    runner._adapter_for_source = lambda source: adapter
    runner._thread_metadata_for_source = lambda source, reply_to_message_id=None: None
    runner._reply_anchor_for_event = lambda event: None
    runner._peek_session_state = lambda key: q_state
    runner.adapters = {Platform.TELEGRAM: adapter}

    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=_source())
    result = await runner._handle_stop_command(event)

    state = _reload(sid).state
    assert state.status == "paused"
    assert adapter._pending_messages.get(session_key) is None
    remaining = q_state.conversation.queued_events
    assert len(remaining) == 1
    assert remaining[0].text == "real user queued follow-up"
    text = str(getattr(result, "text", result)).lower()
    assert "goal" in text and "paused" in text


@pytest.mark.asyncio
async def test_stop_pauses_goal_on_each_sibling_session(hermes_home):
    """Sibling-thread /stop must pause goals on interrupted siblings."""
    sid_caller = f"sid-caller-{uuid.uuid4().hex}"
    sid_sibling = f"sid-sib-{uuid.uuid4().hex}"
    GoalManager(session_id=sid_sibling, default_max_turns=10).set("Sibling goal work")
    # Caller has no goal; sibling does.

    key_caller = "agent:main:discord:forum:chan:thr:userA"
    key_sibling = "agent:main:discord:forum:chan:thr:userB"

    class _Agent:
        def interrupt(self, reason=None):
            return None

    adapter = _FakeAdapter()
    entries = {
        key_caller: _StoreEntry(key_caller, sid_caller),
        key_sibling: _StoreEntry(key_sibling, sid_sibling),
    }
    store = _FakeStore(entries=entries)
    store._key = key_caller
    store._sid = sid_caller

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running_agents = {key_sibling: _Agent()}  # only sibling is running
    runner.session_store = store
    runner._is_user_authorized = lambda source: True
    runner._sibling_thread_run_keys = lambda source, key: [key_sibling]
    runner._goal_max_turns_from_config = lambda: 10
    runner._session_key_for_source = lambda source: key_caller
    runner._adapter_for_source = lambda source: adapter
    runner._thread_metadata_for_source = lambda source, reply_to_message_id=None: None
    runner._invalidate_session_run_generation = lambda *a, **k: None
    runner._release_running_agent_state = lambda *a, **k: None
    runner._evict_cached_agent = lambda *a, **k: None
    runner._peek_session_state = lambda key: SimpleNamespace(
        turn=SimpleNamespace(agent=_Agent() if key == key_sibling else None),
        persistent=SimpleNamespace(pending_command_text=None),
        conversation=SimpleNamespace(queued_events=[]),
    )
    runner.adapters = {Platform.DISCORD: adapter}

    source = _source(uid="userA", chat_id="chan", thread_id="thr")
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source)
    result = await runner._handle_stop_command(event)

    # Sibling goal paused even though caller issued /stop.
    assert _reload(sid_sibling).state.status == "paused"
    text = str(getattr(result, "text", result)).lower()
    assert "paused" in text
