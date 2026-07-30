"""Regression: standing /goal must not spin on provider failure or survive /stop.

Observed failure mode (gateway messaging platforms):
1. A session has an active standing /goal (Ralph loop).
2. The configured model starts returning non-retryable provider errors
   (e.g. HTTP 404 model-not-found after a catalog removal).
3. Each failed turn still produces a short final_response.
4. The goal judge fail-opens to ``continue`` ("no evidence of progress").
5. Gateway enqueues another synthetic ``[Continuing toward your standing goal]``
   user message → tight loop until turn budget (default 20) is exhausted.
6. Gateway ``/stop`` only interrupted the agent and did NOT pause the goal,
   so the loop resumed between turns / after restart. CLI Ctrl+C already
   auto-pauses goals; gateway did not.

These tests cover the two escape hatches:
- auto-pause on provider-looking final responses (skip judge + no enqueue)
- gateway /stop pauses the goal and drains queued goal continuations
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


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        user_id="user-1",
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


class _StoreEntry:
    def __init__(self, session_key, session_id):
        self.session_key = session_key
        self.session_id = session_id


class _FakeStore:
    """Sync SessionStore-shaped stub used via GatewayRunner.async_session_store."""

    def __init__(self, session_key, session_id):
        self._key = session_key
        self._sid = session_id

    def get_or_create_session(self, source):
        return _StoreEntry(self._key, self._sid)


# ---------------------------------------------------------------------------
# Provider-failure detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "⚠️ The model provider failed after retries. I kept raw provider details out of chat.",
        "HTTP 404: Model 'example-model:free' not found. The requested model does not exist.",
        "Error code: 404 - {'status': 404, 'message': \"Model 'x' not found.\"}",
        "Non-retryable client error: model not found",
        "API call failed: provider authentication failed",
    ],
)
def test_provider_failure_detector_positive(text):
    assert GatewayRunner._looks_like_goal_blocking_provider_failure(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Goal completed successfully. Outbound call placed and audio played back.",
        # long prose that mentions HTTP 404 mid-sentence, not an error envelope
        (
            "When debugging APIs, remember that clients often treat missing routes "
            "similarly to missing objects. For example, some docs say that "
            "receiving a not-found status simply means try another path. "
            "Here is a longer explanation of DNS resolution and caching layers "
            "so the body is clearly ordinary assistant prose rather than a "
            "provider failure envelope returned to chat."
        ),
        "The model 'priya' voice sounded good on the call.",
    ],
)
def test_provider_failure_detector_negative(text):
    assert GatewayRunner._looks_like_goal_blocking_provider_failure(text) is False


# ---------------------------------------------------------------------------
# Post-turn continuation short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_failure_auto_pauses_goal_and_skips_judge(hermes_home):
    sid = f"sid-provider-{uuid.uuid4().hex}"
    GoalManager(session_id=sid, default_max_turns=20).set(
        "Ship a working integration end-to-end"
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = _FakeAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.config = SimpleNamespace()
    runner._goal_max_turns_from_config = lambda: 20
    runner._session_key_for_source = lambda source: "telegram:chat-1"
    runner._adapter_for_source = lambda source: adapter
    runner._enqueue_fifo = MagicMock()
    runner._peek_session_state = lambda key: SimpleNamespace(
        conversation=SimpleNamespace(queued_events=[])
    )
    runner._defer_goal_status_notice_after_delivery = AsyncMock()

    session_entry = SimpleNamespace(session_id=sid)
    source = _source()
    failure = (
        "⚠️ The model provider failed after retries. I kept raw provider details "
        "out of chat; check gateway logs for diagnostics."
    )

    with patch("hermes_cli.goals.judge_goal") as judge_mock:
        judge_mock.side_effect = AssertionError("judge must not run on provider failure")
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=source,
            final_response=failure,
        )

    state = _reload(sid).state
    assert state is not None
    assert state.status == "paused"
    assert "provider" in (state.paused_reason or "").lower()
    runner._enqueue_fifo.assert_not_called()


@pytest.mark.asyncio
async def test_normal_response_still_evaluates_goal(hermes_home):
    sid = f"sid-ok-{uuid.uuid4().hex}"
    GoalManager(session_id=sid, default_max_turns=5).set("Finish the task")

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = _FakeAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.config = SimpleNamespace()
    runner._goal_max_turns_from_config = lambda: 5
    runner._session_key_for_source = lambda source: "telegram:chat-1"
    runner._adapter_for_source = lambda source: adapter
    runner._enqueue_fifo = MagicMock()
    runner._peek_session_state = lambda key: SimpleNamespace(
        conversation=SimpleNamespace(queued_events=[])
    )
    runner._defer_goal_status_notice_after_delivery = AsyncMock()

    session_entry = SimpleNamespace(session_id=sid)
    source = _source()

    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("done", "verified", False, None, False),
    ):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=source,
            final_response="Done. Verification: all checks passed with concrete output.",
        )

    assert _reload(sid).state.status == "done"
    runner._enqueue_fifo.assert_not_called()


# ---------------------------------------------------------------------------
# /stop pauses goal when no agent is running (between Ralph turns)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_with_no_agent_pauses_active_goal_and_drains_queue(hermes_home):
    sid = f"sid-stop-{uuid.uuid4().hex}"
    GoalManager(session_id=sid, default_max_turns=20).set(
        "Keep working until integration is complete"
    )

    session_key = "agent:main:telegram:dm:chat-1"
    adapter = _FakeAdapter()
    cont = MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal="keep going"),
        message_type=MessageType.TEXT,
        source=_source(),
    )
    adapter._pending_messages[session_key] = cont

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
    assert "stop" in (state.paused_reason or "").lower()
    assert adapter._pending_messages.get(session_key) is None
    remaining = q_state.conversation.queued_events
    assert len(remaining) == 1
    assert remaining[0].text == "real user queued follow-up"
    text = str(getattr(result, "text", result)).lower()
    assert "goal" in text and "paused" in text


@pytest.mark.asyncio
async def test_stop_with_running_agent_also_pauses_goal(hermes_home):
    sid = f"sid-stop-run-{uuid.uuid4().hex}"
    GoalManager(session_id=sid, default_max_turns=10).set("Do the thing")

    session_key = "agent:main:telegram:dm:chat-1"
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running_agents = {session_key: object()}
    runner.session_store = _FakeStore(session_key, sid)
    runner._is_user_authorized = lambda source: True
    runner._sibling_thread_run_keys = lambda source, key: []
    runner._goal_max_turns_from_config = lambda: 10
    runner._session_key_for_source = lambda source: session_key
    runner._peek_session_state = lambda key: SimpleNamespace(
        conversation=SimpleNamespace(queued_events=[])
    )
    runner.adapters = {Platform.TELEGRAM: _FakeAdapter()}

    interrupted = []

    async def _fake_interrupt(session_key, source, *, interrupt_reason, invalidation_reason):
        interrupted.append((session_key, invalidation_reason))

    runner._interrupt_and_clear_session = _fake_interrupt

    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=_source())
    result = await runner._handle_stop_command(event)

    assert interrupted
    state = _reload(sid).state
    assert state.status == "paused"
    text = str(getattr(result, "text", result)).lower()
    assert "paused" in text
