"""#75362 — /goal resume must restart work, not just flip persisted state.

After a goal auto-pauses on turn-budget exhaustion, each surface's
``resume`` handler must schedule exactly one canonical
``GoalManager.next_continuation_prompt()`` turn through that surface's
existing input path:

- classic CLI  → ``self._pending_input``
- messaging gateway → the adapter FIFO (``_enqueue_fifo``)

The TUI/Desktop ``command.dispatch`` boundary is covered in
``tests/tui_gateway/test_goal_command.py``.
"""

from __future__ import annotations

import queue
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import goals


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _exhaust_budget(session_id: str, goal_text: str = "ship the benchmark"):
    """Set a 1-turn goal and drive it to budget-exhaustion auto-pause."""
    mgr = goals.GoalManager(session_id)
    mgr.set(goal_text, max_turns=1)
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("continue", "needs more steps", False, None, False),
    ):
        decision = mgr.evaluate_after_turn("worked a bit")
    assert decision["status"] == "paused"
    assert decision["should_continue"] is False
    assert "turn budget exhausted" in (mgr.state.paused_reason or "")
    return mgr


# ──────────────────────────────────────────────────────────────────────
# Classic CLI
# ──────────────────────────────────────────────────────────────────────


def _make_cli(session_id: str):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    cli.session_id = session_id
    cli.agent = MagicMock()
    cli.agent.session_id = session_id
    return cli


class TestCliResumeRestartsWork:
    def test_resume_after_budget_exhaustion_queues_continuation(self, hermes_home):
        sid = f"sid-cli-resume-{uuid.uuid4().hex}"
        cli = _make_cli(sid)
        _exhaust_budget(sid)

        cli._handle_goal_command("goal resume")

        assert not cli._pending_input.empty(), (
            "/goal resume must queue the continuation prompt on _pending_input "
            "— otherwise the goal sits idle until the user types something"
        )
        queued = cli._pending_input.get_nowait()
        assert queued.startswith("[Continuing toward your standing goal]")

        state = goals.GoalManager(sid).state
        assert state.status == "active"
        assert state.turns_used == 0

    def test_resume_without_goal_queues_nothing(self, hermes_home):
        sid = f"sid-cli-noresume-{uuid.uuid4().hex}"
        cli = _make_cli(sid)

        cli._handle_goal_command("goal resume")

        assert cli._pending_input.empty()


# ──────────────────────────────────────────────────────────────────────
# Messaging gateway
# ──────────────────────────────────────────────────────────────────────

_GW_SID = "sid-gateway-goal-resume"
_GW_KEY = "agent:main:discord:channel:goal-resume"


class _FakeSessionEntry:
    session_id = _GW_SID


class _FakeSessionStore:
    def __init__(self):
        self.entry = _FakeSessionEntry()

    def get_or_create_session(self, source, **_kwargs):
        return self.entry

    def _generate_session_key(self, source):
        return _GW_KEY


class _FakeAdapter:
    def __init__(self):
        self._pending_messages = {}


def _make_runner() -> tuple[GatewayRunner, _FakeAdapter]:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    adapter = _FakeAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._queued_events = {}
    return runner, adapter


def _resume_event() -> MessageEvent:
    return MessageEvent(
        text="/goal resume",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-goal-resume",
            chat_type="channel",
            user_id="user-goal-resume",
        ),
        message_id="msg-goal-resume",
    )


class TestGatewayResumeRestartsWork:
    @pytest.mark.asyncio
    async def test_resume_after_budget_exhaustion_enqueues_continuation(
        self, hermes_home
    ):
        runner, adapter = _make_runner()
        _exhaust_budget(_GW_SID)

        response = await GatewayRunner._handle_goal_command(runner, _resume_event())

        assert "resume" in response.lower() or "Goal" in response
        pending = adapter._pending_messages.get(_GW_KEY)
        assert pending is not None, (
            "/goal resume must enqueue the continuation on the adapter FIFO "
            "— otherwise the goal sits idle until the next real user message"
        )
        assert pending.text.startswith("[Continuing toward your standing goal]")
        # The pause/clear stale-work guard must recognize the queued turn as
        # a synthetic goal continuation so it can be cleaned up on /goal pause.
        assert GatewayRunner._is_goal_continuation_event(pending)

        state = goals.GoalManager(_GW_SID).state
        assert state.status == "active"
        assert state.turns_used == 0

    @pytest.mark.asyncio
    async def test_resume_without_goal_enqueues_nothing(self, hermes_home):
        runner, adapter = _make_runner()

        response = await GatewayRunner._handle_goal_command(runner, _resume_event())

        assert "No goal to resume" in response
        assert adapter._pending_messages == {}
