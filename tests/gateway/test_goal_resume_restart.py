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
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import goals


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # get_hermes_home() prefers the context-local override over the env
    # var, so a set_hermes_home_override() leaked by ANY earlier test in
    # this xdist worker would silently point the goals DB at a dead tmp
    # dir and make resume enqueue nothing (CI-only flake). Pin the
    # override to THIS home so the fixture is immune to leaks.
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    token = set_hermes_home_override(str(home))
    goals._DB_CACHE.clear()
    # Pre-warm the SessionDB cache from sync context so async GoalManager
    # writes never race the bounded loop-thread bootstrap window on loaded
    # CI runners (goal silently unpersisted; main run 33455779041).
    goals._get_session_db()
    yield home
    try:
        reset_hermes_home_override(token)
    except Exception:
        pass
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

    def test_cli_persistence_drop_reports_failure_without_success_notice(
        self, hermes_home, monkeypatch
    ):
        sid = f"sid-cli-drop-{uuid.uuid4().hex}"
        cli = _make_cli(sid)

        class DroppedWriteDB:
            def get_meta(self, _key):
                return None

            def set_meta(self, _key, _value):
                return None

            def compare_and_set_meta(self, _key, _expected, _replacement):
                raise OSError("simulated persistence drop")

        monkeypatch.setattr(goals, "_get_session_db", lambda: DroppedWriteDB())
        notices = []
        with patch("cli._cprint", side_effect=notices.append):
            cli._handle_goal_command("goal must persist")

        rendered = "\n".join(str(item) for item in notices)
        assert "may have committed" in rendered.lower()
        assert "do not retry blindly" in rendered.lower()
        assert "Goal set" not in rendered
        assert cli._pending_input.empty()

    @pytest.mark.parametrize("failure", ["missing", "read", "invalid"])
    def test_fresh_cli_status_reports_unavailable_persistence(
        self, hermes_home, monkeypatch, failure
    ):
        sid = f"sid-cli-status-{failure}-{uuid.uuid4().hex}"
        cli = _make_cli(sid)

        class StatusDB:
            def get_meta(self, _key):
                if failure == "read":
                    raise OSError("status read failed")
                return "{invalid goal json"

        monkeypatch.setattr(
            goals,
            "_get_session_db",
            (lambda: None) if failure == "missing" else (lambda: StatusDB()),
        )
        notices = []
        with patch("cli._cprint", side_effect=notices.append):
            cli._handle_goal_command("goal status")

        rendered = "\n".join(str(item) for item in notices)
        assert "No active goal" not in rendered
        assert "unavailable" in rendered.lower() or "failed" in rendered.lower()

    def test_fresh_cli_status_reports_verified_empty_goal(self, hermes_home):
        cli = _make_cli(f"sid-cli-status-empty-{uuid.uuid4().hex}")
        notices = []
        with patch("cli._cprint", side_effect=notices.append):
            cli._handle_goal_command("goal status")
        assert "No active goal" in "\n".join(str(item) for item in notices)

    @pytest.mark.parametrize("failure", ["missing", "read", "invalid"])
    @pytest.mark.parametrize("command", ["subgoal", "goal gate list"])
    def test_human_goal_reads_never_report_persistence_failure_as_absence(
        self, hermes_home, monkeypatch, failure, command
    ):
        cli = _make_cli(f"sid-cli-human-read-{failure}-{uuid.uuid4().hex}")

        class StatusDB:
            def get_meta(self, _key):
                if failure == "read":
                    raise OSError("private human-read detail")
                return "{invalid goal json"

        monkeypatch.setattr(
            goals,
            "_get_session_db",
            (lambda: None) if failure == "missing" else (lambda: StatusDB()),
        )
        notices = []
        with patch("cli._cprint", side_effect=notices.append):
            if command == "subgoal":
                cli._handle_subgoal_command(command)
            else:
                cli._handle_goal_command(command)

        rendered = "\n".join(str(item) for item in notices)
        assert "Goal status unavailable" in rendered
        assert "No active goal" not in rendered
        assert "(no active goal)" not in rendered
        assert "private human-read detail" not in rendered

    def test_indeterminate_cli_mutation_does_not_claim_state_is_unchanged(
        self, hermes_home
    ):
        cli = _make_cli(f"sid-cli-indeterminate-{uuid.uuid4().hex}")

        class IndeterminateManager:
            def set(self, *_args, **_kwargs):
                raise goals.GoalMutationOutcomeUnknownError(
                    "persisted goal read-back failed"
                )

        cli._get_goal_manager = lambda: IndeterminateManager()
        notices = []
        with patch("cli._cprint", side_effect=notices.append):
            cli._handle_goal_command("goal publish carefully")

        rendered = "\n".join(str(item) for item in notices)
        assert "may have committed" in rendered
        assert "do not retry blindly" in rendered.lower()
        assert "unchanged" not in rendered.lower()


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
        event = _resume_event()

        response = await GatewayRunner._handle_goal_command(runner, event)

        assert "resume" in response.lower() or "Goal" in response
        pending = adapter._pending_messages.get(_GW_KEY)
        assert pending is not None, (
            "/goal resume must enqueue the continuation on the adapter FIFO "
            "— otherwise the goal sits idle until the next real user message"
        )
        assert pending.text.startswith("[Continuing toward your standing goal]")
        assert pending.internal is False
        assert pending.allow_gateway_control is False
        assert pending.goal_continuation is True
        assert pending.source is event.source
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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", ["missing", "read", "invalid"])
    async def test_fresh_gateway_status_reports_unavailable_persistence(
        self, hermes_home, monkeypatch, failure
    ):
        runner, _adapter = _make_runner()

        class StatusDB:
            def get_meta(self, _key):
                if failure == "read":
                    raise OSError("status read failed")
                return "{invalid goal json"

        monkeypatch.setattr(
            goals,
            "_get_session_db",
            (lambda: None) if failure == "missing" else (lambda: StatusDB()),
        )
        event = _resume_event()
        event.text = "/goal status"

        response = await GatewayRunner._handle_goal_command(runner, event)

        assert "No active goal" not in response
        assert "unavailable" in response.lower() or "failed" in response.lower()

    @pytest.mark.asyncio
    async def test_fresh_gateway_status_reports_verified_empty_goal(self, hermes_home):
        runner, _adapter = _make_runner()
        event = _resume_event()
        event.text = "/goal status"
        response = await GatewayRunner._handle_goal_command(runner, event)
        assert "No active goal" in response

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", ["missing", "read", "invalid"])
    @pytest.mark.parametrize("command", ["subgoal", "goal gate list"])
    async def test_human_goal_reads_never_report_persistence_failure_as_absence(
        self, hermes_home, monkeypatch, failure, command
    ):
        runner, _adapter = _make_runner()

        class StatusDB:
            def get_meta(self, _key):
                if failure == "read":
                    raise OSError("private gateway human-read detail")
                return "{invalid goal json"

        monkeypatch.setattr(
            goals,
            "_get_session_db",
            (lambda: None) if failure == "missing" else (lambda: StatusDB()),
        )
        event = _resume_event()
        event.text = f"/{command}"
        if command == "subgoal":
            response = await GatewayRunner._handle_subgoal_command(runner, event)
        else:
            response = await GatewayRunner._handle_goal_command(runner, event)

        assert "Goal status unavailable" in response
        assert "No active goal" not in response
        assert "(no active goal)" not in response
        assert "private gateway human-read detail" not in response

    @pytest.mark.asyncio
    async def test_indeterminate_gateway_mutation_does_not_claim_state_is_unchanged(
        self, hermes_home
    ):
        runner, _adapter = _make_runner()

        class IndeterminateManager:
            def set(self, *_args, **_kwargs):
                raise goals.GoalMutationOutcomeUnknownError(
                    "persisted goal read-back failed"
                )

        async def get_manager(_event):
            return IndeterminateManager(), _FakeSessionEntry()

        runner._get_goal_manager_for_event = get_manager
        event = _resume_event()
        event.text = "/goal publish carefully"
        response = await GatewayRunner._handle_goal_command(runner, event)

        assert "may have committed" in response
        assert "do not retry blindly" in response.lower()
        assert "unchanged" not in response.lower()
