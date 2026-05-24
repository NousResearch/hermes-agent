import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import goals


class _FakeSessionEntry:
    session_id = "sid-gateway-goal-config"


class _FakeSessionStore:
    def __init__(self):
        self.entry = _FakeSessionEntry()

    def get_or_create_session(self, source):
        return self.entry

    def _generate_session_key(self, source):
        return "agent:main:discord:channel:goal-config"


def _runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {}
    runner._queued_events = {}
    runner._kanban_notifier_profile = "default"
    return runner


def _source():
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat-goal-config",
        chat_type="channel",
        user_id="user-goal-config",
    )


def _event(text: str, source: SessionSource | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source or _source(),
        message_id=f"msg-{text.rsplit(' ', 1)[-1]}",
    )


@pytest.mark.asyncio
async def test_gateway_goal_uses_goals_max_turns_from_full_config(tmp_path, monkeypatch):
    """Gateway /goal should honor top-level goals.max_turns from config.yaml."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("goals:\n  max_turns: 7\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = _runner()
    event = _event("/goal ship the benchmark")

    response = await GatewayRunner._handle_goal_command(runner, event)

    try:
        assert "⊙ Goal set (7-turn budget): ship the benchmark" in response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.max_turns == 7
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_gateway_goal_extended_creates_ready_kanban_task_not_inline(tmp_path, monkeypatch):
    """/goal extended should hand off to Kanban without starting inline continuation."""
    home = tmp_path / ".hermes"
    home.mkdir()
    kanban_db = tmp_path / "kanban.db"
    (home / "config.yaml").write_text(
        "goals:\n  max_turns: 7\nkanban:\n  default_assignee: codex\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_db))
    goals._DB_CACHE.clear()

    runner = _runner()
    event = _event("/goal extended ship the benchmark")

    response = await GatewayRunner._handle_goal_command(runner, event)

    try:
        assert "Extended goal handed to Kanban task" in response
        assert "assignee: codex" in response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.goal == "ship the benchmark"
        assert state.mode == "extended"
        assert state.inline_active is False
        assert state.kanban_task_id
        assert state.kanban_board == "default"
        assert not goals.GoalManager("sid-gateway-goal-config").is_active()
        assert goals.GoalManager("sid-gateway-goal-config").has_goal()

        from hermes_cli import kanban_db as kb

        conn = kb.connect(db_path=kanban_db)
        try:
            task = kb.get_task(conn, state.kanban_task_id)
        finally:
            conn.close()
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "codex"
        assert task.session_id == "sid-gateway-goal-config"
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_gateway_goal_extended_persists_board_and_controls_stored_board(tmp_path, monkeypatch):
    """Extended-goal lifecycle controls must use the board captured at creation."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "kanban:\n  default_assignee: codex\n  extended_goal_board: long-goals\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = _runner()
    source = _source()

    response = await GatewayRunner._handle_goal_command(
        runner, _event("/goal extended ship the benchmark", source)
    )

    try:
        assert "Extended goal handed to Kanban task" in response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.kanban_task_id
        assert state.kanban_board == "long-goals"

        # Simulate a config/current-board change after creation. Clear must still
        # archive the original task instead of resolving the new board.
        (home / "config.yaml").write_text(
            "kanban:\n  default_assignee: codex\n  extended_goal_board: other-goals\n",
            encoding="utf-8",
        )

        clear_response = await GatewayRunner._handle_goal_command(
            runner, _event("/goal clear", source)
        )
        assert "Goal cleared" in clear_response
        assert f"Kanban task {state.kanban_task_id} archived" in clear_response

        reloaded = goals.GoalManager("sid-gateway-goal-config").state
        assert reloaded is None or reloaded.status == "cleared"

        from hermes_cli import kanban_db as kb

        conn = kb.connect(board="long-goals")
        try:
            task = kb.get_task(conn, state.kanban_task_id)
        finally:
            conn.close()
        assert task is not None
        assert task.status == "archived"
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_gateway_goal_extended_lifecycle_failures_preserve_goal_state(tmp_path, monkeypatch):
    """Kanban control errors must not sever local GoalState from the durable task."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = _runner()
    source = _source()

    def explode(*args, **kwargs):
        raise RuntimeError("kanban db unavailable")

    runner._control_extended_goal_kanban_task = explode

    try:
        mgr = goals.GoalManager("sid-gateway-goal-config")
        mgr.set(
            "ship safely",
            mode="extended",
            inline_active=False,
            kanban_task_id="task-1",
            kanban_board="long-goals",
        )
        pause_response = await GatewayRunner._handle_goal_command(
            runner, _event("/goal pause", source)
        )
        assert "Goal pause aborted" in pause_response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.status == "active"
        assert state.kanban_task_id == "task-1"

        state = mgr.pause(reason="already-paused")
        assert state is not None and state.status == "paused"
        resume_response = await GatewayRunner._handle_goal_command(
            runner, _event("/goal resume", source)
        )
        assert "Goal resume aborted" in resume_response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.status == "paused"
        assert state.kanban_task_id == "task-1"

        state = mgr.resume()
        assert state is not None and state.status == "active"
        clear_response = await GatewayRunner._handle_goal_command(
            runner, _event("/goal clear", source)
        )
        assert "Goal clear aborted" in clear_response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.status == "active"
        assert state.kanban_task_id == "task-1"
    finally:
        goals._DB_CACHE.clear()
