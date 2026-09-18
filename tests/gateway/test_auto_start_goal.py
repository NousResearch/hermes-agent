import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import goals


class _SessionEntry:
    session_id = "sid-auto-start-goal"


class _SessionStore:
    def get_or_create_session(self, source, **_kwargs):
        return _SessionEntry()


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _SessionStore()
    return runner


def _event(text: str, *, internal: bool = False) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD, chat_id="chat", chat_type="channel", user_id="user"
        ),
        internal=internal,
    )


@pytest.mark.asyncio
async def test_auto_start_sets_goal_for_normal_external_message(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "goals:\n  auto_start: true\n  max_turns: 3\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    goals._get_session_db()

    try:
        await GatewayRunner._auto_start_goal_for_inbound_event(_runner(), _event("finish the report"))
        state = goals.GoalManager("sid-auto-start-goal").state
        assert state is not None
        assert state.goal == "finish the report"
        assert state.max_turns == 3
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_auto_start_ignores_commands_and_internal_continuations(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("goals:\n  auto_start: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    goals._get_session_db()
    manager = goals.GoalManager("sid-auto-start-goal")
    manager.set("keep this goal")

    try:
        runner = _runner()
        await GatewayRunner._auto_start_goal_for_inbound_event(runner, _event("/status"))
        await GatewayRunner._auto_start_goal_for_inbound_event(runner, _event("continue", internal=True))
        assert goals.GoalManager("sid-auto-start-goal").state.goal == "keep this goal"
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_auto_start_is_off_for_false_string(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("goals:\n  auto_start: 'false'\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    goals._get_session_db()

    try:
        await GatewayRunner._auto_start_goal_for_inbound_event(_runner(), _event("do not start"))
        assert goals.GoalManager("sid-auto-start-goal").state is None
    finally:
        goals._DB_CACHE.clear()