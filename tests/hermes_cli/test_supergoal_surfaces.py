"""Real slash dispatch and persistence across CLI, Telegram, TUI and Desktop."""
import asyncio
import queue
from types import SimpleNamespace

import pytest

from hermes_cli import goals
from hermes_cli.commands import resolve_command, should_bypass_active_session
from hermes_cli.commands_platforms import telegram_bot_commands


@pytest.mark.parametrize("name", ["supergoal", "sg"])
def test_registry_and_adapter_guard(name):
    command = resolve_command(name)
    assert command is not None
    assert command.name == "supergoal"
    assert command.busy_policy == "dispatch" and command.busy_handler == "goal"
    assert command.argument_mode == "mixed" and not command.desktop
    assert should_bypass_active_session(name)
    assert "supergoal" in {item[0] for item in telegram_bot_commands()}


@pytest.mark.parametrize("name", ["supergoal", "sg"])
def test_cli_process_command_starts_and_resumes_same_mode(name):
    from cli import HermesCLI
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "cli-supergoal"
    cli.config = {}
    cli._pending_resume_sessions = None
    cli._pending_input = queue.Queue()
    cli.conversation_history = []
    mgr = goals.GoalManager(cli.session_id)
    cli._get_goal_manager = lambda: mgr
    assert cli.process_command(f"/{name} build result") is True
    assert goals.load_goal(cli.session_id).mode == "supergoal"
    assert "Do not ask" in cli._pending_input.get_nowait()
    cli.process_command(f"/{name} pause")
    cli.process_command("/goal resume")
    assert goals.load_goal(cli.session_id).mode == "supergoal"
    assert "Do not ask" in cli._pending_input.get_nowait()
    cli.process_command("/goal ordinary objective")
    assert goals.load_goal(cli.session_id).mode == "goal"
    assert cli._pending_input.get_nowait() == "ordinary objective"


@pytest.mark.parametrize("name", ["supergoal", "sg"])
def test_gateway_idle_and_busy_commands_share_state(name):
    from gateway.run_busy import GatewayBusySessionMixin
    from gateway.slash_commands_goals import GatewayGoalCommandsMixin
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import SessionSource
    from gateway.config import Platform
    class Runner(GatewayBusySessionMixin, GatewayGoalCommandsMixin):
        pass
    runner = Runner()
    mgr = goals.GoalManager("gateway-supergoal")
    prompts = []
    async def manager(event):
        return mgr, None
    async def execute(fn, *args):
        return fn(*args)
    runner._get_goal_manager_for_event = manager
    runner._run_in_executor_with_context = execute
    runner._adapter_and_key_for = lambda event: (None, None)
    runner._enqueue_goal_turn = lambda event, text, **kw: prompts.append(text)
    runner._resume_caller_is_admin = lambda source: False
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="test", user_id="user")
    def event(arg):
        return MessageEvent(text=f"/{name} {arg}", message_type=MessageType.TEXT, source=source)
    command = resolve_command(name)
    assert command is not None
    handler = runner._command_handler_table((command.name,))[command.name]
    asyncio.run(handler(event("build result")))
    assert goals.load_goal(mgr.session_id).mode == "supergoal"
    assert "Do not ask" in prompts[-1]
    for control in ("status", "pause", "resume", "show"):
        result = asyncio.run(runner._dispatch_busy_slash_command(event(control), command, "key", source))
        assert "goal" in result.lower()
        assert goals.load_goal(mgr.session_id).mode == "supergoal"
    assert "Do not ask" in prompts[-1]
    denied = asyncio.run(handler(event("gate add echo unauthorized")))
    assert "admin" in denied and not mgr.state.gates
    rejected = asyncio.run(runner._dispatch_busy_slash_command(event("replace objective"), command, "key", source))
    assert "running" in rejected.lower()
    assert mgr.state.goal == "build result"


@pytest.mark.parametrize("name", ["supergoal", "sg"])
@pytest.mark.parametrize("method", ["command.dispatch", "slash.exec"])
def test_tui_desktop_rpc_routes_without_worker(name, method):
    from tui_gateway import server
    sid = "rpc-supergoal"
    server._sessions[sid] = {"session_key": sid}
    def call(arg):
        params = {"session_id": sid, "name": name, "arg": arg}
        if method == "slash.exec":
            params = {"session_id": sid, "command": f"/{name} {arg}"}
        return server._methods[method](1, params)
    try:
        result = call("build result")
        assert "error" not in result, result
        assert result["result"]["type"] == "send"
        assert "Do not ask" in result["result"]["message"]
        assert goals.load_goal(sid).mode == "supergoal"
        assert "slash_worker" not in server._sessions[sid]
        call("pause")
        result = call("resume")
        assert result["result"]["type"] == "send"
        assert "Do not ask" in result["result"]["message"]
        assert goals.load_goal(sid).mode == "supergoal"
        assert "supergoal" in call("status")["result"]["output"].lower()
    finally:
        server._sessions.pop(sid, None)
