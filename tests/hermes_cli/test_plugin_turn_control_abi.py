import asyncio
import time
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.plugins import (
    CommandContext,
    PluginContext,
    PluginManager,
    PluginManifest,
    TurnControlContext,
    TurnDirective,
    dispatch_plugin_command,
    dispatch_plugin_command_async,
    invoke_turn_controllers,
)


def _manager():
    mgr = PluginManager()
    mgr._discovered = True
    return mgr


def test_legacy_plugin_command_signature_still_receives_raw_args():
    mgr = _manager()
    ctx = PluginContext(PluginManifest(name="plug"), mgr)
    seen = []

    ctx.register_command("legacy", lambda raw: seen.append(raw) or "ok")

    with patch("hermes_cli.plugins._plugin_manager", mgr):
        assert dispatch_plugin_command("legacy", "raw args") == "ok"

    assert seen == ["raw args"]


def test_context_aware_plugin_command_receives_cli_session_context():
    mgr = _manager()
    ctx = PluginContext(PluginManifest(name="plug"), mgr)
    seen = []

    def handler(command_ctx, raw):
        seen.append((command_ctx.session_id, command_ctx.surface, raw))
        return "ok"

    ctx.register_command("ctx", handler, context_aware=True)

    async def enqueue(_: str) -> bool:
        return True

    command_ctx = CommandContext(
        surface="cli",
        session_id="cli-sess",
        platform="cli",
        source=None,
        task_id="cli-sess",
        metadata={},
        enqueue_followup=enqueue,
    )
    with patch("hermes_cli.plugins._plugin_manager", mgr):
        assert dispatch_plugin_command("ctx", "start", command_ctx) == "ok"

    assert seen == [("cli-sess", "cli", "start")]


@pytest.mark.asyncio
async def test_async_context_aware_plugin_command_can_enqueue_followup():
    mgr = _manager()
    ctx = PluginContext(PluginManifest(name="plug"), mgr)
    queued = []

    async def handler(command_ctx, raw):
        assert raw == "start"
        return await command_ctx.enqueue_followup("continue")

    ctx.register_command("ctx-async", handler, context_aware=True)

    async def enqueue(prompt: str) -> bool:
        queued.append(prompt)
        return True

    command_ctx = CommandContext(
        surface="gateway",
        session_id="gw-sess",
        platform="telegram",
        source=None,
        task_id="task",
        metadata={},
        enqueue_followup=enqueue,
    )
    with patch("hermes_cli.plugins._plugin_manager", mgr):
        assert await dispatch_plugin_command_async("ctx-async", "start", command_ctx) is True

    assert queued == ["continue"]


@pytest.mark.asyncio
async def test_turn_controller_accepts_one_directive_and_isolates_exceptions(caplog):
    mgr = _manager()
    ctx = PluginContext(PluginManifest(name="plug"), mgr)

    def broken(_ctx):
        raise RuntimeError("boom")

    def continuing(_ctx):
        return TurnDirective(
            action="continue",
            continuation_prompt="next",
            dedupe_key="dedupe-1",
        )

    def second(_ctx):
        return TurnDirective(action="continue", continuation_prompt="second")

    ctx.register_turn_controller("broken", broken, priority=10)
    ctx.register_turn_controller("continuing", continuing, priority=20)
    ctx.register_turn_controller("second", second, priority=30)

    turn_ctx = TurnControlContext(
        surface="gateway",
        session_id="sess",
        platform="telegram",
        source=None,
        task_id="task",
        turn_id="turn-1",
        user_message="hi",
        final_response="done",
        interrupted=False,
        background_processes=[],
    )
    with patch("hermes_cli.plugins._plugin_manager", mgr), caplog.at_level("WARNING"):
        directive = await invoke_turn_controllers(turn_ctx)

    assert directive == TurnDirective(
        action="continue",
        continuation_prompt="next",
        dedupe_key="dedupe-1",
    )
    assert "raised" in caplog.text


@pytest.mark.asyncio
async def test_sync_plugin_command_timeout_does_not_block_event_loop():
    mgr = _manager()
    ctx = PluginContext(PluginManifest(name="plug"), mgr)

    def slow_handler(_raw):
        time.sleep(0.05)
        return "late"

    ctx.register_command("slow", slow_handler)
    with patch("hermes_cli.plugins._plugin_manager", mgr):
        with pytest.raises(asyncio.TimeoutError):
            await dispatch_plugin_command_async("slow", "", timeout=0.01)


@pytest.mark.asyncio
async def test_sync_turn_controller_timeout_equals_noop(monkeypatch, caplog):
    import hermes_cli.plugins as plugins

    mgr = _manager()
    ctx = PluginContext(PluginManifest(name="plug"), mgr)

    def slow_controller(_ctx):
        time.sleep(0.05)
        return TurnDirective(action="continue", continuation_prompt="late")

    ctx.register_turn_controller("slow", slow_controller)
    monkeypatch.setattr(plugins, "_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS", 0.01)
    turn_ctx = TurnControlContext(
        surface="gateway",
        session_id="sess",
        platform="telegram",
        source=None,
        task_id="task",
        turn_id="turn-timeout",
        user_message="hi",
        final_response="done",
        interrupted=False,
        background_processes=[],
    )
    with patch("hermes_cli.plugins._plugin_manager", mgr), caplog.at_level("WARNING"):
        assert await invoke_turn_controllers(turn_ctx) is None
    assert "raised" in caplog.text


@pytest.mark.asyncio
async def test_turn_directive_dedupe_is_scoped_by_session():
    mgr = _manager()
    ctx = PluginContext(PluginManifest(name="plug"), mgr)
    ctx.register_turn_controller(
        "same-key",
        lambda _ctx: TurnDirective(
            action="continue",
            continuation_prompt="next",
            dedupe_key="version-1",
        ),
    )

    def turn_context(session_id):
        return TurnControlContext(
            surface="gateway",
            session_id=session_id,
            platform="telegram",
            source=None,
            task_id=session_id,
            turn_id=f"{session_id}:turn",
            user_message="hi",
            final_response="done",
            interrupted=False,
            background_processes=[],
        )

    with patch("hermes_cli.plugins._plugin_manager", mgr):
        assert await invoke_turn_controllers(turn_context("session-a")) is not None
        assert await invoke_turn_controllers(turn_context("session-b")) is not None
        assert await invoke_turn_controllers(turn_context("session-a")) is None


def test_cli_surface_invokes_turn_controller_and_queues_followup():
    from cli import HermesCLI

    mgr = _manager()
    plugin_ctx = PluginContext(PluginManifest(name="plug"), mgr)
    seen = []

    def controller(turn_ctx):
        seen.append((turn_ctx.surface, turn_ctx.session_id, turn_ctx.user_message, turn_ctx.final_response))
        return TurnDirective(action="continue", continuation_prompt="cli-next")

    plugin_ctx.register_turn_controller("cli-controller", controller)
    cli = object.__new__(HermesCLI)
    cli._pending_input = Queue()
    cli._last_turn_interrupted = False
    cli.session_id = "cli-parent"
    cli.agent = SimpleNamespace(session_id="cli-physical")

    with patch("hermes_cli.plugins._plugin_manager", mgr):
        directive = cli._maybe_continue_plugin_after_turn(
            user_message="hello",
            final_response="done",
        )

    assert directive.action == "continue"
    assert cli._pending_input.get_nowait() == "cli-next"
    assert seen == [("cli", "cli-physical", "hello", "done")]


def test_cli_surface_real_user_pending_preempts_plugin_controller():
    from cli import HermesCLI

    mgr = _manager()
    plugin_ctx = PluginContext(PluginManifest(name="plug"), mgr)
    calls = []
    plugin_ctx.register_turn_controller(
        "should-not-run",
        lambda _ctx: calls.append(True) or TurnDirective(
            action="continue", continuation_prompt="auto"
        ),
    )
    cli = object.__new__(HermesCLI)
    cli._pending_input = Queue()
    cli._pending_input.put("real user")
    cli._last_turn_interrupted = False
    cli.session_id = "cli-parent"
    cli.agent = SimpleNamespace(session_id="cli-physical")

    with patch("hermes_cli.plugins._plugin_manager", mgr):
        assert cli._maybe_continue_plugin_after_turn(
            user_message="hello",
            final_response="done",
        ) is None

    assert calls == []
    assert cli._pending_input.get_nowait() == "real user"


def test_cli_enqueue_followup_fails_closed_when_session_rotates():
    queued = Queue()
    bound_session_id = "old"
    current_session_id = "new"

    async def enqueue(prompt: str) -> bool:
        if current_session_id != bound_session_id:
            return False
        queued.put(prompt)
        return True

    assert asyncio.run(enqueue("next")) is False
    assert queued.empty()
