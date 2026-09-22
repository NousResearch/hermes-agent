"""Tests for plugin message injection across CLI and gateway hosts."""

import asyncio
import concurrent.futures
from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _context(name: str = "notify-plugin") -> tuple[PluginContext, PluginManager]:
    manager = PluginManager()
    manifest = PluginManifest(name=name, key=name, source="user")
    return PluginContext(manifest, manager), manager


def _write_plugin_config(tmp_path, monkeypatch, entry: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"entries": {"notify-plugin": entry}}})
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))


def test_cli_idle_injection_keeps_existing_queue_behaviour():
    context, manager = _context()
    cli = SimpleNamespace(
        _agent_running=False,
        _pending_input=SimpleQueue(),
        _interrupt_queue=SimpleQueue(),
    )
    manager._cli_ref = cli

    assert context.inject_message("new input") is True
    assert cli._pending_input.get_nowait() == "new input"
    assert cli._interrupt_queue.empty()


def test_cli_running_injection_keeps_existing_interrupt_behaviour():
    context, manager = _context()
    cli = SimpleNamespace(
        _agent_running=True,
        _pending_input=SimpleQueue(),
        _interrupt_queue=SimpleQueue(),
    )
    manager._cli_ref = cli

    assert context.inject_message("status", "system") is True
    assert cli._interrupt_queue.get_nowait() == "[system] status"
    assert cli._pending_input.empty()


def test_gateway_injection_requires_session_key(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)

    assert context.inject_message("wake up") is False
    injector.assert_not_called()


def test_gateway_injection_requires_explicit_permission(tmp_path, monkeypatch):
    _write_plugin_config(tmp_path, monkeypatch, {})
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)

    assert (
        context.inject_message(
            "wake up",
            session_key="agent:main:telegram:dm:42",
        )
        is False
    )
    injector.assert_not_called()


def test_gateway_injection_does_not_treat_string_as_permission(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": "false"},
    )
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)

    assert (
        context.inject_message(
            "wake up",
            session_key="agent:main:telegram:dm:42",
        )
        is False
    )
    injector.assert_not_called()


def test_gateway_injection_fails_closed_when_config_cannot_be_read():
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)

    with patch(
        "hermes_cli.plugins.load_config_readonly",
        side_effect=OSError("config unavailable"),
    ):
        assert (
            context.inject_message(
                "wake up",
                session_key="agent:main:telegram:dm:42",
            )
            is False
        )

    injector.assert_not_called()


def test_gateway_injection_requires_live_host(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()

    assert manager.has_gateway_message_injector is False
    assert (
        context.inject_message(
            "wake up",
            session_key="agent:main:telegram:dm:42",
        )
        is False
    )


def test_gateway_injection_passes_host_owned_plugin_identity(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)

    result = context.inject_message(
        "wake up",
        role="system",
        session_key="agent:main:telegram:dm:42",
    )

    assert result is True
    injector.assert_called_once_with(
        session_key="agent:main:telegram:dm:42",
        content="[system] wake up",
        plugin_id="notify-plugin",
    )


def test_gateway_injection_returns_host_rejection(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    manager.set_gateway_message_injector(
        object(),
        MagicMock(return_value=False),
    )

    assert (
        context.inject_message(
            "wake up",
            session_key="agent:main:telegram:dm:42",
        )
        is False
    )


def test_gateway_injection_fails_closed_on_host_exception(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    injector = MagicMock(side_effect=RuntimeError("gateway unavailable"))
    manager.set_gateway_message_injector(object(), injector)

    assert (
        context.inject_message(
            "wake up",
            session_key="agent:main:telegram:dm:42",
        )
        is False
    )


def test_system_event_passes_typed_marker_and_terminal_receipt(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path, monkeypatch, {"allow_gateway_injection": True}
    )
    context, manager = _context()
    terminal = concurrent.futures.Future()
    terminal.set_result({"status": "completed", "receipt_id": "receipt"})
    injector = MagicMock(return_value=terminal)
    manager.set_gateway_message_injector(object(), injector)
    route = dict(_EXPECTED_ROUTE)

    receipt = context.inject_gateway_system_event(
        "trusted event body\n",
        session_key="agent:main:telegram:dm:42",
        expected_session_id="session-42",
        event_id="event-42",
        event_kind="external_tool_completed",
        expected_route=route,
        eligibility_check=_eligible,
    )
    route["topic_id"] = "mutated-after-call"

    assert receipt is terminal
    assert receipt.result() == {"status": "completed", "receipt_id": "receipt"}
    kwargs = injector.call_args.kwargs
    assert kwargs["content"] == "trusted event body\n"
    assert kwargs["system_event"].plugin_id == "notify-plugin"
    assert kwargs["system_event"].expected_session_id == "session-42"
    assert kwargs["system_event"].expected_route.topic_id == ""
    assert kwargs["system_event"].eligibility_check is _eligible
    assert isinstance(kwargs["receipt"], concurrent.futures.Future)


def test_system_event_snapshots_route_and_rejects_ineligible_request(
    tmp_path, monkeypatch
):
    _write_plugin_config(
        tmp_path, monkeypatch, {"allow_gateway_injection": True}
    )
    context, manager = _context()
    injector = MagicMock()
    manager.set_gateway_message_injector(object(), injector)
    route = dict(_EXPECTED_ROUTE)

    receipt = context.inject_gateway_system_event(
        "body",
        session_key="agent:main:telegram:dm:42",
        expected_session_id="session-42",
        event_id="event-ineligible",
        event_kind="external_tool_completed",
        expected_route=route,
        eligibility_check=lambda: False,
    )
    route["user_id"] = "mutated"

    assert receipt.result()["status"] == "unauthorized"
    injector.assert_not_called()


def test_system_event_fails_closed_when_eligibility_raises_or_is_awaitable(
    tmp_path, monkeypatch
):
    _write_plugin_config(
        tmp_path, monkeypatch, {"allow_gateway_injection": True}
    )
    context, manager = _context()
    injector = MagicMock()
    manager.set_gateway_message_injector(object(), injector)

    def raises():
        raise RuntimeError("binding unavailable")

    async def async_check():
        return True

    for event_id, eligibility_check in (
        ("event-raises", raises),
        ("event-awaitable", async_check),
    ):
        receipt = context.inject_gateway_system_event(
            "body",
            session_key="agent:main:telegram:dm:42",
            expected_session_id="session-42",
            event_id=event_id,
            event_kind="external_tool_completed",
            expected_route=_EXPECTED_ROUTE,
            eligibility_check=eligibility_check,
        )
        assert receipt.result()["status"] == "unauthorized"

    injector.assert_not_called()


def test_system_event_rejects_invalid_kind_without_host_call(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path, monkeypatch, {"allow_gateway_injection": True}
    )
    context, manager = _context()
    injector = MagicMock()
    manager.set_gateway_message_injector(object(), injector)

    assert context.inject_gateway_system_event(
        "body",
        session_key="agent:main:telegram:dm:42",
        expected_session_id="session-42",
        event_id="event-42",
        event_kind="arbitrary",
        expected_route=_EXPECTED_ROUTE,
        eligibility_check=_eligible,
    ) is None
    injector.assert_not_called()


def test_system_event_requires_exact_complete_route_tuple(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path, monkeypatch, {"allow_gateway_injection": True}
    )
    context, manager = _context()
    injector = MagicMock()
    manager.set_gateway_message_injector(object(), injector)

    for route in (
        {key: value for key, value in _EXPECTED_ROUTE.items() if key != "topic_id"},
        {**_EXPECTED_ROUTE, "label": "caller-controlled"},
    ):
        assert context.inject_gateway_system_event(
            "body",
            session_key="agent:main:telegram:dm:42",
            expected_session_id="session-42",
            event_id="event-42",
            event_kind="external_tool_completed",
            expected_route=route,
            eligibility_check=_eligible,
        ) is None

    injector.assert_not_called()


def test_system_event_reports_authorization_and_liveness_refusals(
    tmp_path, monkeypatch
):
    _write_plugin_config(tmp_path, monkeypatch, {})
    context, manager = _context()
    denied = context.inject_gateway_system_event(
        "body",
        session_key="agent:main:telegram:dm:42",
        expected_session_id="session-42",
        event_id="denied",
        event_kind="external_tool_error",
        expected_route=_EXPECTED_ROUTE,
        eligibility_check=_eligible,
    )
    assert denied.result()["status"] == "unauthorized"

    _write_plugin_config(
        tmp_path, monkeypatch, {"allow_gateway_injection": True}
    )
    unavailable = context.inject_gateway_system_event(
        "body",
        session_key="agent:main:telegram:dm:42",
        expected_session_id="session-42",
        event_id="unavailable",
        event_kind="external_tool_pending_decision",
        expected_route=_EXPECTED_ROUTE,
        eligibility_check=_eligible,
    )
    assert unavailable.result()["status"] == "stopping"


def test_gateway_task_factory_runs_on_host_loop_and_cancels_on_stop():
    async def exercise():
        context, manager = _context()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def worker():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        handle = context.register_gateway_task(worker, name="continuation-worker")
        manager._start_gateway_tasks(asyncio.get_running_loop())
        await asyncio.wait_for(started.wait(), timeout=1)
        manager._stop_gateway_tasks()
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        handle.dispose()
        assert manager._gateway_task_factories == {}

    asyncio.run(exercise())



_EXPECTED_ROUTE = {"profile_name":"default", "platform":"telegram", "user_id":"42", "chat_id":"42", "topic_id":""}

def _eligible():
    return True


def test_discovered_plugin_owns_observer_readiness_and_cleans_up(tmp_path, monkeypatch):
    home = tmp_path / "home"
    plugin = home / "plugins" / "observer-fixture"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: observer-fixture\nversion: 1.0.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "import asyncio\n"
        "def register(ctx):\n"
        "    async def observe():\n"
        "        ctx.set_continuation_observer_ready(True, surface='gateway')\n"
        "        try:\n"
        "            await asyncio.Event().wait()\n"
        "        finally:\n"
        "            ctx.set_continuation_observer_ready(False, surface='gateway')\n"
        "    ctx.register_gateway_task(observe, name='external-completion')\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [observer-fixture]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    loaded = manager._plugins["observer-fixture"]
    assert loaded.error is None

    async def exercise():
        manager._start_gateway_tasks(asyncio.get_running_loop())
        await asyncio.sleep(0)
        tasks = tuple(manager._gateway_tasks.values())
        assert len(tasks) == 1 and not tasks[0].done()
        manager._stop_gateway_tasks()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert tasks[0].cancelled()
        assert not manager._gateway_tasks
    asyncio.run(exercise())


def test_gateway_grant_stays_in_plugin_owning_profile(tmp_path, monkeypatch):
    owner = tmp_path / "owner"
    unrelated = tmp_path / "unrelated"
    owner.mkdir()
    unrelated.mkdir()
    (owner / "config.yaml").write_text(
        "plugins:\n  entries:\n    notify-plugin:\n      allow_gateway_injection: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(owner))
    context, _ = _context()
    monkeypatch.setenv("HERMES_HOME", str(unrelated))
    assert context._gateway_injection_allowed() is True
    (owner / "config.yaml").write_text("plugins: {}\n", encoding="utf-8")
    assert context._gateway_injection_allowed() is False
