"""Tests for plugin message injection across CLI and gateway hosts."""

from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import hermes_yaml as yaml

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _context(name: str = "notify-plugin") -> tuple[PluginContext, PluginManager]:
    manager = PluginManager()
    manifest = PluginManifest(name=name, key=name, source="user")
    return PluginContext(manifest, manager), manager


def _write_plugin_config(tmp_path, monkeypatch, entry: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
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


def _manager_for_home(monkeypatch, home) -> PluginManager:
    from hermes_cli.plugins import get_plugin_manager

    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return get_plugin_manager()


def test_published_gateway_host_reaches_every_profile_manager(tmp_path, monkeypatch):
    from hermes_cli.plugins import (
        clear_published_gateway_message_host,
        publish_gateway_message_host,
    )

    launch = _manager_for_home(monkeypatch, tmp_path / "launch")
    secondary = _manager_for_home(monkeypatch, tmp_path / "profiles" / "ven")
    assert launch is not secondary
    owner = object()
    injector = MagicMock(return_value=True)

    publish_gateway_message_host(owner, injector)

    assert launch.has_gateway_message_injector is True
    assert secondary.has_gateway_message_injector is True
    assert secondary.inject_gateway_message(session_key="agent:ven:telegram:dm:7") is True
    injector.assert_called_once_with(session_key="agent:ven:telegram:dm:7")

    later = _manager_for_home(monkeypatch, tmp_path / "profiles" / "later")
    assert later is not secondary
    assert later.has_gateway_message_injector is True

    clear_published_gateway_message_host(owner)

    assert launch.has_gateway_message_injector is False
    assert secondary.has_gateway_message_injector is False
    assert later.has_gateway_message_injector is False
    after = _manager_for_home(monkeypatch, tmp_path / "profiles" / "after")
    assert after.has_gateway_message_injector is False


def test_published_gateway_host_preserves_newer_owner_and_tui_slot(tmp_path, monkeypatch):
    from hermes_cli.plugins import (
        clear_published_gateway_message_host,
        publish_gateway_message_host,
    )

    manager = _manager_for_home(monkeypatch, tmp_path / "launch")
    tui_owner, tui = object(), MagicMock(return_value=False)
    manager.set_tui_message_injector(tui_owner, tui)
    old_owner, new_owner = object(), object()
    newer = MagicMock(return_value=True)

    publish_gateway_message_host(old_owner, MagicMock(return_value=True))
    manager.set_gateway_message_injector(new_owner, newer)
    clear_published_gateway_message_host(old_owner)

    assert manager.has_gateway_message_injector is True
    assert manager.inject_gateway_message(value="kept") is True
    newer.assert_called_once_with(value="kept")
    assert manager.has_tui_message_injector is True
    assert manager.inject_tui_message(session_key="ses_tui") is False


class _LockWithHook:
    """The published-host lock, running ``hook`` once at the first release: whatever a late attach
    still has to do after that point races the hook."""

    def __init__(self, hook):
        import threading

        self._lock = threading.Lock()
        self._hook = hook

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, *exc):
        self._lock.release()
        hook, self._hook = self._hook, None
        if hook is not None:
            hook()


def test_late_attach_cannot_resurrect_a_cleared_gateway_owner(tmp_path, monkeypatch):
    from hermes_cli import plugins
    from hermes_cli.plugins import clear_published_gateway_message_host, publish_gateway_message_host

    owner = object()
    publish_gateway_message_host(owner, MagicMock(return_value=True))
    manager = _manager_for_home(monkeypatch, tmp_path / "late")
    manager._gateway_message_injector = None
    monkeypatch.setattr(
        plugins, "_published_gateway_host_lock",
        _LockWithHook(lambda: clear_published_gateway_message_host(owner)))

    plugins._attach_published_gateway_host(manager)

    assert plugins._published_gateway_message_injector is None
    assert manager.has_gateway_message_injector is False


def test_late_attach_yields_to_a_newer_gateway_owner(tmp_path, monkeypatch):
    from hermes_cli import plugins
    from hermes_cli.plugins import PluginManager, publish_gateway_message_host

    newer = MagicMock(return_value=True)
    publish_gateway_message_host(object(), MagicMock(return_value=True))
    manager = _manager_for_home(monkeypatch, tmp_path / "late")
    manager._gateway_message_injector = None
    monkeypatch.setattr(
        plugins, "_published_gateway_host_lock",
        _LockWithHook(lambda: publish_gateway_message_host("newer", newer)))

    plugins._attach_published_gateway_host(manager)

    assert manager.inject_gateway_message(value="routed") is True
    newer.assert_called_once_with(value="routed")


def _home_with_injection(tmp_path, name: str, allowed: bool):
    home = tmp_path / name
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"entries": {"notify-plugin": {"allow_gateway_injection": allowed}}}})
    )
    return home


@pytest.mark.parametrize(("launch_allows", "secondary_allows"), [(True, False), (False, True)])
def test_gateway_injection_permission_comes_from_the_plugins_own_profile(
    tmp_path, monkeypatch, launch_allows, secondary_allows
):
    from hermes_cli.plugins import publish_gateway_message_host

    launch = _home_with_injection(tmp_path, "launch", launch_allows)
    secondary = _home_with_injection(tmp_path, "secondary", secondary_allows)
    injector = MagicMock(return_value=True)
    publish_gateway_message_host(object(), injector)
    secondary_manager = _manager_for_home(monkeypatch, secondary)
    context = PluginContext(PluginManifest(name="notify-plugin", key="notify-plugin", source="user"), secondary_manager)
    monkeypatch.setenv("HERMES_HOME", str(launch))

    assert context.inject_message("wake up", session_key="agent:secondary:telegram:dm:7") is secondary_allows
    assert injector.call_count == (1 if secondary_allows else 0)
