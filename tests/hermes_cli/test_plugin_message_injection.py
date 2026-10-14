"""Tests for plugin message injection across CLI and gateway hosts."""

from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml

import hermes_cli.plugins as plugins_mod
from agent import secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
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


def test_published_gateway_host_reaches_existing_and_future_profile_managers(tmp_path, monkeypatch):
    homes = [tmp_path / name for name in ("launch", "secondary", "late")]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text(
            yaml.safe_dump({
                "plugins": {"entries": {"notify-plugin": {"allow_gateway_injection": True}}}
            })
        )
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    plugins_mod._reset_plugin_managers_for_tests()
    # Multiplex gateway: launch home + secondary profiles served by one process.
    was_active = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)

    def manager_for(home):
        token = set_hermes_home_override(str(home))
        try:
            return plugins_mod.get_plugin_manager()
        finally:
            reset_hermes_home_override(token)

    owner = object()
    injector = MagicMock(return_value=True)
    try:
        launch_manager = manager_for(homes[0])
        secondary_manager = manager_for(homes[1])
        assert secondary_manager is not launch_manager
        assert secondary_manager.has_gateway_message_injector is False

        plugins_mod.publish_gateway_message_host(owner, injector)
        assert launch_manager.has_gateway_message_injector is True
        assert secondary_manager.has_gateway_message_injector is True

        late_manager = manager_for(homes[2])
        assert late_manager.has_gateway_message_injector is True
        # A -> B -> A: returning to the launch home resolves the same, still-stamped manager.
        assert manager_for(homes[0]) is launch_manager
        assert launch_manager.has_gateway_message_injector is True

        token = set_hermes_home_override(str(homes[1]))
        try:
            context = PluginContext(
                PluginManifest(name="notify-plugin", key="notify-plugin", source="user"),
                secondary_manager,
            )
            assert context.inject_message(
                "continue", session_key="agent:secondary:telegram:dm:42"
            ) is True
        finally:
            reset_hermes_home_override(token)

        injector.assert_called_once_with(
            session_key="agent:secondary:telegram:dm:42",
            content="continue",
            plugin_id="notify-plugin",
        )

        plugins_mod.clear_published_gateway_message_host(owner)
        assert launch_manager.has_gateway_message_injector is False
        assert secondary_manager.has_gateway_message_injector is False
        assert late_manager.has_gateway_message_injector is False
    finally:
        plugins_mod.clear_published_gateway_message_host(owner)
        plugins_mod._reset_plugin_managers_for_tests()
        secret_scope.set_multiplex_active(was_active)


ORIGIN = {"platform": "discord", "chat_id": "42", "chat_type": "thread", "thread_id": "42", "user_id": "u1"}


def test_origin_injection_targets_the_plugins_own_home(tmp_path, monkeypatch):
    """``origin`` is forwarded with the manager's immutable home — never an ambient one — so the
    gateway can only start the session in the profile that loaded the plugin."""
    _write_plugin_config(tmp_path, monkeypatch, {"allow_gateway_injection": True})
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)
    other = tmp_path / "other"
    other.mkdir()
    token = set_hermes_home_override(other)
    try:
        assert context.inject_message("Kick off", origin=ORIGIN) is True
    finally:
        reset_hermes_home_override(token)
    injector.assert_called_once_with(
        origin=ORIGIN, plugin_home=manager.home_path, content="Kick off", plugin_id="notify-plugin",
    )


def test_origin_injection_permission_comes_from_the_plugins_own_profile(tmp_path, monkeypatch):
    """The allow_gateway_injection gate reads the plugin's profile config even when the calling
    thread's ambient home is another profile that grants it."""
    _write_plugin_config(tmp_path, monkeypatch, {"allow_gateway_injection": False})
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)
    granting = tmp_path / "granting"
    granting.mkdir()
    (granting / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"entries": {"notify-plugin": {"allow_gateway_injection": True}}}}),
        encoding="utf-8",
    )
    token = set_hermes_home_override(granting)
    try:
        assert context.inject_message("x", origin=ORIGIN) is False
    finally:
        reset_hermes_home_override(token)
    injector.assert_not_called()


def test_origin_injection_rejects_ambiguous_or_malformed_targets(tmp_path, monkeypatch):
    _write_plugin_config(tmp_path, monkeypatch, {"allow_gateway_injection": True})
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)

    assert context.inject_message("x", session_key="agent:main:discord:thread:42", origin=ORIGIN) is False
    assert context.inject_message("x", origin="discord:42") is False
    injector.assert_not_called()


def test_origin_injection_never_lands_in_the_attached_cli():
    """An origin names a messaging chat; a CLI-hosted plugin gets False, not a REPL queue entry."""
    context, manager = _context()
    cli = SimpleNamespace(_agent_running=False, _pending_input=SimpleQueue(), _interrupt_queue=SimpleQueue())
    manager._cli_ref = cli

    assert context.inject_message("x", origin=ORIGIN) is False
    assert cli._pending_input.empty() and cli._interrupt_queue.empty()
