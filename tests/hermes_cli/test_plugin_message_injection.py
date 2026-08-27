"""Tests for plugin message injection across CLI and gateway hosts."""

from unittest.mock import MagicMock, patch

import yaml

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


def test_plugin_context_rejects_invalid_mode_before_host_dispatch():
    context, manager = _context()
    cli = MagicMock()
    setattr(manager, "_cli_ref", cli)

    assert context.inject_message("wake up", mode="bogus") is False
    cli.inject_message.assert_not_called()


def test_cli_idle_injection_delegates_to_public_host_seam():
    context, manager = _context()
    cli = MagicMock()
    cli.inject_message.return_value = True
    manager._cli_ref = cli

    assert context.inject_message("new input") is True
    cli.inject_message.assert_called_once_with(
        "new input", role="user", mode="queue", target_session=None
    )


def test_cli_injection_forwards_role_and_mode_to_public_host_seam():
    context, manager = _context()
    cli = MagicMock()
    cli.inject_message.return_value = True
    manager._cli_ref = cli

    assert context.inject_message("status", "system", mode="interrupt") is True
    cli.inject_message.assert_called_once_with(
        "status", role="system", mode="interrupt", target_session=None
    )


def test_cli_injection_unwraps_exact_surface_target():
    """Walkie host targets are ``surface:host-owned-token`` (ADR-0002)."""
    context, manager = _context()
    cli = MagicMock()
    cli.inject_message.return_value = True
    manager._cli_ref = cli

    assert context.inject_message("wake up", target_session="cli:session-42") is True
    cli.inject_message.assert_called_once_with(
        "wake up", role="user", mode="queue", target_session="session-42"
    )


def test_cli_host_rejects_foreign_surface_targets():
    context, manager = _context()
    cli = MagicMock()
    setattr(manager, "_cli_ref", cli)

    for target in ("tui:session-42", "gateway:session-42"):
        assert context.inject_message("wake up", target_session=target) is False

    cli.inject_message.assert_not_called()


def test_surface_target_selects_only_its_router():
    context, _manager = _context()
    tui_router = MagicMock(return_value=True)
    gateway_router = MagicMock(return_value=True)

    with patch.dict(
        PluginContext.inject_message.__globals__["_INJECTION_ROUTERS"],
        {"tui": tui_router, "gateway": gateway_router},
        clear=True,
    ):
        assert (
            context.inject_message(
                "wake up",
                target_session="gateway:session-42",
            )
            is True
        )

    tui_router.assert_not_called()
    gateway_router.assert_called_once_with(
        "wake up",
        role="user",
        mode="queue",
        target_session="session-42",
        plugin_id="notify-plugin",
    )


def test_surface_target_skips_irrelevant_manager_injector():
    """A gateway fallback must not shadow an explicitly addressed TUI router."""
    context, manager = _context()
    gateway_injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), gateway_injector)
    tui_router = MagicMock(return_value=True)

    with patch.dict(
        PluginContext.inject_message.__globals__["_INJECTION_ROUTERS"],
        {"tui": tui_router},
        clear=True,
    ):
        assert (
            context.inject_message(
                "wake up",
                target_session="tui:session-42",
            )
            is True
        )

    gateway_injector.assert_not_called()
    tui_router.assert_called_once_with(
        "wake up",
        role="user",
        mode="queue",
        target_session="session-42",
        plugin_id="notify-plugin",
    )


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


def test_gateway_manager_injector_is_queue_only(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)

    for mode in ("interrupt", "steer"):
        assert (
            context.inject_message(
                "wake up",
                mode=mode,
                session_key="agent:main:telegram:dm:42",
            )
            is False
        )

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


def test_gateway_permission_uses_owning_manager_profile(tmp_path, monkeypatch):
    active_home = tmp_path / "active"
    isolated_home = tmp_path / "isolated"
    active_home.mkdir()
    isolated_home.mkdir()
    (active_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {
                    "entries": {
                        "notify-plugin": {"allow_gateway_injection": True},
                    },
                },
            }
        )
    )
    (isolated_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {
                    "entries": {
                        "notify-plugin": {"allow_gateway_injection": False},
                    },
                },
            }
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(active_home))

    manager = PluginManager(scope_key=str(isolated_home))
    context = PluginContext(
        PluginManifest(name="notify-plugin", key="notify-plugin", source="user"),
        manager,
    )
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


def test_gateway_injection_unwraps_exact_surface_target(tmp_path, monkeypatch):
    """The gateway receives its raw route token, not the Walkie surface wrapper."""
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)

    assert (
        context.inject_message(
            "wake up",
            target_session="gateway:session-42",
        )
        is True
    )
    injector.assert_called_once_with(
        session_key="session-42",
        content="wake up",
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
