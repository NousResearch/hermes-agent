"""Startup must remain idle until a real temporary MCP configuration is enabled."""

import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_cli import mcp_startup


@pytest.fixture(autouse=True)
def _isolated_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_started", set())
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", {})
    yield
    for thread in list(mcp_startup._mcp_discovery_thread.values()):
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("config", [None, "{}", "mcp_servers: {}", "mcp_servers:\n  demo:\n    enabled: false\n    command: unused\n"])
def test_idle_config_is_silent_on_repeated_start(tmp_path, monkeypatch, caplog, config):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if config is not None:
        (tmp_path / "config.yaml").write_text(config, encoding="utf-8")
    discovery = Mock()
    monkeypatch.setattr(mcp_startup, "_discover_mcp_tools_without_interactive_oauth", discovery)
    logger = logging.getLogger(__name__)
    for _ in range(3):
        mcp_startup.start_background_mcp_discovery(logger=logger, thread_name="test-idle")
        mcp_startup.join_mcp_discovery(timeout=5)
    discovery.assert_not_called()
    assert not caplog.records


@pytest.mark.parametrize("initial", ["{}", "mcp_servers:\n  demo:\n    enabled: false\n"])
def test_late_enable_is_first_attempt_not_retry(tmp_path, monkeypatch, caplog, initial):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text(initial, encoding="utf-8")
    discovery = Mock()
    monkeypatch.setattr(mcp_startup, "_discover_mcp_tools_without_interactive_oauth", discovery)
    logger = logging.getLogger(__name__)
    mcp_startup.start_background_mcp_discovery(logger=logger, thread_name="test-late")
    config.write_text("mcp_servers:\n  demo:\n    command: unused\n", encoding="utf-8")
    mcp_startup.start_background_mcp_discovery(logger=logger, thread_name="test-late")
    mcp_startup.join_mcp_discovery(timeout=5)
    discovery.assert_called_once()
    assert not any("retrying" in record.message for record in caplog.records)


@pytest.mark.parametrize("enabled", ["false", "0", "0.0", "'off'", "' NO '"])
def test_disabled_values_use_discovery_semantics(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"mcp_servers:\n  demo:\n    enabled: {enabled}\n    command: unused\n",
        encoding="utf-8",
    )
    assert mcp_startup._has_configured_mcp_servers() is False


@pytest.mark.parametrize("managed_enabled", [True, False])
def test_gate_honors_managed_enabled_override(tmp_path, monkeypatch, managed_enabled):
    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    for directory, enabled in ((home, not managed_enabled), (managed, managed_enabled)):
        (directory / "config.yaml").write_text(
            f"mcp_servers:\n  demo:\n    command: unused\n    enabled: {str(enabled).lower()}\n",
            encoding="utf-8",
        )
    assert mcp_startup._has_configured_mcp_servers() is managed_enabled


def test_failed_discovery_retries_then_stays_connected(tmp_path, monkeypatch, caplog):
    (tmp_path / "config.yaml").write_text(
        "mcp_servers:\n  demo:\n    command: unused\n", encoding="utf-8",
    )
    discovery = Mock(side_effect=[RuntimeError("connection failed"), None])
    monkeypatch.setattr(mcp_startup, "_discover_mcp_tools_without_interactive_oauth", discovery)
    monkeypatch.setattr(mcp_startup, "_any_mcp_connected", lambda: discovery.call_count == 2)
    monkeypatch.setattr(mcp_startup, "_servers_awaiting_connect", list)
    for _ in range(3):
        mcp_startup.start_background_mcp_discovery(
            logger=logging.getLogger(__name__), thread_name="test-retry",
        )
        assert mcp_startup.join_mcp_discovery(timeout=5)
    assert discovery.call_count == 2
    assert sum("retrying discovery thread" in record.message for record in caplog.records) == 1


@pytest.fixture
def portable_plugin(tmp_path, monkeypatch):
    from hermes_cli.agent_plugins import MCP_SCHEMA_V1, PLUGIN_SCHEMA_V1

    plugin = tmp_path / "plugins" / "portable"
    plugin.mkdir(parents=True)
    (plugin / "plugin.json").write_text(json.dumps({
        "$schema": PLUGIN_SCHEMA_V1, "name": "portable.test",
    }), encoding="utf-8")
    (plugin / "mcp.json").write_text(json.dumps({
        "$schema": MCP_SCHEMA_V1,
        "mcpServers": {"worker": {"type": "stdio", "command": "unused"}},
    }), encoding="utf-8")
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    return plugin


@pytest.mark.parametrize("enabled", [True, False])
def test_portable_plugin_only_config(tmp_path, portable_plugin, enabled):
    (tmp_path / "config.yaml").write_text(
        "plugins:\n  enabled: [portable.test]\n"
        + ("" if enabled else "  disabled: [portable.test]\n"), encoding="utf-8",
    )
    assert mcp_startup._has_configured_mcp_servers() is enabled


@pytest.mark.parametrize("native_worker", [
    pytest.param(
        {"enabled": False, "command": "sh", "args": ["-c", "curl https://example.test"]},
        id="rejected-native-entry",
    ),
    pytest.param("malformed", id="non-dict-native-entry"),
])
def test_discarded_native_entry_does_not_shadow_portable(
    tmp_path, portable_plugin, native_worker,
):
    from hermes_cli.mcp_security import validate_mcp_server_entry

    if isinstance(native_worker, dict):
        assert validate_mcp_server_entry("worker", native_worker)
    (tmp_path / "config.yaml").write_text(json.dumps({
        "plugins": {"enabled": ["portable.test"]},
        "mcp_servers": {"worker": native_worker},
    }), encoding="utf-8")

    assert mcp_startup._has_configured_mcp_servers() is True


def test_disabled_native_server_shadows_portable_on_repeated_start(
    tmp_path, portable_plugin, monkeypatch, caplog,
):
    from hermes_cli.agent_plugins import MCP_SCHEMA_V1
    from hermes_constants import hermes_home_key

    (tmp_path / "config.yaml").write_text(
        "plugins:\n  enabled: [portable.test]\n"
        "mcp_servers:\n  worker:\n    enabled: false\n    command: unused\n",
        encoding="utf-8",
    )
    discovery = Mock()
    monkeypatch.setattr(mcp_startup, "_discover_mcp_tools_without_interactive_oauth", discovery)
    monkeypatch.setattr(mcp_startup, "_any_mcp_connected", lambda: False)
    monkeypatch.setattr(mcp_startup, "_servers_awaiting_connect", list)
    logger = logging.getLogger(__name__)

    for _ in range(3):
        mcp_startup.start_background_mcp_discovery(
            logger=logger, thread_name="test-shadowed-portable",
        )
        assert mcp_startup.join_mcp_discovery(timeout=5)

    discovery.assert_not_called()
    assert not any("retrying" in record.message for record in caplog.records)
    assert hermes_home_key() not in mcp_startup._mcp_discovery_started

    # A distinct portable server must still permit the first discovery attempt.
    (portable_plugin / "mcp.json").write_text(json.dumps({
        "$schema": MCP_SCHEMA_V1,
        "mcpServers": {
            "worker": {"type": "stdio", "command": "unused"},
            "other": {"type": "stdio", "command": "unused"},
        },
    }), encoding="utf-8")
    assert mcp_startup._has_configured_mcp_servers() is True
    mcp_startup.start_background_mcp_discovery(
        logger=logger, thread_name="test-unshadowed-portable",
    )
    assert mcp_startup.join_mcp_discovery(timeout=5)
    discovery.assert_called_once()
    assert hermes_home_key() in mcp_startup._mcp_discovery_started
    assert not any("retrying" in record.message for record in caplog.records)


def test_a_b_a_profile_config_and_discovery_are_isolated(tmp_path, monkeypatch, caplog):
    from hermes_constants import (
        get_hermes_home, reset_hermes_home_override, set_hermes_home_override,
    )

    a, b = tmp_path / "a", tmp_path / "b"
    for home, enabled in ((a, False), (b, True)):
        home.mkdir()
        (home / "config.yaml").write_text(
            "mcp_servers:\n  demo:\n    command: unused\n"
            f"    enabled: {str(enabled).lower()}\n", encoding="utf-8",
        )
    seen = []
    monkeypatch.setattr(
        mcp_startup, "_discover_mcp_tools_without_interactive_oauth",
        lambda: seen.append(get_hermes_home()),
    )
    monkeypatch.setattr(mcp_startup, "_any_mcp_connected", lambda: get_hermes_home() in seen)
    monkeypatch.setattr(mcp_startup, "_servers_awaiting_connect", list)

    def start(home):
        token = set_hermes_home_override(home)
        try:
            mcp_startup.start_background_mcp_discovery(
                logger=logging.getLogger(__name__), thread_name="test-profile",
            )
            assert mcp_startup.join_mcp_discovery(timeout=5)
        finally:
            reset_hermes_home_override(token)

    start(a)
    start(b)
    start(a)
    assert seen == [b]
    (a / "config.yaml").write_text(
        "mcp_servers:\n  demo:\n    command: unused\n", encoding="utf-8",
    )
    start(a)
    start(b)
    start(a)
    assert seen == [b, a]
    assert not caplog.records
