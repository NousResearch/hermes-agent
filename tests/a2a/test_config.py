"""A2A behavioral settings are loaded from config.yaml-shaped data."""

from copy import deepcopy

from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli.tools_config import _get_platform_tools
from plugins.platforms.a2a.config import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_TOOL_IO,
    A2ASettings,
    apply_yaml_config,
)


def test_behavioral_settings_accept_config_values():
    settings = A2ASettings.from_mapping({
        "max_concurrency": 7,
        "max_sessions": 99,
        "tool_io": "none",
        "host": "0.0.0.0",
        "port": 9200,
    })

    assert settings.max_concurrency == 7
    assert settings.max_sessions == 99
    assert settings.tool_io == "none"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9200


def test_invalid_behavioral_settings_fall_back_safely():
    settings = A2ASettings.from_mapping({
        "max_concurrency": 0,
        "max_tasks": 1,
        "tool_io": "secrets",
        "port": 70000,
    })

    assert settings.max_concurrency == DEFAULT_MAX_CONCURRENCY
    assert settings.max_tasks >= settings.max_concurrency
    assert settings.tool_io == DEFAULT_TOOL_IO
    assert settings.port == 9100


def test_platform_yaml_bridge_seeds_a2a_extras():
    seeded = apply_yaml_config(
        {},
        {
            "enabled": True,
            "host": "127.0.0.1",
            "max_concurrency": 4,
            "tool_io": "full",
        },
    )

    assert seeded == {
        "host": "127.0.0.1",
        "max_concurrency": 4,
        "tool_io": "full",
    }


def test_gateway_loader_applies_a2a_config_through_plugin_registry(
    tmp_path, monkeypatch
):
    """Exercise the real YAML -> plugin hook -> PlatformConfig chain."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "a2a:\n"
        "  enabled: true\n"
        "  host: 0.0.0.0\n"
        "  port: 9200\n"
        "  max_concurrency: 7\n"
        "  max_sessions: 99\n"
        "  tool_io: none\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from gateway.config import Platform, load_gateway_config

    platform_config = load_gateway_config().platforms[Platform("a2a")]

    assert platform_config.enabled is True
    assert platform_config.extra == {
        "host": "0.0.0.0",
        "port": 9200,
        "max_concurrency": 7,
        "max_sessions": 99,
        "tool_io": "none",
    }


def test_default_and_disabled_tools_use_generic_platform_configuration():
    defaults = _get_platform_tools(DEFAULT_CONFIG, "a2a")
    assert {"terminal", "file", "web"}.issubset(defaults)

    configured = deepcopy(DEFAULT_CONFIG)
    configured["platform_toolsets"]["a2a"] = ["terminal", "file", "no_mcp"]
    configured.setdefault("agent", {})["disabled_toolsets"] = ["terminal"]

    resolved = _get_platform_tools(configured, "a2a")
    assert "file" in resolved
    assert "terminal" not in resolved
