"""Shared application tool settings preserve profile isolation and persistence."""
import os
from contextlib import contextmanager

import pytest


@pytest.fixture
def multiplex():
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        yield
    finally:
        set_multiplex_active(previous)


@pytest.mark.parametrize("operation", ["selection", "change", "mcp"])
def test_settings_roundtrip_across_served_profiles(operation, tmp_path, multiplex):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.config_toolsets import (
        apply_mcp_change, apply_toolset_change, save_platform_tools)
    from tui_gateway import server

    homes = [tmp_path / "profiles" / name for name in ("a", "b")]
    for home in homes:
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            'platform_toolsets:\n  cli: [file]\nagent:\n  disabled_toolsets: [terminal, browser]\n'
            'mcp_servers:\n  demo:\n    tools:\n      exclude: []\n',
            encoding="utf-8")
        (home / ".env").write_text("", encoding="utf-8")
    before_env = dict(os.environ)

    @contextmanager
    def bound(home):
        tokens = server._profile_runtime_scope_tokens(str(home), hydrate_secrets=False)
        try:
            yield
        finally:
            server._release_profile_runtime_scope_tokens(tokens)

    with bound(homes[0]):
        cfg = load_config()
        if operation == "selection":
            save_platform_tools(cfg, "cli", {"file", "terminal"})
        elif operation == "change":
            apply_toolset_change(cfg, "cli", ["terminal"], "enable")
        else:
            assert apply_mcp_change(cfg, ["demo:one", "missing:one"], "disable") == {"missing"}
            save_config(cfg)
        saved_a = load_config()
    with bound(homes[1]):
        cfg = load_config()
        assert cfg["platform_toolsets"]["cli"] == ["file"]
        assert cfg["agent"]["disabled_toolsets"] == ["terminal", "browser"]
        assert cfg["mcp_servers"]["demo"]["tools"]["exclude"] == []
        apply_toolset_change(cfg, "cli", ["video"], "enable")
    with bound(homes[0]):
        actual = load_config()
        assert actual["platform_toolsets"] == saved_a["platform_toolsets"]
        assert actual["agent"]["disabled_toolsets"] == saved_a["agent"]["disabled_toolsets"]
        assert actual["mcp_servers"] == saved_a["mcp_servers"]
        if operation == "mcp":
            assert actual["mcp_servers"]["demo"]["tools"]["exclude"] == ["one"]
        else:
            assert "terminal" in actual["platform_toolsets"]["cli"]
            assert actual["agent"]["disabled_toolsets"] == ["browser"]
    assert dict(os.environ) == before_env


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False), (1, True), (0, False), (-1, True),
    (" TRUE ", True), ("yes", True), ("ON", True), ("1", True),
    (" FALSE ", False), ("no", False), ("OFF", False), ("0", False),
    (None, None), ("unknown", None), ("", None), ([], None), (1.5, None),
])
@pytest.mark.parametrize("default", [False, True])
def test_enabled_flag_preserves_legacy_default_semantics(value, expected, default):
    from hermes_cli.config_toolsets import parse_enabled_flag
    assert parse_enabled_flag(value, default=default) is (default if expected is None else expected)
