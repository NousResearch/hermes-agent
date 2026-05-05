"""Regression tests for scoped MCP discovery at CLI startup."""

import sys


def test_chat_explicit_toolsets_do_not_start_unrequested_mcp(monkeypatch):
    """`hermes chat -t ...` must not discover PostHog before chat filtering."""
    import hermes_cli.main as main_mod
    import tools.mcp_tool as mcp_mod

    cfg = {
        "mcp_servers": {
            "posthog": {"command": "posthog-mcp", "enabled": True},
            "notion": {"command": "notion-mcp", "enabled": True},
        }
    }
    captured = {}

    def fake_helper(toolsets, config, include_default=False):
        captured["toolsets"] = list(toolsets)
        captured["config"] = config
        captured["include_default"] = include_default
        return []

    def fake_discover(*, server_names=None):
        captured["server_names"] = server_names
        return []

    chat_calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "chat",
            "-Q",
            "--source",
            "eod-cron",
            "-t",
            "file,terminal,skills,zoho,notion",
            "-q",
            "eod prompt",
        ],
    )
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config",
        lambda config, accept_hooks=False: None,
    )
    monkeypatch.setattr(mcp_mod, "mcp_server_names_from_toolsets", fake_helper)
    monkeypatch.setattr(mcp_mod, "discover_mcp_tools", fake_discover)
    monkeypatch.setattr(main_mod, "cmd_chat", lambda args: chat_calls.append(args))

    main_mod.main()

    assert captured["toolsets"] == ["file", "terminal", "skills", "zoho", "notion"]
    assert captured["config"] is cfg
    assert captured["include_default"] is False
    assert captured["server_names"] == []
    assert chat_calls
    assert chat_calls[0].toolsets == "file,terminal,skills,zoho,notion"


def test_chat_explicit_mcp_toolset_allows_only_that_server(monkeypatch):
    """Explicit MCP server names remain an opt-in for startup discovery."""
    import hermes_cli.main as main_mod
    import tools.mcp_tool as mcp_mod

    cfg = {
        "mcp_servers": {
            "posthog": {"command": "posthog-mcp", "enabled": True},
            "product-analytics": {"command": "analytics-mcp", "enabled": True},
        }
    }
    captured = {}

    def fake_discover(*, server_names=None):
        captured["server_names"] = server_names
        return []

    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "chat", "-Q", "-t", "file,posthog", "-q", "analytics task"],
    )
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config",
        lambda config, accept_hooks=False: None,
    )
    monkeypatch.setattr(mcp_mod, "discover_mcp_tools", fake_discover)
    monkeypatch.setattr(main_mod, "cmd_chat", lambda args: None)

    main_mod.main()

    assert captured["server_names"] == ["posthog"]
