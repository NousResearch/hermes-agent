"""Tests for hermes_cli.tools_config platform tool persistence."""

from unittest.mock import patch

from hermes_cli.tools_config import (
    _configure_provider,
    _get_platform_tools,
    _platform_toolset_summary,
    _save_platform_tools,
    _toolset_has_keys,
    TOOL_CATEGORIES,
    _visible_providers,
    tools_command,
)


def test_get_platform_tools_uses_default_when_platform_not_configured():
    config = {}

    enabled = _get_platform_tools(config, "cli")

    assert enabled


def test_get_platform_tools_preserves_explicit_empty_selection():
    config = {"platform_toolsets": {"cli": []}}

    enabled = _get_platform_tools(config, "cli")

    assert enabled == set()


def test_platform_toolset_summary_uses_explicit_platform_list():
    config = {}

    summary = _platform_toolset_summary(config, platforms=["cli"])

    assert set(summary.keys()) == {"cli"}
    assert summary["cli"] == _get_platform_tools(config, "cli")


def test_get_platform_tools_includes_enabled_mcp_servers_by_default():
    config = {
        "mcp_servers": {
            "exa": {"url": "https://mcp.exa.ai/mcp"},
            "web-search-prime": {"url": "https://api.z.ai/api/mcp/web_search_prime/mcp"},
            "disabled-server": {"url": "https://example.com/mcp", "enabled": False},
        }
    }

    enabled = _get_platform_tools(config, "cli")

    assert "exa" in enabled
    assert "web-search-prime" in enabled
    assert "disabled-server" not in enabled


def test_get_platform_tools_keeps_enabled_mcp_servers_with_explicit_builtin_selection():
    config = {
        "platform_toolsets": {"cli": ["web", "memory"]},
        "mcp_servers": {
            "exa": {"url": "https://mcp.exa.ai/mcp"},
            "web-search-prime": {"url": "https://api.z.ai/api/mcp/web_search_prime/mcp"},
        },
    }

    enabled = _get_platform_tools(config, "cli")

    assert "web" in enabled
    assert "memory" in enabled
    assert "exa" in enabled
    assert "web-search-prime" in enabled


def test_get_platform_tools_no_mcp_sentinel_excludes_all_mcp_servers():
    """The 'no_mcp' sentinel in platform_toolsets excludes all MCP servers."""
    config = {
        "platform_toolsets": {"cli": ["web", "terminal", "no_mcp"]},
        "mcp_servers": {
            "exa": {"url": "https://mcp.exa.ai/mcp"},
            "web-search-prime": {"url": "https://api.z.ai/api/mcp/web_search_prime/mcp"},
        },
    }

    enabled = _get_platform_tools(config, "cli")

    assert "web" in enabled
    assert "terminal" in enabled
    assert "exa" not in enabled
    assert "web-search-prime" not in enabled
    assert "no_mcp" not in enabled


def test_get_platform_tools_no_mcp_sentinel_does_not_affect_other_platforms():
    """The 'no_mcp' sentinel only affects the platform it's configured on."""
    config = {
        "platform_toolsets": {
            "api_server": ["web", "terminal", "no_mcp"],
        },
        "mcp_servers": {
            "exa": {"url": "https://mcp.exa.ai/mcp"},
        },
    }

    # api_server should exclude MCP
    api_enabled = _get_platform_tools(config, "api_server")
    assert "exa" not in api_enabled

    # cli (not configured with no_mcp) should include MCP
    cli_enabled = _get_platform_tools(config, "cli")
    assert "exa" in cli_enabled


def test_toolset_has_keys_for_vision_accepts_codex_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(
        '{"active_provider":"openai-codex","providers":{"openai-codex":{"tokens":{"access_token": "codex-...oken","refresh_token": "codex-...oken"}}}}'
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AUXILIARY_VISION_PROVIDER", raising=False)
    monkeypatch.delenv("CONTEXT_VISION_PROVIDER", raising=False)
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_vision_provider_client",
        lambda: ("openai-codex", object(), "gpt-4.1"),
    )

    assert _toolset_has_keys("vision") is True


def test_save_platform_tools_preserves_mcp_server_names():
    """Ensure MCP server names are preserved when saving platform tools.

    Regression test for https://github.com/NousResearch/hermes-agent/issues/1247
    """
    config = {
        "platform_toolsets": {
            "cli": ["web", "terminal", "time", "github", "custom-mcp-server"]
        }
    }

    new_selection = {"web", "browser"}

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", new_selection)

    saved_toolsets = config["platform_toolsets"]["cli"]

    assert "time" in saved_toolsets
    assert "github" in saved_toolsets
    assert "custom-mcp-server" in saved_toolsets
    assert "web" in saved_toolsets
    assert "browser" in saved_toolsets
    assert "terminal" not in saved_toolsets


def test_save_platform_tools_handles_empty_existing_config():
    """Saving platform tools works when no existing config exists."""
    config = {}

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "telegram", {"web", "terminal"})

    saved_toolsets = config["platform_toolsets"]["telegram"]
    assert "web" in saved_toolsets
    assert "terminal" in saved_toolsets


def test_save_platform_tools_handles_invalid_existing_config():
    """Saving platform tools works when existing config is not a list."""
    config = {
        "platform_toolsets": {
            "cli": "invalid-string-value"
        }
    }

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", {"web"})

    saved_toolsets = config["platform_toolsets"]["cli"]
    assert "web" in saved_toolsets


def test_save_platform_tools_does_not_preserve_platform_default_toolsets():
    """Platform default toolsets (hermes-cli, hermes-telegram, etc.) must NOT
    be preserved across saves.

    These "super" toolsets resolve to ALL tools, so if they survive in the
    config, they silently override any tools the user unchecked. Previously,
    the preserve filter only excluded configurable toolset keys (web, browser,
    terminal, etc.) and treated platform defaults as unknown custom entries
    (like MCP server names), causing them to be kept unconditionally.

    Regression test: user unchecks image_gen and homeassistant via
    ``hermes tools``, but hermes-cli stays in the config and re-enables
    everything on the next read.
    """
    config = {
        "platform_toolsets": {
            "cli": [
                "browser", "clarify", "code_execution", "cronjob",
                "delegation", "file", "hermes-cli",  # <-- the culprit
                "memory", "session_search", "skills", "terminal",
                "todo", "tts", "vision", "web",
            ]
        }
    }

    # User unchecks image_gen, homeassistant, moa — keeps the rest
    new_selection = {
        "browser", "clarify", "code_execution", "cronjob",
        "delegation", "file", "memory", "session_search",
        "skills", "terminal", "todo", "tts", "vision", "web",
    }

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", new_selection)

    saved = config["platform_toolsets"]["cli"]

    # hermes-cli must NOT survive — it's a platform default, not an MCP server
    assert "hermes-cli" not in saved

    # The individual toolset keys the user selected must be present
    assert "web" in saved
    assert "terminal" in saved
    assert "browser" in saved

    # Tools the user unchecked must NOT be present
    assert "image_gen" not in saved
    assert "homeassistant" not in saved
    assert "moa" not in saved


def test_save_platform_tools_does_not_preserve_hermes_telegram():
    """Same bug for Telegram — hermes-telegram must not be preserved."""
    config = {
        "platform_toolsets": {
            "telegram": [
                "browser", "file", "hermes-telegram", "terminal", "web",
            ]
        }
    }

    new_selection = {"browser", "file", "terminal", "web"}

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "telegram", new_selection)

    saved = config["platform_toolsets"]["telegram"]
    assert "hermes-telegram" not in saved
    assert "web" in saved


def test_save_platform_tools_still_preserves_mcp_with_platform_default_present():
    """MCP server names must still be preserved even when platform defaults
    are being stripped out."""
    config = {
        "platform_toolsets": {
            "cli": [
                "web", "terminal", "hermes-cli", "my-mcp-server", "github-tools",
            ]
        }
    }

    new_selection = {"web", "browser"}

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", new_selection)

    saved = config["platform_toolsets"]["cli"]

    # MCP servers preserved
    assert "my-mcp-server" in saved
    assert "github-tools" in saved

    # Platform default stripped
    assert "hermes-cli" not in saved

    # User selections present
    assert "web" in saved
    assert "browser" in saved

    # Deselected configurable toolset removed
    assert "terminal" not in saved


def test_visible_providers_include_nous_subscription_when_logged_in(monkeypatch):
    monkeypatch.setenv("HERMES_ENABLE_NOUS_MANAGED_TOOLS", "1")
    config = {"model": {"provider": "nous"}}

    monkeypatch.setattr(
        "hermes_cli.nous_subscription.get_nous_auth_status",
        lambda: {"logged_in": True},
    )

    providers = _visible_providers(TOOL_CATEGORIES["browser"], config)

    assert providers[0]["name"].startswith("Nous Subscription")


def test_visible_providers_hide_nous_subscription_when_feature_flag_is_off(monkeypatch):
    monkeypatch.delenv("HERMES_ENABLE_NOUS_MANAGED_TOOLS", raising=False)
    config = {"model": {"provider": "nous"}}

    monkeypatch.setattr(
        "hermes_cli.nous_subscription.get_nous_auth_status",
        lambda: {"logged_in": True},
    )

    providers = _visible_providers(TOOL_CATEGORIES["browser"], config)

    assert all(not provider["name"].startswith("Nous Subscription") for provider in providers)


def test_local_browser_provider_is_saved_explicitly(monkeypatch):
    config = {}
    local_provider = next(
        provider
        for provider in TOOL_CATEGORIES["browser"]["providers"]
        if provider.get("browser_provider") == "local"
    )
    monkeypatch.setattr("hermes_cli.tools_config._run_post_setup", lambda key: None)

    _configure_provider(local_provider, config)

    assert config["browser"]["cloud_provider"] == "local"


def test_first_install_nous_auto_configures_managed_defaults(monkeypatch):
    monkeypatch.setenv("HERMES_ENABLE_NOUS_MANAGED_TOOLS", "1")
    config = {
        "model": {"provider": "nous"},
        "platform_toolsets": {"cli": []},
    }
    for env_var in (
        "VOICE_TOOLS_OPENAI_KEY",
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "TAVILY_API_KEY",
        "PARALLEL_API_KEY",
        "BROWSERBASE_API_KEY",
        "BROWSERBASE_PROJECT_ID",
        "BROWSER_USE_API_KEY",
        "FAL_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setattr(
        "hermes_cli.tools_config._prompt_toolset_checklist",
        lambda *args, **kwargs: {"web", "image_gen", "tts", "browser"},
    )
    monkeypatch.setattr("hermes_cli.tools_config.save_config", lambda config: None)
    # Prevent leaked platform tokens (e.g. DISCORD_BOT_TOKEN from gateway.run
    # import) from adding extra platforms. The loop in tools_command runs
    # apply_nous_managed_defaults per platform; a second iteration sees values
    # set by the first as "explicit" and skips them.
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_enabled_platforms",
        lambda: ["cli"],
    )
    monkeypatch.setattr(
        "hermes_cli.nous_subscription.get_nous_auth_status",
        lambda: {"logged_in": True},
    )

    configured = []
    monkeypatch.setattr(
        "hermes_cli.tools_config._configure_toolset",
        lambda ts_key, config: configured.append(ts_key),
    )

    tools_command(first_install=True, config=config)

    assert config["web"]["backend"] == "firecrawl"
    assert config["tts"]["provider"] == "openai"
    assert config["browser"]["cloud_provider"] == "browser-use"
    assert configured == []

# ── Platform / toolset consistency ────────────────────────────────────────────


class TestPlatformToolsetConsistency:
    """Every platform in tools_config.PLATFORMS must have a matching toolset."""

    def test_all_platforms_have_toolset_definitions(self):
        """Each platform's default_toolset must exist in TOOLSETS."""
        from hermes_cli.tools_config import PLATFORMS
        from toolsets import TOOLSETS

        for platform, meta in PLATFORMS.items():
            ts_name = meta["default_toolset"]
            assert ts_name in TOOLSETS, (
                f"Platform {platform!r} references toolset {ts_name!r} "
                f"which is not defined in toolsets.py"
            )

    def test_gateway_toolset_includes_all_messaging_platforms(self):
        """hermes-gateway includes list should cover all messaging platforms."""
        from hermes_cli.tools_config import PLATFORMS
        from toolsets import TOOLSETS

        gateway_includes = set(TOOLSETS["hermes-gateway"]["includes"])
        # Exclude non-messaging platforms from the check
        non_messaging = {"cli", "api_server"}
        for platform, meta in PLATFORMS.items():
            if platform in non_messaging:
                continue
            ts_name = meta["default_toolset"]
            assert ts_name in gateway_includes, (
                f"Platform {platform!r} toolset {ts_name!r} missing from "
                f"hermes-gateway includes"
            )

    def test_skills_config_covers_tools_config_platforms(self):
        """skills_config.PLATFORMS should have entries for all gateway platforms."""
        from hermes_cli.tools_config import PLATFORMS as TOOLS_PLATFORMS
        from hermes_cli.skills_config import PLATFORMS as SKILLS_PLATFORMS

        non_messaging = {"api_server"}
        for platform in TOOLS_PLATFORMS:
            if platform in non_messaging:
                continue
            assert platform in SKILLS_PLATFORMS, (
                f"Platform {platform!r} in tools_config but missing from "
                f"skills_config PLATFORMS"
            )


def test_numeric_mcp_server_name_does_not_crash_sorted():
    """YAML parses bare numeric keys (e.g. ``12306:``) as int.

    _get_platform_tools must normalise them to str so that sorted()
    on the returned set never raises TypeError on mixed int/str.

    Regression test for https://github.com/NousResearch/hermes-agent/issues/6901
    """
    config = {
        "platform_toolsets": {"cli": ["web", 12306]},
        "mcp_servers": {
            12306: {"url": "https://example.com/mcp"},
            "normal-server": {"url": "https://example.com/mcp2"},
        },
    }

    enabled = _get_platform_tools(config, "cli")

    # All names must be str — no int leaking through
    assert all(isinstance(name, str) for name in enabled), (
        f"Non-string toolset names found: {enabled}"
    )
    assert "12306" in enabled

    # sorted() must not raise TypeError
    sorted(enabled)
