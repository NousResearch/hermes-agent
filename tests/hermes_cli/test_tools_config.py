"""Tests for hermes_cli.tools_config platform tool persistence."""

from unittest.mock import patch

import pytest

from hermes_cli.tools_config import (
    _DEFAULT_OFF_TOOLSETS,
    _apply_toolset_change,
    _configure_provider,
    _reconfigure_provider,
    _get_platform_tools,
    _platform_toolset_summary,
    _reconfigure_tool,
    _save_platform_tools,
    _toolset_has_keys,
    CONFIGURABLE_TOOLSETS,
    TOOL_CATEGORIES,
    _visible_providers,
    tools_command,
)


def test_agent_disabled_toolsets_suppresses_across_platforms():
    """agent.disabled_toolsets in config.yaml should remove those toolsets
    from the enabled set, regardless of platform defaults or explicit config.
    """
    config = {
        "agent": {"disabled_toolsets": ["memory"]},
    }

    cli_enabled = _get_platform_tools(config, "cli")
    discord_enabled = _get_platform_tools(config, "discord")

    assert "memory" not in cli_enabled
    assert "memory" not in discord_enabled


def test_agent_disabled_toolsets_with_explicit_platform_config():
    """agent.disabled_toolsets should still suppress even when the platform
    has an explicit toolset list that includes the disabled toolset.
    """
    config = {
        "agent": {"disabled_toolsets": ["memory"]},
        "platform_toolsets": {"cli": ["web", "terminal", "memory"]},
    }

    enabled = _get_platform_tools(config, "cli")

    assert "memory" not in enabled
    assert "web" in enabled
    assert "terminal" in enabled


def test_agent_disabled_toolsets_empty_list_is_noop():
    """Empty or missing disabled_toolsets should not change behavior."""
    config_empty = {"agent": {"disabled_toolsets": []}}
    config_none = {"agent": {}}
    config_missing = {}

    default = _get_platform_tools({}, "cli")

    assert _get_platform_tools(config_empty, "cli") == default
    assert _get_platform_tools(config_none, "cli") == default
    assert _get_platform_tools(config_missing, "cli") == default


def test_get_platform_tools_uses_default_when_platform_not_configured():
    config = {}

    enabled = _get_platform_tools(config, "cli")

    assert enabled
    assert enabled.isdisjoint(_DEFAULT_OFF_TOOLSETS)


def test_configurable_toolsets_include_messaging():
    assert any(ts_key == "messaging" for ts_key, _, _ in CONFIGURABLE_TOOLSETS)

def test_get_platform_tools_default_telegram_includes_messaging():
    enabled = _get_platform_tools({}, "telegram")

    assert "messaging" in enabled


def test_get_platform_tools_homeassistant_platform_keeps_homeassistant_toolset():
    enabled = _get_platform_tools({}, "homeassistant")

    assert "homeassistant" in enabled


def test_get_platform_tools_homeassistant_toolset_enabled_for_cron_when_hass_token_set(monkeypatch):
    """HA toolset is runtime-gated by check_fn (requires HASS_TOKEN).

    When HASS_TOKEN is set, the user has explicitly opted in — _DEFAULT_OFF_TOOLSETS
    shouldn't also strip HA from platforms (like cron) that run through
    _get_platform_tools without an explicit saved toolset list.

    Regression guard for Norbert's HA cron breakage after #14798 made cron
    honor per-platform tool config.
    """
    monkeypatch.setenv("HASS_TOKEN", "fake-test-token")

    cron_enabled = _get_platform_tools({}, "cron")
    assert "homeassistant" in cron_enabled
    # moa must stay off — the original goal of #14798
    assert "moa" not in cron_enabled

    cli_enabled = _get_platform_tools({}, "cli")
    assert "homeassistant" in cli_enabled


def test_get_platform_tools_homeassistant_toolset_off_for_cron_when_hass_token_missing(monkeypatch):
    """Without HASS_TOKEN, HA stays off by default — preserves #14798's behavior
    for users who never configured HA."""
    monkeypatch.delenv("HASS_TOKEN", raising=False)

    cron_enabled = _get_platform_tools({}, "cron")
    assert "homeassistant" not in cron_enabled


def test_get_platform_tools_expands_composite_when_mixed_with_configurable():
    """``[hermes-cli, spotify]`` (composite + configurable) must keep the full
    ``hermes-cli`` toolset alongside the explicit Spotify opt-in. The
    has_explicit_config branch used to drop ``hermes-cli`` on the floor,
    leaving sessions with only ``{spotify, kanban}``."""
    config = {"platform_toolsets": {"cli": ["hermes-cli", "spotify"]}}

    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)

    # Native tools must reappear.
    for ts in ("terminal", "file", "web", "browser", "memory", "delegation",
               "code_execution", "todo", "session_search", "skills"):
        assert ts in enabled, f"{ts} should be enabled when hermes-cli is listed"
    # User explicitly opted into Spotify — must survive _DEFAULT_OFF_TOOLSETS subtraction.
    assert "spotify" in enabled


def test_get_platform_tools_composite_only_unchanged():
    """Composite-only config (no configurable in list) must still take the
    else-branch path and produce the full toolset — guards against the new
    code accidentally hijacking the composite-only case."""
    composite_only = _get_platform_tools(
        {"platform_toolsets": {"cli": ["hermes-cli"]}},
        "cli",
        include_default_mcp_servers=False,
    )
    default = _get_platform_tools({}, "cli", include_default_mcp_servers=False)

    assert composite_only == default


def test_get_platform_tools_configurable_only_no_expansion():
    """Configurable-only list (no composite) must not pull in unrelated
    toolsets — guards against the expansion firing when ``composite_tools``
    is empty."""
    config = {"platform_toolsets": {"cli": ["terminal", "file"]}}

    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)

    assert "terminal" in enabled
    assert "file" in enabled
    # Web shouldn't sneak in via the new expansion path.
    assert "web" not in enabled


def test_get_platform_tools_mixed_does_not_resurrect_default_off():
    """Expansion must subtract _DEFAULT_OFF_TOOLSETS from the implicit
    pull-in. Without this, ``hermes-cli`` expansion would re-enable
    ``moa`` / ``rl`` / ``homeassistant`` for users who never opted in."""
    config = {"platform_toolsets": {"cli": ["hermes-cli", "terminal"]}}

    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)

    assert "terminal" in enabled
    assert "moa" not in enabled
    assert "rl" not in enabled


def test_get_platform_tools_preserves_explicit_empty_selection():
    config = {"platform_toolsets": {"cli": []}}

    enabled = _get_platform_tools(config, "cli")

    # An explicit empty list disables every CONFIGURABLE toolset (web,
    # terminal, memory, …). Non-configurable platform toolsets that ride
    # along on the platform's default composite (e.g. `kanban`, whose tools
    # live in _HERMES_CORE_TOOLS but aren't user-toggleable) are still
    # auto-recovered by _get_platform_tools so saving via `hermes tools`
    # doesn't silently drop them. The contract this test guards is the
    # configurable side: nothing the user could have checked in the TUI
    # checklist should reappear here.
    configurable = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}
    assert enabled.isdisjoint(configurable)


def test_apply_toolset_change_from_default_does_not_enable_default_off_toolsets():
    """Disabling one default toolset on a fresh config must not persist
    default-off toolsets as explicitly enabled.
    """
    config = {}

    with patch("hermes_cli.tools_config.save_config"):
        _apply_toolset_change(config, "cli", ["memory"], "disable")

    saved = set(config["platform_toolsets"]["cli"])
    assert "memory" not in saved
    assert "terminal" in saved
    assert saved.isdisjoint(_DEFAULT_OFF_TOOLSETS)


def test_apply_toolset_change_can_enable_default_off_toolset_from_default():
    config = {}

    with patch("hermes_cli.tools_config.save_config"):
        _apply_toolset_change(config, "cli", ["homeassistant"], "enable")

    saved = set(config["platform_toolsets"]["cli"])
    assert "homeassistant" in saved
    assert "terminal" in saved


def test_get_platform_tools_handles_null_platform_toolsets():
    """YAML `platform_toolsets:` with no value parses as None — the old
    ``config.get("platform_toolsets", {})`` pattern would then crash with
    ``NoneType has no attribute 'get'`` on the next line. Guard against that.
    """
    config = {"platform_toolsets": None}

    enabled = _get_platform_tools(config, "cli")

    # Falls through to defaults instead of raising
    assert enabled


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
    monkeypatch.setattr("hermes_cli.tools_config.managed_nous_tools_enabled", lambda: True)
    config = {"model": {"provider": "nous"}}

    monkeypatch.setattr(
        "hermes_cli.nous_subscription.get_nous_auth_status",
        lambda: {"logged_in": True},
    )

    providers = _visible_providers(TOOL_CATEGORIES["browser"], config)

    assert providers[0]["name"].startswith("Nous Subscription")


def test_visible_providers_hide_nous_subscription_when_feature_flag_is_off(monkeypatch):
    monkeypatch.setattr("hermes_cli.tools_config.managed_nous_tools_enabled", lambda: False)
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


def test_reconfigure_lists_enabled_web_without_existing_provider_config(monkeypatch):
    config = {"platform_toolsets": {"cli": ["web"]}}
    seen = {}
    configured = []

    monkeypatch.setattr(
        "hermes_cli.tools_config._toolset_has_keys",
        lambda ts_key, config=None: False,
    )

    def fake_prompt_choice(question, choices, default=0):
        seen["choices"] = choices
        return 0

    monkeypatch.setattr("hermes_cli.tools_config._prompt_choice", fake_prompt_choice)
    monkeypatch.setattr(
        "hermes_cli.tools_config._configure_tool_category_for_reconfig",
        lambda ts_key, cat, config: configured.append(ts_key),
    )
    monkeypatch.setattr("hermes_cli.tools_config.save_config", lambda config: None)

    _reconfigure_tool(config)

    assert any("Web Search" in choice for choice in seen["choices"])
    assert configured == ["web"]


def test_first_install_nous_auto_configures_managed_defaults(monkeypatch):
    monkeypatch.setattr("hermes_cli.tools_config.managed_nous_tools_enabled", lambda: True)
    monkeypatch.setattr("hermes_cli.nous_subscription.managed_nous_tools_enabled", lambda: True)
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
        non_messaging = {"cli", "api_server", "cron"}
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


# ─── Imagegen Backend Picker Wiring ────────────────────────────────────────

class TestImagegenBackendRegistry:
    """IMAGEGEN_BACKENDS tags drive the model picker flow in tools_config."""

    def test_fal_backend_registered(self):
        from hermes_cli.tools_config import IMAGEGEN_BACKENDS
        assert "fal" in IMAGEGEN_BACKENDS

    def test_fal_catalog_loads_lazily(self):
        """catalog_fn should defer import to avoid import cycles."""
        from hermes_cli.tools_config import IMAGEGEN_BACKENDS
        catalog, default = IMAGEGEN_BACKENDS["fal"]["catalog_fn"]()
        assert default == "fal-ai/flux-2/klein/9b"
        assert "fal-ai/flux-2/klein/9b" in catalog
        assert "fal-ai/flux-2-pro" in catalog

    def test_image_gen_providers_tagged_with_fal_backend(self):
        """Both Nous Subscription and FAL.ai providers must carry the
        imagegen_backend tag so _configure_provider fires the picker."""
        from hermes_cli.tools_config import TOOL_CATEGORIES
        providers = TOOL_CATEGORIES["image_gen"]["providers"]
        for p in providers:
            assert p.get("imagegen_backend") == "fal", (
                f"{p['name']} missing imagegen_backend tag"
            )


class TestImagegenModelPicker:
    """_configure_imagegen_model writes selection to config and respects
    curses fallback semantics (returns default when stdin isn't a TTY)."""

    def test_picker_writes_chosen_model_to_config(self):
        from hermes_cli.tools_config import _configure_imagegen_model
        config = {}
        # Force _prompt_choice to pick index 1 (second-in-ordered-list).
        with patch("hermes_cli.tools_config._prompt_choice", return_value=1):
            _configure_imagegen_model("fal", config)
        # ordered[0] == current (default klein), ordered[1] == first non-default
        assert config["image_gen"]["model"] != "fal-ai/flux-2/klein/9b"
        assert config["image_gen"]["model"].startswith("fal-ai/")

    def test_picker_with_gpt_image_does_not_prompt_quality(self):
        """GPT-Image quality is pinned to medium in the tool's defaults —
        no follow-up prompt, no config write for quality_setting."""
        from hermes_cli.tools_config import (
            _configure_imagegen_model,
            IMAGEGEN_BACKENDS,
        )
        catalog, default_model = IMAGEGEN_BACKENDS["fal"]["catalog_fn"]()
        model_ids = list(catalog.keys())
        ordered = [default_model] + [m for m in model_ids if m != default_model]
        gpt_idx = ordered.index("fal-ai/gpt-image-1.5")

        # Only ONE picker call is expected (for model) — not two (model + quality).
        call_count = {"n": 0}
        def fake_prompt(*a, **kw):
            call_count["n"] += 1
            return gpt_idx

        config = {}
        with patch("hermes_cli.tools_config._prompt_choice", side_effect=fake_prompt):
            _configure_imagegen_model("fal", config)

        assert call_count["n"] == 1, (
            f"Expected 1 picker call (model only), got {call_count['n']}"
        )
        assert config["image_gen"]["model"] == "fal-ai/gpt-image-1.5"
        assert "quality_setting" not in config["image_gen"]

    def test_picker_no_op_for_unknown_backend(self):
        from hermes_cli.tools_config import _configure_imagegen_model
        config = {}
        _configure_imagegen_model("nonexistent-backend", config)
        assert config == {}  # untouched

    def test_picker_repairs_corrupt_config_section(self):
        """When image_gen is a non-dict (user-edit YAML), the picker should
        replace it with a fresh dict rather than crash."""
        from hermes_cli.tools_config import _configure_imagegen_model
        config = {"image_gen": "some-garbage-string"}
        with patch("hermes_cli.tools_config._prompt_choice", return_value=0):
            _configure_imagegen_model("fal", config)
        assert isinstance(config["image_gen"], dict)
        assert config["image_gen"]["model"] == "fal-ai/flux-2/klein/9b"


def test_save_platform_tools_normalizes_numeric_entries():
    """YAML may parse bare numeric toolset names as int. They should be
    normalized to str so they survive the save round-trip.
    """
    config = {
        "platform_toolsets": {
            "cli": ["web", "terminal", 12306, "custom-mcp"]
        }
    }

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", {"web", "browser"})

    saved = config["platform_toolsets"]["cli"]
    assert "12306" in saved
    assert 12306 not in saved


def test_save_platform_tools_clears_no_mcp_sentinel():
    """`hermes tools` has no UI for no_mcp, so saving from the picker clears
    the sentinel unconditionally — otherwise a user who once set no_mcp by
    hand could never re-enable MCP servers through the UI.
    """
    config = {
        "platform_toolsets": {
            "cli": ["web", "terminal", "no_mcp"]
        }
    }

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", {"web", "browser"})

    saved = config["platform_toolsets"]["cli"]
    assert "no_mcp" not in saved


def test_save_platform_tools_preserves_mcp_server_names():
    """Non-sentinel passthrough entries (MCP server names) must still survive
    the save — we only clear `no_mcp`, not every non-configurable entry.
    """
    config = {
        "platform_toolsets": {
            "cli": ["web", "terminal", "custom-mcp", "another-mcp"]
        }
    }

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", {"web", "browser"})

    saved = config["platform_toolsets"]["cli"]
    assert "custom-mcp" in saved
    assert "another-mcp" in saved


def test_get_platform_tools_recovers_non_configurable_toolsets_from_composite():
    """Non-configurable toolsets whose tools are in the composite but not in
    CONFIGURABLE_TOOLSETS should still appear in the result.
    """
    from toolsets import TOOLSETS
    from hermes_cli.tools_config import PLATFORMS
    from unittest.mock import patch as mock_patch

    fake_toolsets = dict(TOOLSETS)
    fake_toolsets["_test_platform_tool"] = {
        "description": "test",
        "tools": ["_test_special_tool"],
        "includes": [],
    }
    fake_toolsets["hermes-_test_platform"] = {
        "description": "test composite",
        "tools": ["web_search", "web_extract", "terminal", "process", "_test_special_tool"],
        "includes": [],
    }

    test_platforms = {
        "_test_platform": {"label": "Test", "default_toolset": "hermes-_test_platform"},
    }

    with mock_patch("hermes_cli.tools_config.PLATFORMS", {**PLATFORMS, **test_platforms}):
        with mock_patch("toolsets.TOOLSETS", fake_toolsets):
            enabled = _get_platform_tools({}, "_test_platform")

    assert "_test_platform_tool" in enabled
    assert "web" in enabled
    assert "terminal" in enabled


def test_get_platform_tools_second_pass_skips_fully_claimed_toolsets():
    """Toolsets whose tools are fully covered by configurable keys should NOT
    be added by the second pass (prevents 'search', 'hermes-acp' noise).
    """
    enabled = _get_platform_tools({}, "cli")

    assert "search" not in enabled


def test_get_platform_tools_discord_both_off_by_default():
    """Both `discord` and `discord_admin` are opt-in via `hermes tools`,
    even on the Discord platform itself.  Users shouldn't auto-inherit 19
    extra tools just because DISCORD_BOT_TOKEN is set."""
    enabled = _get_platform_tools({}, "discord")
    assert "discord" not in enabled
    assert "discord_admin" not in enabled


def test_discord_toolsets_in_configurable_toolsets():
    keys = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}
    assert "discord" in keys
    assert "discord_admin" in keys


def test_discord_toolsets_in_default_off():
    assert "discord" in _DEFAULT_OFF_TOOLSETS
    assert "discord_admin" in _DEFAULT_OFF_TOOLSETS


def test_discord_toolsets_not_available_on_other_platforms():
    """Platform-scoping: discord / discord_admin should not appear on CLI,
    Telegram, etc. — not even as an opt-in."""
    from hermes_cli.tools_config import _toolset_allowed_for_platform
    for plat in ["cli", "telegram", "slack", "whatsapp", "signal"]:
        assert not _toolset_allowed_for_platform("discord", plat), (
            f"`discord` toolset leaked onto {plat}"
        )
        assert not _toolset_allowed_for_platform("discord_admin", plat), (
            f"`discord_admin` toolset leaked onto {plat}"
        )
    assert _toolset_allowed_for_platform("discord", "discord")
    assert _toolset_allowed_for_platform("discord_admin", "discord")


def test_discord_toolsets_user_enabled_are_honored():
    """When the user opts in via `hermes tools`, the toolset appears."""
    config = {"platform_toolsets": {"discord": ["web", "terminal", "discord"]}}
    enabled = _get_platform_tools(config, "discord")
    assert "discord" in enabled
    assert "discord_admin" not in enabled


def test_save_platform_tools_strips_restricted_toolsets():
    """Hand-edited or all-platforms checklist with `discord` selected for
    Telegram must be stripped at save time."""
    from hermes_cli.tools_config import _save_platform_tools
    config = {}
    _save_platform_tools(config, "telegram", {"web", "terminal", "discord", "discord_admin"})
    saved = config["platform_toolsets"]["telegram"]
    assert "discord" not in saved
    assert "discord_admin" not in saved
    assert "web" in saved
    assert "terminal" in saved


def test_get_platform_tools_feishu_includes_doc_and_drive():
    enabled = _get_platform_tools({}, "feishu")
    assert "feishu_doc" in enabled
    assert "feishu_drive" in enabled


def test_get_platform_tools_feishu_tools_not_on_other_platforms():
    for plat in ["cli", "telegram", "discord"]:
        enabled = _get_platform_tools({}, plat)
        assert "feishu_doc" not in enabled, f"feishu_doc leaked onto {plat}"
        assert "feishu_drive" not in enabled, f"feishu_drive leaked onto {plat}"


def test_get_effective_configurable_toolsets_dedupes_bundled_plugins():
    """Bundled plugins (plugins/spotify) share their toolset key with the
    built-in CONFIGURABLE_TOOLSETS entry. The effective list must not list
    them twice — otherwise `hermes tools` → "reconfigure existing" shows
    the same toolset two rows in a row.
    """
    from hermes_cli.tools_config import _get_effective_configurable_toolsets

    all_ts = _get_effective_configurable_toolsets()
    keys = [ts_key for ts_key, _, _ in all_ts]
    assert len(keys) == len(set(keys)), (
        f"duplicate toolset keys in effective list: "
        f"{[k for k in keys if keys.count(k) > 1]}"
    )
    # Spotify specifically — the bug that motivated the dedupe.
    spotify_rows = [t for t in all_ts if t[0] == "spotify"]
    assert len(spotify_rows) == 1, spotify_rows
    # Built-in label wins over the plugin label.
    assert spotify_rows[0][1] == "🎵 Spotify"


@pytest.mark.parametrize("provider,config_key,expected", [
    # managed provider → use_gateway True
    ({"name": "T", "tts_provider": "elevenlabs", "managed_nous_feature": "tts", "env_vars": []}, "tts", True),
    ({"name": "B", "browser_provider": "browserbase", "managed_nous_feature": "browser", "env_vars": []}, "browser", True),
    ({"name": "W", "web_backend": "tavily", "managed_nous_feature": "web", "env_vars": []}, "web", True),
    # self-hosted provider → use_gateway False
    ({"name": "T", "tts_provider": "elevenlabs", "env_vars": []}, "tts", False),
    ({"name": "B", "browser_provider": "browserbase", "env_vars": []}, "browser", False),
    ({"name": "W", "web_backend": "tavily", "env_vars": []}, "web", False),
])
def test_reconfigure_provider_syncs_use_gateway(provider, config_key, expected):
    config = {}
    _reconfigure_provider(provider, config)
    assert config[config_key]["use_gateway"] is expected


def test_reconfigure_browser_provider_overwrites_stale_use_gateway():
    # Switching from managed (use_gateway=True) to self-hosted must clear the stale flag.
    config = {"browser": {"cloud_provider": "managed-browser", "use_gateway": True}}
    provider = {"name": "Browserbase", "browser_provider": "browserbase", "env_vars": []}
    _reconfigure_provider(provider, config)
    assert config["browser"]["use_gateway"] is False
