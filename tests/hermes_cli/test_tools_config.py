"""Tests for hermes_cli.tools_config platform tool persistence."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.nous_account import NousPortalAccountInfo, NousToolAccessInfo
from hermes_cli.nous_subscription import NousSubscriptionFeatures
from hermes_cli.tools_config import (
    _DEFAULT_OFF_TOOLSETS,
    _RECENTLY_SHIPPED_TOOLSETS,
    _apply_toolset_change,
    _checklist_toolset_keys,
    _configure_provider,
    _reconfigure_provider,
    _get_platform_tools,
    _platform_toolset_summary,
    _reconfigure_tool,
    _run_post_setup,
    _save_platform_tools,
    _toolset_has_keys,
    _toolset_needs_configuration_prompt,
    CONFIGURABLE_TOOLSETS,
    TOOL_CATEGORIES,
    gui_toolset_label,
    _visible_providers,
    provider_readiness_status,
    tools_command,
)




def test_all_invalid_platform_toolsets_logs_runtime_warning(caplog):
    """#38798: an explicit platform config whose toolset names are all invalid
    (e.g. 'hermes' instead of 'hermes-cli') must warn at resolve time so an
    already-corrupted config is caught at runtime, not just during migration."""
    import hermes_cli.tools_config as _tc
    # The runtime warning fires once per platform per process; clear the guard
    # so this test is deterministic regardless of prior resolutions.
    _tc._warned_invalid_platform_toolsets.discard("cli")
    config = {"platform_toolsets": {"cli": ["hermes"]}}

    with caplog.at_level(logging.WARNING, logger="hermes_cli.tools_config"):
        _get_platform_tools(config, "cli")

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("#38798" in m and "hermes" in m for m in warnings), warnings


def test_valid_platform_toolsets_no_runtime_warning(caplog):
    """A correctly-configured platform must not emit the #38798 warning."""
    config = {"platform_toolsets": {"cli": ["hermes-cli"]}}

    with caplog.at_level(logging.WARNING, logger="hermes_cli.tools_config"):
        _get_platform_tools(config, "cli")

    assert not any("#38798" in r.getMessage() for r in caplog.records)


def test_partially_valid_platform_toolsets_no_runtime_warning(caplog):
    """When at least one configured toolset is valid, tools still resolve, so
    the runtime zero-tools warning must not fire (the migration-time check still
    flags the individual bad name)."""
    config = {"platform_toolsets": {"cli": ["hermes-cli", "bogus"]}}

    with caplog.at_level(logging.WARNING, logger="hermes_cli.tools_config"):
        _get_platform_tools(config, "cli")

    assert not any("#38798" in r.getMessage() for r in caplog.records)










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


def test_get_platform_tools_homeassistant_uses_active_profile_token(monkeypatch):
    from agent import secret_scope

    monkeypatch.delenv("HASS_TOKEN", raising=False)
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"HASS_TOKEN": "profile-token"})
    try:
        assert "homeassistant" in _get_platform_tools({}, "cron")
        assert "homeassistant" in _get_platform_tools({}, "cli")
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


# ─── #35527: platform-restricted default-off toolsets (discord/discord_admin)
# are stripped by _DEFAULT_OFF_TOOLSETS even when the user explicitly opts in
# via the platform's native composite. The composite ``hermes-discord``
# contains both ``discord`` and ``discord_admin`` tools, so configuring it is
# an explicit opt-in that should survive the default-off strip. ───────────────


def test_discord_toolsets_do_not_leak_to_other_platforms():
    """Layer 4 (guard): discord/discord_admin are platform-restricted — they
    must never appear on a non-discord platform even when that platform is
    explicitly configured."""
    config = {"platform_toolsets": {"telegram": ["hermes-telegram", "discord"]}}
    enabled = _get_platform_tools(config, "telegram")
    assert "discord" not in enabled
    assert "discord_admin" not in enabled








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






















def test_first_install_nous_auto_configures_video_gen(monkeypatch):
    """When a Nous subscriber checks video_gen in the toolset checklist,
    apply_nous_managed_defaults must write video_gen.provider and
    video_gen.use_gateway so the FAL plugin can route through the gateway
    at runtime.  Regression test for the bug where video_gen was marked as
    auto-configured but no config was actually written."""
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
        lambda *args, **kwargs: {"video_gen"},
    )
    monkeypatch.setattr("hermes_cli.tools_config.save_config", lambda config: None)
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_enabled_platforms",
        lambda: ["cli"],
    )
    monkeypatch.setattr(
        "hermes_cli.nous_subscription.get_nous_portal_account_info",
        lambda *args, **kwargs: NousPortalAccountInfo(
            logged_in=True,
            source="jwt",
            fresh=False,
            paid_service_access=True,
        ),
    )

    configured = []
    monkeypatch.setattr(
        "hermes_cli.tools_config._configure_toolset",
        lambda ts_key, config: configured.append(ts_key),
    )

    tools_command(first_install=True, config=config)

    assert config["video_gen"]["provider"] == "fal"
    assert config["video_gen"]["use_gateway"] is True
    # video_gen should NOT appear in the manual configure list — it's auto-configured
    assert "video_gen" not in configured

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


    def test_picker_repairs_corrupt_config_section(self):
        """When image_gen is a non-dict (user-edit YAML), the picker should
        replace it with a fresh dict rather than crash."""
        from hermes_cli.tools_config import _configure_imagegen_model
        config = {"image_gen": "some-garbage-string"}
        with patch("hermes_cli.tools_config._prompt_choice", return_value=0):
            _configure_imagegen_model("fal", config)
        assert isinstance(config["image_gen"], dict)
        assert config["image_gen"]["model"] == "fal-ai/flux-2/klein/9b"






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






# ---------------------------------------------------------------------------
# Inline Nous Portal login gate on managed-provider selection
# ---------------------------------------------------------------------------










# ── Checklist diff scope: non-configurable toolsets (kanban) must not be
#    reported as added/removed by `hermes tools` ──────────────────────────




def test_kanban_not_reported_as_removed_in_diff():
    """Reproduces the false-signal bug: `hermes tools` printed ``- kanban``
    when saving a platform that resolves kanban as enabled, even though the
    checklist never offered kanban as a toggle.

    The printed diff must be scoped to ``_checklist_toolset_keys`` so a tool
    the user could not deselect is never reported as removed. The persisted
    config still keeps kanban (verified separately by _save_platform_tools).
    """
    config = {"platform_toolsets": {"telegram": ["kanban", "web", "terminal"]}}
    current = _get_platform_tools(config, "telegram", include_default_mcp_servers=False)
    assert "kanban" in current  # resolved as enabled at read time

    # The checklist can only return configurable keys it was shown; kanban
    # is never one of them.
    universe = _checklist_toolset_keys("telegram")
    new_enabled = {t for t in current if t != "kanban"}

    # Unscoped (old, buggy) diff would surface kanban.
    assert (current - new_enabled) == {"kanban"}
    # Scoped (fixed) diff drops it.
    assert ((current - new_enabled) & universe) == set()






def test_vision_picker_custom_endpoint(tmp_path, monkeypatch):
    """Custom endpoint writes base_url+model to config and the key to env."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.tools_config as tc
    from hermes_cli.config import load_config

    seq = iter([2])  # Custom OpenAI-compatible endpoint
    prompts = iter(["https://my.endpoint/v1", "sk-secret", "my-vision-model"])
    with patch.object(tc, "_prompt_choice", side_effect=lambda *a, **k: next(seq)), \
         patch.object(tc, "_prompt", side_effect=lambda *a, **k: next(prompts)), \
         patch.object(tc, "save_env_value") as save_env, \
         patch.object(tc, "_toolset_has_keys", return_value=False):
        tc._configure_vision_backend()

    v = load_config().get("auxiliary", {}).get("vision", {})
    assert v.get("base_url") == "https://my.endpoint/v1"
    assert v.get("model") == "my-vision-model"
    # provider pinned to "custom" so the resolver routes through base_url.
    assert v.get("provider") == "custom"
    save_env.assert_called_once_with("OPENAI_API_KEY", "sk-secret")


def test_vision_picker_filters_text_only_models():
    """#78884: the vision picker must not offer models the models.dev
    catalog marks as text-only. Known-capable (supports_vision=True) and
    uncatalogued (None) models stay; the custom-id option stays last."""
    import hermes_cli.tools_config as tc

    rows = [
        {
            "slug": "nous",
            "name": "Nous",
            "models": [
                "openai/gpt-5.6-sol-pro",      # supports_vision=True  → kept
                "deepseek/deepseek-v4-flash",  # supports_vision=False → filtered
                "vendor/uncatalogued",         # None (fail open)      → kept
            ],
        }
    ]

    def _caps(provider, model):
        if model == "openai/gpt-5.6-sol-pro":
            return SimpleNamespace(supports_vision=True)
        if model == "deepseek/deepseek-v4-flash":
            return SimpleNamespace(supports_vision=False)
        return None

    picked = []

    def _pick(question, choices, default=0):
        picked.append(choices)
        return 0  # first provider, then first (vision-capable) model

    config = {"auxiliary": {"vision": {}}}
    with (
        patch("hermes_cli.inventory.build_aux_picker_rows", return_value=rows),
        patch("agent.models_dev.get_model_capabilities", side_effect=_caps),
        patch.object(tc, "_prompt_choice", side_effect=_pick),
        patch.object(tc, "save_config"),
    ):
        tc._configure_vision_provider_model(config, config["auxiliary"]["vision"])

    assert len(picked) == 2  # provider prompt, then model prompt
    assert picked[1] == [
        "openai/gpt-5.6-sol-pro",
        "vendor/uncatalogued",
        "Type a custom model id…",
    ]
    assert config["auxiliary"]["vision"]["provider"] == "nous"
    assert config["auxiliary"]["vision"]["model"] == "openai/gpt-5.6-sol-pro"


def test_vision_picker_warns_when_all_curated_models_text_only():
    """#78884: if every curated model for the chosen provider is known
    text-only, warn loudly but keep the custom-id escape hatch."""
    import hermes_cli.tools_config as tc

    rows = [
        {
            "slug": "deepseek",
            "name": "DeepSeek",
            "models": ["deepseek/deepseek-v4-flash", "deepseek/deepseek-v3"],
        }
    ]

    def _caps(provider, model):
        return SimpleNamespace(supports_vision=False)

    picked = []

    def _pick(question, choices, default=0):
        picked.append(choices)
        return 0  # first provider; custom-id row is the only model row left

    warnings = []

    def _warn(msg):
        warnings.append(msg)

    config = {"auxiliary": {"vision": {}}}
    with (
        patch("hermes_cli.inventory.build_aux_picker_rows", return_value=rows),
        patch("agent.models_dev.get_model_capabilities", side_effect=_caps),
        patch.object(tc, "_prompt_choice", side_effect=_pick),
        patch.object(tc, "_prompt", return_value="my/custom-vision-model"),
        patch.object(tc, "_print_warning", side_effect=_warn),
        patch.object(tc, "save_config"),
    ):
        tc._configure_vision_provider_model(config, config["auxiliary"]["vision"])

    assert picked[1] == ["Type a custom model id…"]
    assert warnings, "expected a warning when every curated model is text-only"
    assert config["auxiliary"]["vision"]["provider"] == "deepseek"
    assert config["auxiliary"]["vision"]["model"] == "my/custom-vision-model"




# ─── provider_readiness_status ────────────────────────────────────────────────
#
# Server-side truth for the GUI "Ready" pill (issue: Capabilities tab showed
# Ready for every zero-env-var provider row, including logged-out Nous
# Subscription rows and never-installed KittenTTS/Piper).


def _fake_features(*, logged_in: bool, paid: bool = True):
    account = (
        NousPortalAccountInfo(
            logged_in=True, source="jwt", fresh=False, paid_service_access=paid
        )
        if logged_in
        else NousPortalAccountInfo(
            logged_in=False, source="none", fresh=False, paid_service_access=None
        )
    )
    return SimpleNamespace(nous_auth_present=logged_in, account_info=account)


def test_visible_providers_reuses_logged_out_feature_snapshot(monkeypatch):
    import hermes_cli.tools_config as tools_config

    account = NousPortalAccountInfo(
        logged_in=False,
        source="none",
        fresh=False,
        paid_service_access=None,
    )
    features = NousSubscriptionFeatures(
        subscribed=False,
        nous_auth_present=False,
        provider_is_nous=False,
        features={},
        account_info=account,
    )
    monkeypatch.setattr(
        tools_config,
        "get_nous_subscription_features",
        lambda *args, **kwargs: pytest.fail("feature snapshot was resolved again"),
    )

    providers = _visible_providers(
        TOOL_CATEGORIES["image_gen"], {}, features=features
    )

    assert any(
        provider.get("managed_nous_feature") == "image_gen"
        for provider in providers
    )


def test_visible_providers_reuses_pool_video_feature_snapshot(monkeypatch):
    import hermes_cli.tools_config as tools_config

    account = NousPortalAccountInfo(
        logged_in=True,
        source="jwt",
        fresh=False,
        paid_service_access=False,
        tool_access=NousToolAccessInfo(
            enabled=True,
            coverage={"fal-video": False},
        ),
    )
    features = NousSubscriptionFeatures(
        subscribed=True,
        nous_auth_present=True,
        provider_is_nous=False,
        features={},
        account_info=account,
    )
    monkeypatch.setattr(
        tools_config,
        "get_nous_subscription_features",
        lambda *args, **kwargs: pytest.fail("feature snapshot was resolved again"),
    )

    providers = _visible_providers(
        TOOL_CATEGORIES["video_gen"], {}, features=features
    )

    assert not any(
        provider.get("managed_nous_feature") == "video_gen"
        for provider in providers
    )




# ── Windows console-flash guard for post-setup subprocess spawns ──────────────
#
# The desktop GUI runs post-setup hooks through a detached, console-less
# `hermes tools post-setup <key>` child. On Windows each console child (npm,
# npx, pip, powershell) spawned without CREATE_NO_WINDOW materializes a brand
# new console window — the "terminal flash" reported on the Capabilities
# browser-setup journey. `_post_setup_no_window_flags` is the single wrapper
# every hook spawn passes as `creationflags`.






# ── Post-setup readiness predicates for the browser rows ─────────────────────
#
# The GUI's "Run setup" idempotence rides on provider_readiness_status
# reporting ready/needs_setup honestly. agent_browser (local browser) must
# track the FULL local install (CLI + Chromium), the cloud-provider hook
# ("browserbase") only the CLI, and camofox its npm package.


# ── Toolsets that shipped after a platform's last `hermes tools` save ────────
#
# Saving the picker (or one toggle in the desktop Toolsets UI) replaces a
# platform's composite (``[hermes-cli]``) with a frozen explicit list, and
# nothing ever adds to that list — so a toolset shipped later stays off
# forever, while everyone still on the composite inherits it on upgrade.
# ``_RECENTLY_SHIPPED_TOOLSETS`` closes that gap for toolsets new enough that
# absence from a saved list cannot mean the user declined them.
#
# Every assertion here is a subset test against that set, which passes
# vacuously once it empties out — and empty is the steady state between
# releases. Skip loudly rather than going quietly green.
_requires_recently_shipped = pytest.mark.skipif(
    not _RECENTLY_SHIPPED_TOOLSETS,
    reason="no toolset is currently inside its first release",
)


def _saved_list_from_before(platform="cli"):
    """A saved explicit list as it looked before the new toolsets existed."""
    from hermes_cli.tools_config import (
        _CONFIG_ONLY_TOOLSETS,
        _toolset_allowed_for_platform,
    )

    return {
        "platform_toolsets": {
            platform: sorted(
                ts_key
                for ts_key, _, _ in CONFIGURABLE_TOOLSETS
                if ts_key not in _RECENTLY_SHIPPED_TOOLSETS
                and ts_key not in _DEFAULT_OFF_TOOLSETS
                and ts_key not in _CONFIG_ONLY_TOOLSETS
                and _toolset_allowed_for_platform(ts_key, platform)
            )
        }
    }


@_requires_recently_shipped
def test_saved_list_gains_toolsets_that_shipped_after_it_was_written():
    """The bug: a frozen list never gained bfl, so composite users got Nous
    Portal video generation on upgrade and picker users silently did not."""
    on_composite = _get_platform_tools(
        {"platform_toolsets": {"cli": ["hermes-cli"]}},
        "cli",
        include_default_mcp_servers=False,
    )
    on_saved_list = _get_platform_tools(
        _saved_list_from_before(), "cli", include_default_mcp_servers=False
    )

    assert _RECENTLY_SHIPPED_TOOLSETS <= (on_composite & on_saved_list)


@_requires_recently_shipped
def test_unchecking_the_new_toolset_sticks():
    """Saving records it as offered, so the next read reads absence as a
    decline instead of turning it back on."""
    config = {"platform_toolsets": {"cli": ["hermes-cli"]}}
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", enabled - _RECENTLY_SHIPPED_TOOLSETS)

    reread = _get_platform_tools(config, "cli", include_default_mcp_servers=False)

    assert not (_RECENTLY_SHIPPED_TOOLSETS & reread)


@_requires_recently_shipped
def test_agent_disabled_toolsets_still_wins():
    """The other way to say no — a global suppression list applied last."""
    config = _saved_list_from_before()
    config["agent"] = {"disabled_toolsets": sorted(_RECENTLY_SHIPPED_TOOLSETS)}

    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)

    assert not (_RECENTLY_SHIPPED_TOOLSETS & enabled)


@_requires_recently_shipped
def test_platforms_whose_composite_excludes_it_are_left_narrow():
    """Parity is the justification, so don't widen a deliberately small
    composite (hermes-acp, hermes-webhook) that never carried the toolset."""
    from toolsets import TOOLSETS, resolve_toolset

    narrow = [
        platform
        for platform in ("acp", "webhook")
        if f"hermes-{platform}" in TOOLSETS
        and not any(
            set(resolve_toolset(ts, include_registry=False))
            <= set(resolve_toolset(f"hermes-{platform}"))
            for ts in _RECENTLY_SHIPPED_TOOLSETS
        )
    ]
    assert narrow, "expected a composite that excludes the new toolset"

    for platform in narrow:
        enabled = _get_platform_tools(
            _saved_list_from_before(platform),
            platform,
            include_default_mcp_servers=False,
        )
        assert not (_RECENTLY_SHIPPED_TOOLSETS & enabled), platform
