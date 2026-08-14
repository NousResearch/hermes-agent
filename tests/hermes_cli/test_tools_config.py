"""Tests for hermes_cli.tools_config platform tool persistence."""

import logging
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.browser_tool import AGENT_BROWSER_NPX_SPEC
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







class TestAgentBrowserPostSetup:
    """_run_post_setup('agent_browser'/'browserbase') — #43564.

    agent-browser is no longer a root package.json dependency (there's no
    local `npm install` step anymore); it resolves at runtime via
    tools.browser_tool._find_agent_browser (PATH -> Homebrew/Hermes-managed
    node -> local .bin -> npx). This class exercises the Chromium-install
    branch of _run_post_setup, which now delegates to that same resolution
    cascade instead of hand-rolling its own node_modules/.bin/agent-browser
    (and Windows .cmd-shim) lookup.
    """

    def test_warns_when_neither_npx_nor_agent_browser_on_path(self):
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as run, patch("hermes_cli.tools_config._print_warning") as warn:
            _run_post_setup("agent_browser")

        run.assert_not_called()
        warn.assert_called_once()
        assert "npx not found" in warn.call_args.args[0]

    def test_browserbase_returns_before_any_chromium_check(self):
        """browserbase hosts its own Chromium; it must never reach the
        agent-browser-only Chromium-install branch. It must still verify
        the agent-browser CLI is runnable (issue #86117)."""
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as run, patch(
            "tools.browser_tool._chromium_installed"
        ) as chromium_check, patch(
            "tools.browser_tool._find_agent_browser",
            return_value="npx agent-browser",
        ), patch(
            "tools.browser_tool.warm_agent_browser_npx_cache",
            return_value=True,
        ), patch(
            "hermes_cli.tools_config._print_success"
        ):
            _run_post_setup("browserbase")

        run.assert_not_called()
        chromium_check.assert_not_called()

    def test_browserbase_npx_warm_cache_fails_raises(self):
        """When agent-browser resolves only to the npx sentinel and the
        npx cache warm fails, post-setup must raise RuntimeError so
        run_post_setup_command returns exit code 1 (issue #86117)."""
        with patch("shutil.which", return_value=None), patch(
            "tools.browser_tool._find_agent_browser",
            return_value="npx agent-browser",
        ), patch(
            "tools.browser_tool.warm_agent_browser_npx_cache",
            return_value=False,
        ), patch(
            "hermes_cli.tools_config._print_info"
        ):
            with pytest.raises(RuntimeError, match="could not be resolved via npx"):
                _run_post_setup("browserbase")

    def test_browserbase_real_binary_verifies_and_succeeds(self):
        """When agent-browser resolves to a real binary, post-setup
        verifies it via agent_browser_runnable and reports success."""
        with patch("shutil.which", return_value=None), patch(
            "tools.browser_tool._find_agent_browser",
            return_value="/usr/local/bin/agent-browser",
        ), patch(
            "hermes_constants.agent_browser_runnable",
            return_value=True,
        ), patch(
            "hermes_cli.tools_config._print_success"
        ) as success:
            _run_post_setup("browserbase")

        assert any("already installed" in c.args[0] for c in success.call_args_list)

    def test_browserbase_real_binary_not_runnable_raises(self):
        """When agent-browser resolves to a binary that fails --version,
        post-setup must raise RuntimeError (issue #86117)."""
        with patch("shutil.which", return_value=None), patch(
            "tools.browser_tool._find_agent_browser",
            return_value="/usr/local/bin/agent-browser",
        ), patch(
            "hermes_constants.agent_browser_runnable",
            return_value=False,
        ), patch(
            "hermes_cli.tools_config._print_success"
        ):
            with pytest.raises(RuntimeError, match="not runnable"):
                _run_post_setup("browserbase")

    def test_browserbase_file_not_found_raises(self):
        """When agent-browser cannot be resolved at all (no binary, no
        npx), browserbase post-setup must raise RuntimeError instead of
        silently reporting success (issue #86117)."""
        with patch("shutil.which", return_value=None), patch(
            "tools.browser_tool._find_agent_browser",
            side_effect=FileNotFoundError("agent-browser CLI not found"),
        ), patch(
            "hermes_cli.tools_config._print_warning"
        ):
            with pytest.raises(RuntimeError, match="agent-browser CLI not found"):
                _run_post_setup("browserbase")

    def test_chromium_already_installed_skips_subprocess(self):
        with patch("shutil.which", return_value="/usr/bin/npx"), patch(
            "tools.browser_tool.node_tool_runnable", return_value=True
        ), patch(
            "subprocess.run"
        ) as run, patch(
            "tools.browser_tool._chromium_installed", return_value=True
        ), patch(
            "hermes_cli.tools_config._print_success"
        ) as success:
            _run_post_setup("agent_browser")

        run.assert_not_called()
        success.assert_called_once()
        assert "already installed" in success.call_args.args[0]

    def test_docker_with_missing_chromium_warns_instead_of_installing(self):
        with patch("shutil.which", return_value="/usr/bin/npx"), patch(
            "tools.browser_tool.node_tool_runnable", return_value=True
        ), patch(
            "subprocess.run"
        ) as run, patch(
            "tools.browser_tool._chromium_installed", return_value=False
        ), patch(
            "tools.browser_tool._running_in_docker", return_value=True
        ), patch(
            "hermes_cli.tools_config._print_warning"
        ) as warn:
            _run_post_setup("agent_browser")

        run.assert_not_called()
        assert any("Docker" in c.args[0] for c in warn.call_args_list)

    def test_find_agent_browser_not_found_warns_before_any_chromium_check(self):
        """_find_agent_browser is resolved up front now (shared with the
        browserbase early-return gate), so a FileNotFoundError here must
        short-circuit before even checking Chromium/Docker status."""
        with patch("shutil.which", return_value="/usr/bin/npx"), patch(
            "subprocess.run"
        ) as run, patch(
            "tools.browser_tool._chromium_installed"
        ) as chromium_check, patch(
            "tools.browser_tool._running_in_docker"
        ) as docker_check, patch(
            "tools.browser_tool._find_agent_browser",
            side_effect=FileNotFoundError("agent-browser CLI not found"),
        ), patch(
            "hermes_cli.tools_config._print_warning"
        ) as warn:
            _run_post_setup("agent_browser")

        run.assert_not_called()
        chromium_check.assert_not_called()
        docker_check.assert_not_called()
        assert any("browser tools require Node.js" in c.args[0] for c in warn.call_args_list)

    def test_installs_chromium_via_npx_when_no_local_binary_resolved(self):
        """When _find_agent_browser falls through to npx, the install command
        must shell out to npx directly (not the unresolved 'npx agent-browser'
        string as a single argv element)."""
        with patch(
            "shutil.which",
            # accepts the `path=` kwarg _resolve_npx_bin's extended-path rung
            # calls shutil.which with, not just the bare-PATH positional form.
            side_effect=lambda name, path=None: "/usr/bin/npx" if name == "npx" else None,
        ), patch(
            "tools.browser_tool.node_tool_runnable", return_value=True
        ), patch("subprocess.run") as run, patch(
            "tools.browser_tool._chromium_installed", return_value=False
        ), patch(
            "tools.browser_tool._running_in_docker", return_value=False
        ), patch(
            "tools.browser_tool._find_agent_browser", return_value="npx agent-browser"
        ), patch(
            "hermes_cli.tools_config._print_success"
        ):
            run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
            _run_post_setup("agent_browser")

        run.assert_called_once()
        assert run.call_args.args[0] == [
            "/usr/bin/npx", "--ignore-scripts", "-y", AGENT_BROWSER_NPX_SPEC, "install", "--with-deps",
        ]

    def test_installs_chromium_via_npx_resolved_only_through_extended_path(self):
        """Hermes-managed-Node-only setups: npx resolves via
        _find_agent_browser's extended-PATH fallback, not a bare PATH lookup.
        The install command must use that same resolved npx, not silently
        hand subprocess.run a None argument from a bare shutil.which('npx')
        re-derivation (#43564 regression — Copilot review, task #9)."""
        hermes_npx = "/home/user/.hermes/node/bin/npx"
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as run, patch(
            "tools.browser_tool._chromium_installed", return_value=False
        ), patch(
            "tools.browser_tool._running_in_docker", return_value=False
        ), patch(
            "tools.browser_tool._find_agent_browser", return_value="npx agent-browser"
        ), patch(
            "tools.browser_tool._resolve_npx_bin", return_value=hermes_npx
        ), patch(
            "hermes_cli.tools_config._print_success"
        ):
            run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
            _run_post_setup("agent_browser")

        run.assert_called_once()
        assert run.call_args.args[0] == [
            hermes_npx, "--ignore-scripts", "-y", AGENT_BROWSER_NPX_SPEC, "install", "--with-deps",
        ]

    def test_warns_instead_of_crashing_when_npx_unresolvable_after_all(self):
        """Defensive: if _resolve_npx_bin somehow returns None even though
        _find_agent_browser resolved "npx agent-browser" (e.g. a race where
        npx disappears between the two calls), warn and return instead of
        building a command with a None argv element."""
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as run, patch(
            "tools.browser_tool._chromium_installed", return_value=False
        ), patch(
            "tools.browser_tool._running_in_docker", return_value=False
        ), patch(
            "tools.browser_tool._find_agent_browser", return_value="npx agent-browser"
        ), patch(
            "tools.browser_tool._resolve_npx_bin", return_value=None
        ), patch(
            "hermes_cli.tools_config._print_warning"
        ) as warn:
            _run_post_setup("agent_browser")  # must not raise

        run.assert_not_called()
        assert any("npx not found" in c.args[0] for c in warn.call_args_list)

    def test_installs_chromium_via_resolved_local_binary_path(self):
        """When _find_agent_browser resolves a concrete executable (global
        install, Homebrew, or the Windows .cmd shim it already knows how to
        pick), that path must be invoked directly — not re-wrapped in npx."""
        with patch("shutil.which", return_value="/usr/bin/npx"), patch(
            "subprocess.run"
        ) as run, patch(
            "tools.browser_tool._chromium_installed", return_value=False
        ), patch(
            "tools.browser_tool._running_in_docker", return_value=False
        ), patch(
            "tools.browser_tool._find_agent_browser",
            return_value="/usr/local/bin/agent-browser",
        ), patch(
            "hermes_cli.tools_config._print_success"
        ):
            run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
            _run_post_setup("agent_browser")

        run.assert_called_once()
        assert run.call_args.args[0] == [
            "/usr/local/bin/agent-browser", "install", "--with-deps",
        ]

    def test_install_success_invalidates_chromium_cache(self):
        import tools.browser_tool as _bt

        with patch("shutil.which", return_value="/usr/bin/npx"), patch(
            "tools.browser_tool.node_tool_runnable", return_value=True
        ), patch(
            "subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ), patch(
            "tools.browser_tool._chromium_installed", return_value=False
        ), patch(
            "tools.browser_tool._running_in_docker", return_value=False
        ), patch(
            "tools.browser_tool._find_agent_browser", return_value="npx agent-browser"
        ), patch(
            "hermes_cli.tools_config._print_success"
        ):
            _bt._cached_chromium_installed = True
            _run_post_setup("agent_browser")

        assert _bt._cached_chromium_installed is None, (
            "a successful install must invalidate the cached chromium-missing "
            "result so the next check_browser_requirements() call re-probes"
        )

    def test_install_failure_prints_stderr_tail_and_does_not_invalidate_cache(self):
        import tools.browser_tool as _bt

        with patch("shutil.which", return_value="/usr/bin/npx"), patch(
            "tools.browser_tool.node_tool_runnable", return_value=True
        ), patch(
            "subprocess.run",
            return_value=SimpleNamespace(
                returncode=1, stdout="", stderr="line1\nline2\nfatal: network error"
            ),
        ), patch(
            "tools.browser_tool._chromium_installed", return_value=False
        ), patch(
            "tools.browser_tool._running_in_docker", return_value=False
        ), patch(
            "tools.browser_tool._find_agent_browser", return_value="npx agent-browser"
        ), patch(
            "hermes_cli.tools_config._print_warning"
        ) as warn, patch(
            "hermes_cli.tools_config._print_info"
        ) as info:
            _bt._cached_chromium_installed = "sentinel"
            _run_post_setup("agent_browser")

        assert any("Chromium install failed" in c.args[0] for c in warn.call_args_list)
        assert any("fatal: network error" in c.args[0] for c in info.call_args_list)
        assert _bt._cached_chromium_installed == "sentinel", (
            "a failed install must not invalidate the chromium cache"
        )

    def test_install_timeout_warns_without_raising(self):
        with patch("shutil.which", return_value="/usr/bin/npx"), patch(
            "tools.browser_tool.node_tool_runnable", return_value=True
        ), patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["npx"], timeout=600),
        ), patch(
            "tools.browser_tool._chromium_installed", return_value=False
        ), patch(
            "tools.browser_tool._running_in_docker", return_value=False
        ), patch(
            "tools.browser_tool._find_agent_browser", return_value="npx agent-browser"
        ), patch(
            "hermes_cli.tools_config._print_warning"
        ) as warn:
            _run_post_setup("agent_browser")  # must not raise

        assert any("timed out" in c.args[0] for c in warn.call_args_list)


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
