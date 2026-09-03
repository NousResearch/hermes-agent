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


def test_null_platform_toolsets_fall_back_to_platform_default():
    """A YAML ``platform:`` value is absent, not an explicit empty list."""
    config = {"platform_toolsets": {"cli": None}}

    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    default_enabled = _get_platform_tools(
        {}, "cli", include_default_mcp_servers=False
    )

    assert enabled == default_enabled


def test_scalar_platform_toolsets_fall_back_to_platform_default():
    """A non-list platform value is ignored by the resolver."""
    config = {"platform_toolsets": {"cli": "bogus"}}

    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    default_enabled = _get_platform_tools(
        {}, "cli", include_default_mcp_servers=False
    )

    assert enabled == default_enabled










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


@pytest.mark.parametrize(
    ("toolset_names", "action", "expected"),
    [
        (["homeassistant"], "disable", ["terminal", "web"]),
        (["browser"], "enable", ["browser", "terminal", "web"]),
    ],
)
def test_apply_toolset_change_does_not_persist_runtime_injected_toolsets(
    monkeypatch, toolset_names, action, expected
):
    config = {"platform_toolsets": {"cli": ["web", "terminal"]}}
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools",
        lambda *args, **kwargs: {"web", "terminal", "kanban"},
    )

    _apply_toolset_change(config, "cli", toolset_names, action)

    assert config["platform_toolsets"]["cli"] == expected






















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
        "KEENABLE_API_KEY",
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

    assert config["video_gen"]["provider"] == "nous"
    assert "use_gateway" not in config["video_gen"]
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

    @pytest.fixture(autouse=True)
    def _stub_browser_use_install(self):
        """Both browser branches now attempt a Browser Use CLI install first
        (the CLI drives every non-Camofox backend). Stub it so these
        Chromium-branch tests never bootstrap uv / hit the network, and so
        their print/subprocess assertions stay scoped to the agent-browser
        logic under test."""
        with patch("hermes_cli.tools_config._ensure_browser_use_cli") as stub:
            yield stub

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
        agent-browser-only Chromium-install branch."""
        with patch("shutil.which", return_value="/usr/bin/npx"), patch(
            "subprocess.run"
        ) as run, patch(
            "tools.browser_tool._chromium_installed"
        ) as chromium_check:
            _run_post_setup("browserbase")

        run.assert_not_called()
        chromium_check.assert_not_called()

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


class TestBrowserUseCliInstalledForAllNonCamofoxBackends:
    """The Browser Use CLI is the primary driver engine for every browser
    backend except Camofox — so EVERY browser picker selection except
    Camofox must attempt the CLI install, not just the explicit
    "Browser Use" row."""

    @pytest.mark.parametrize("key", ["agent_browser", "browserbase", "browser_use_cli"])
    def test_browser_post_setup_attempts_cli_install(self, key):
        with patch("hermes_cli.tools_config._ensure_browser_use_cli") as ensure, patch(
            "shutil.which", return_value=None
        ), patch("subprocess.run"):
            _run_post_setup(key)
        ensure.assert_called_once()

    def test_camofox_post_setup_never_touches_browser_use(self):
        """Camofox is Firefox-based with no CDP surface; the CDP-only
        browser-use harness cannot drive it, so its setup must not pull
        the CLI in."""
        with patch("hermes_cli.tools_config._ensure_browser_use_cli") as ensure, patch(
            "hermes_constants.find_node_executable", return_value=None
        ), patch("subprocess.run"):
            _run_post_setup("camofox")
        ensure.assert_not_called()

    def test_ensure_helper_always_delegates_to_install_cli(self):
        """MANAGED-FIRST: a browser-use on PATH must not short-circuit the
        helper — install_cli() owns the managed-copy check and provisions
        $HERMES_HOME/bin when only side installs exist."""
        with patch(
            "hermes_cli.tools_config.shutil.which", return_value="/usr/bin/browser-use"
        ), patch(
            "tools.browser_use_cli.install_cli",
            return_value=(True, "browser-use CLI already installed (/managed/bin/browser-use)"),
        ) as install:
            from hermes_cli.tools_config import _ensure_browser_use_cli

            _ensure_browser_use_cli()
        install.assert_called_once()

    def test_ensure_helper_install_failure_is_non_fatal(self):
        """A failed install must warn and fall back, never raise — the
        uvx zero-install path and the built-in tools remain available."""
        from hermes_cli.tools_config import _ensure_browser_use_cli

        with patch(
            "hermes_cli.tools_config.shutil.which", return_value=None
        ), patch(
            "tools.browser_use_cli.install_cli",
            return_value=(False, "`uv tool install browser-use` failed:\nboom"),
        ), patch("hermes_cli.tools_config._print_warning") as warn:
            _ensure_browser_use_cli()  # must not raise

        assert any("failed" in c.args[0] for c in warn.call_args_list)


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

    def test_plugin_picker_falls_back_when_default_is_missing_from_catalog(self):
        """A stale cross-provider model must not become an unindexable row."""
        from hermes_cli.tools_config import _configure_imagegen_model_for_plugin

        catalog = {
            "openai/gpt-5.4-image-2": {"strengths": "quality"},
            "google/gemini-3-pro-image": {"strengths": "fallback"},
        }
        config = {"image_gen": {"model": "gpt-image-2-medium"}}
        with (
            patch(
                "hermes_cli.tools_config._plugin_image_gen_catalog",
                return_value=(catalog, "also-missing"),
            ),
            patch("hermes_cli.tools_config._prompt_choice", return_value=0),
        ):
            _configure_imagegen_model_for_plugin("openrouter", config)

        assert config["image_gen"]["model"] == "openai/gpt-5.4-image-2"






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
    """The bug: a frozen list never gained a newly shipped toolset, so
    composite users got it on upgrade and picker users silently did not."""
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
def test_agent_disabled_toolsets_json_array_string_form_still_wins():
    """#86661: the suppression list may arrive as a JSON-array string (e.g.
    `hermes config set agent.disabled_toolsets '["memory"]'`). It must be
    parsed, not treated as one dead toolset name that filters nothing."""
    config = _saved_list_from_before()
    import json as _json

    config["agent"] = {
        "disabled_toolsets": _json.dumps(sorted(_RECENTLY_SHIPPED_TOOLSETS))
    }

    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)

    assert not (_RECENTLY_SHIPPED_TOOLSETS & enabled)


@_requires_recently_shipped
def test_agent_disabled_toolsets_python_literal_string_form_still_wins():
    """Single-quoted Python-literal form (as written by some config editors)
    must resolve the same way as the JSON form."""
    config = _saved_list_from_before()
    quoted = ", ".join(repr(ts) for ts in sorted(_RECENTLY_SHIPPED_TOOLSETS))
    config["agent"] = {"disabled_toolsets": f"[{quoted}]"}

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


# Regression for issue #81163 (Layer 2): an explicitly-listed plugin toolset
# in ``platform_toolsets.<platform>`` must survive the filter, not be dropped
# because it isn't a built-in CONFIGURABLE_TOOLSETS entry.


def test_explicit_plugin_toolset_admitted_in_platform_toolsets(monkeypatch):
    """When a plugin toolset key is explicitly listed under
    ``platform_toolsets.<platform>`` (alongside a composite like
    ``hermes-cli``), it MUST be admitted as a configurable key instead of
    being silently dropped by the has_explicit_config filter.

    Reproduces the second half of #81163: even after the eager register_tools
    fix lands, ``_get_platform_tools`` was filtering against
    ``CONFIGURABLE_TOOLSETS`` only, so plugin keys in the explicit list were
    excluded from ``enabled_toolsets``.
    """
    # Force a plugin toolset key to be present without depending on the a2a
    # plugin being installed on disk. _get_plugin_toolset_keys() calls
    # discover_plugins(); we patch its source so the test is hermetic.
    import hermes_cli.plugins as _plugins_mod
    import hermes_cli.tools_config as _tc_mod

    class _StubMgr:
        _plugin_tool_names = {"dplat_call"}

        def __getattr__(self, _name):
            return lambda *_a, **_kw: None

    monkeypatch.setattr(
        _plugins_mod, "get_plugin_toolsets",
        lambda: [("dplat_client", "Test", "test toolset")],
    )
    monkeypatch.setattr(
        _tc_mod, "_get_plugin_toolset_keys", lambda: {"dplat_client"},
    )
    # Discover_plugins must succeed silently under the stub.
    monkeypatch.setattr(_plugins_mod, "discover_plugins", lambda: None)
    # Resolve dplat_call inside the dplat_client toolset — _get_platform_tools
    # ends up calling resolve_toolset() which can fall back to the registry
    # for plugin-provided names. Patch resolve_toolset for "dplat_client".
    from toolsets import TOOLSETS as _BASE_TOOLSETS
    import toolsets as _toolsets_mod

    original_resolve = _toolsets_mod.resolve_toolset

    def _resolve_with_plugin(ts_key, include_registry=True):
        if ts_key == "dplat_client":
            return ["dplat_call"]
        return original_resolve(ts_key, include_registry=include_registry)

    monkeypatch.setattr(_toolsets_mod, "resolve_toolset", _resolve_with_plugin)
    monkeypatch.setattr(
        _tc_mod, "resolve_toolset", _resolve_with_plugin,
        raising=False,
    )

    # An explicit platform_toolsets list with a plugin key alongside the
    # standard composite — exactly the "I want hermes-cli AND a2a in my CLI
    # session" config the issue's user was trying to write.
    config = {"platform_toolsets": {"cli": ["hermes-cli", "dplat_client"]}}

    enabled = _get_platform_tools(config, "cli")

    assert "dplat_client" in enabled, (
        "plugin toolset 'dplat_client' listed in platform_toolsets.cli was "
        "dropped by _get_platform_tools — Layer 2 of #81163 not fixed"
    )


def test_explicit_plugin_toolset_admitted_against_real_a2a_plugin(monkeypatch):
    """End-to-end Layer 2 regression: with the bundled a2a plugin enabled and
    a real config like ``platform_toolsets.cli: [hermes-cli, a2a]``, ``a2a``
    must appear in the resolved enabled toolset set. Before the fix, the
    filter dropped all non-CONFIGURABLE keys (a2a included)."""
    # Discover real plugins so _get_plugin_toolset_keys() sees the a2a key.
    # If the worktree lacks bundled plugin manifests, skip — this test
    # exercises real bundled state and is meaningless without it.
    from hermes_cli.plugins import discover_plugins, get_plugin_toolsets
    discover_plugins()
    plugin_ts_keys = {k for k, _, _ in get_plugin_toolsets()}
    if "a2a" not in plugin_ts_keys:
        pytest.skip("bundled a2a plugin not discoverable in this worktree")

    config = {"platform_toolsets": {"cli": ["hermes-cli", "a2a"]}}
    enabled = _get_platform_tools(config, "cli")
    assert "a2a" in enabled, (
        f"plugin-provided 'a2a' toolset dropped by _get_platform_tools "
        f"(Layer 2 of #81163); enabled={sorted(enabled)}"
    )


class TestLightpandaPostSetup:
    """The Lightpanda picker row: no Chromium, just the binary check."""

    @pytest.fixture(autouse=True)
    def _stub_browser_use_install(self):
        with patch("hermes_cli.tools_config._ensure_browser_use_cli") as stub:
            yield stub

    def test_reports_binary_when_found(self, _stub_browser_use_install):
        from hermes_cli.tools_config import _run_post_setup

        with patch("tools.browser_lightpanda.find_lightpanda_binary", return_value="/opt/lightpanda"), \
             patch("hermes_cli.tools_config._print_success") as ok, \
             patch("hermes_cli.tools_config._print_warning") as warn:
            _run_post_setup("lightpanda")
        _stub_browser_use_install.assert_called_once()
        assert "/opt/lightpanda" in ok.call_args.args[0]
        warn.assert_not_called()

    def test_prints_install_hint_when_missing(self):
        from hermes_cli.tools_config import _run_post_setup
        from tools.browser_lightpanda import LIGHTPANDA_INSTALL_URL

        with patch("tools.browser_lightpanda.find_lightpanda_binary", return_value=None), \
             patch("hermes_cli.tools_config._print_warning") as warn, \
             patch("hermes_cli.tools_config._print_info") as info:
            _run_post_setup("lightpanda")
        assert "not found" in warn.call_args.args[0]
        assert any(LIGHTPANDA_INSTALL_URL in c.args[0] for c in info.call_args_list)

    def test_post_setup_key_is_valid_and_readiness_gated(self):
        from hermes_cli.tools_config import (
            _POST_SETUP_INSTALLED,
            _POST_SETUP_READY,
            valid_post_setup_keys,
        )

        assert "lightpanda" in valid_post_setup_keys()
        with patch("tools.browser_lightpanda.find_lightpanda_binary", return_value="/opt/lightpanda"):
            assert _POST_SETUP_READY["lightpanda"]() is True
        with patch("tools.browser_lightpanda.find_lightpanda_binary", return_value=None):
            assert _POST_SETUP_READY["lightpanda"]() is False
        # Not in the forced-setup gate: a missing binary must not nag every
        # user who toggles the browser toolset.
        assert "lightpanda" not in _POST_SETUP_INSTALLED
