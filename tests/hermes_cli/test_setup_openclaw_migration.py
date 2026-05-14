"""Tests for OpenClaw migration integration in the setup wizard."""

from argparse import Namespace
from types import ModuleType
from unittest.mock import MagicMock, patch

from hermes_cli import setup as setup_mod


# ---------------------------------------------------------------------------
# _offer_openclaw_migration — unit tests
# ---------------------------------------------------------------------------


class TestOfferOpenclawMigration:
    """Test the _offer_openclaw_migration helper in isolation."""

    def test_skips_when_no_openclaw_dir(self, tmp_path):
        """Should return False immediately when ~/.openclaw does not exist."""
        with patch("hermes_cli.setup.Path.home", return_value=tmp_path):
            assert setup_mod._offer_openclaw_migration(tmp_path / ".hermes") is False

    def test_skips_when_migration_script_missing(self, tmp_path):
        """Should return False when the migration script file is absent."""
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        with (
            patch("hermes_cli.setup.Path.home", return_value=tmp_path),
            patch.object(setup_mod, "_OPENCLAW_SCRIPT", tmp_path / "nonexistent.py"),
        ):
            assert setup_mod._offer_openclaw_migration(tmp_path / ".hermes") is False

    def test_skips_when_user_declines(self, tmp_path):
        """Should return False when user declines the migration prompt."""
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        script = tmp_path / "openclaw_to_hermes.py"
        script.write_text("# placeholder")
        with (
            patch("hermes_cli.setup.Path.home", return_value=tmp_path),
            patch.object(setup_mod, "_OPENCLAW_SCRIPT", script),
            patch.object(setup_mod, "prompt_yes_no", return_value=False),
        ):
            assert setup_mod._offer_openclaw_migration(tmp_path / ".hermes") is False

    def test_runs_migration_when_user_accepts(self, tmp_path):
        """Should run dry-run preview first, then execute after confirmation."""
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()

        # Create a fake hermes home with config
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("agent:\n  max_turns: 90\n")

        # Build a fake migration module
        fake_mod = ModuleType("openclaw_to_hermes")
        fake_mod.resolve_selected_options = MagicMock(return_value={"soul", "memory"})
        fake_migrator = MagicMock()
        fake_migrator.migrate.return_value = {
            "summary": {"migrated": 3, "skipped": 1, "conflict": 0, "error": 0},
            "items": [{"kind": "config", "status": "migrated", "destination": "/tmp/x"}],
            "output_dir": str(hermes_home / "migration"),
        }
        fake_mod.Migrator = MagicMock(return_value=fake_migrator)

        script = tmp_path / "openclaw_to_hermes.py"
        script.write_text("# placeholder")

        with (
            patch("hermes_cli.setup.Path.home", return_value=tmp_path),
            patch.object(setup_mod, "_OPENCLAW_SCRIPT", script),
            # Both prompts answered Yes: preview offer + proceed confirmation
            patch.object(setup_mod, "prompt_yes_no", return_value=True),
            patch.object(setup_mod, "get_config_path", return_value=config_path),
            patch("importlib.util.spec_from_file_location") as mock_spec_fn,
        ):
            # Wire up the fake module loading
            mock_spec = MagicMock()
            mock_spec.loader = MagicMock()
            mock_spec_fn.return_value = mock_spec

            def exec_module(mod):
                mod.resolve_selected_options = fake_mod.resolve_selected_options
                mod.Migrator = fake_mod.Migrator

            mock_spec.loader.exec_module = exec_module

            result = setup_mod._offer_openclaw_migration(hermes_home)

        assert result is True
        fake_mod.resolve_selected_options.assert_called_once_with(
            None, None, preset="full"
        )
        # Migrator called twice: once for dry-run preview, once for execution
        assert fake_mod.Migrator.call_count == 2

        # First call: dry-run preview (execute=False, overwrite=True to show all)
        preview_kwargs = fake_mod.Migrator.call_args_list[0][1]
        assert preview_kwargs["execute"] is False
        assert preview_kwargs["overwrite"] is True
        assert preview_kwargs["migrate_secrets"] is True
        assert preview_kwargs["preset_name"] == "full"

        # Second call: actual execution (execute=True, overwrite=False to preserve)
        exec_kwargs = fake_mod.Migrator.call_args_list[1][1]
        assert exec_kwargs["execute"] is True
        assert exec_kwargs["overwrite"] is False
        assert exec_kwargs["migrate_secrets"] is True
        assert exec_kwargs["preset_name"] == "full"

        # migrate() called twice (once per Migrator instance)
        assert fake_migrator.migrate.call_count == 2

    def test_user_declines_after_preview(self, tmp_path):
        """Should return False when user sees preview but declines to proceed."""
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("agent:\n  max_turns: 90\n")

        fake_mod = ModuleType("openclaw_to_hermes")
        fake_mod.resolve_selected_options = MagicMock(return_value={"soul", "memory"})
        fake_migrator = MagicMock()
        fake_migrator.migrate.return_value = {
            "summary": {"migrated": 3, "skipped": 0, "conflict": 0, "error": 0},
            "items": [{"kind": "config", "status": "migrated", "destination": "/tmp/x"}],
        }
        fake_mod.Migrator = MagicMock(return_value=fake_migrator)

        script = tmp_path / "openclaw_to_hermes.py"
        script.write_text("# placeholder")

        # First prompt (preview): Yes, Second prompt (proceed): No
        prompt_responses = iter([True, False])

        with (
            patch("hermes_cli.setup.Path.home", return_value=tmp_path),
            patch.object(setup_mod, "_OPENCLAW_SCRIPT", script),
            patch.object(setup_mod, "prompt_yes_no", side_effect=prompt_responses),
            patch.object(setup_mod, "get_config_path", return_value=config_path),
            patch("importlib.util.spec_from_file_location") as mock_spec_fn,
        ):
            mock_spec = MagicMock()
            mock_spec.loader = MagicMock()
            mock_spec_fn.return_value = mock_spec

            def exec_module(mod):
                mod.resolve_selected_options = fake_mod.resolve_selected_options
                mod.Migrator = fake_mod.Migrator

            mock_spec.loader.exec_module = exec_module

            result = setup_mod._offer_openclaw_migration(hermes_home)

        assert result is False
        # Only dry-run Migrator was created, not the execute one
        assert fake_mod.Migrator.call_count == 1
        preview_kwargs = fake_mod.Migrator.call_args[1]
        assert preview_kwargs["execute"] is False

    def test_handles_migration_error_gracefully(self, tmp_path):
        """Should catch exceptions and return False."""
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("")

        script = tmp_path / "openclaw_to_hermes.py"
        script.write_text("# placeholder")

        with (
            patch("hermes_cli.setup.Path.home", return_value=tmp_path),
            patch.object(setup_mod, "_OPENCLAW_SCRIPT", script),
            patch.object(setup_mod, "prompt_yes_no", return_value=True),
            patch.object(setup_mod, "get_config_path", return_value=config_path),
            patch(
                "importlib.util.spec_from_file_location",
                side_effect=RuntimeError("boom"),
            ),
        ):
            result = setup_mod._offer_openclaw_migration(hermes_home)

        assert result is False

    def test_creates_config_if_missing(self, tmp_path):
        """Should bootstrap config.yaml before running migration."""
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        # config does NOT exist yet

        script = tmp_path / "openclaw_to_hermes.py"
        script.write_text("# placeholder")

        with (
            patch("hermes_cli.setup.Path.home", return_value=tmp_path),
            patch.object(setup_mod, "_OPENCLAW_SCRIPT", script),
            patch.object(setup_mod, "prompt_yes_no", return_value=True),
            patch.object(setup_mod, "get_config_path", return_value=config_path),
            patch.object(setup_mod, "load_config", return_value={"agent": {}}),
            patch.object(setup_mod, "save_config") as mock_save,
            patch(
                "importlib.util.spec_from_file_location",
                side_effect=RuntimeError("stop early"),
            ),
        ):
            setup_mod._offer_openclaw_migration(hermes_home)

        # save_config should have been called to bootstrap the file
        mock_save.assert_called_once_with({"agent": {}})


# ---------------------------------------------------------------------------
# Integration with run_setup_wizard — first-time flow
# ---------------------------------------------------------------------------


def _first_time_args() -> Namespace:
    return Namespace(
        section=None,
        non_interactive=False,
        reset=False,
    )


class TestSetupWizardOpenclawIntegration:
    """Verify _offer_openclaw_migration is called during first-time setup."""

    def test_migration_offered_during_first_time_setup(self, tmp_path):
        """On first-time setup, _offer_openclaw_migration should be called."""
        args = _first_time_args()

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(setup_mod, "load_config", return_value={}),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(setup_mod, "get_env_value", return_value=""),
            patch.object(setup_mod, "is_interactive_stdin", return_value=True),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            # User presses Enter to start
            patch("builtins.input", return_value=""),
            # Select "Full setup" (index 1) so we exercise the full path
            patch.object(setup_mod, "prompt_choice", return_value=1),
            # Mock the migration offer
            patch.object(
                setup_mod, "_offer_openclaw_migration", return_value=False
            ) as mock_migration,
            # Mock the actual setup sections so they don't run
            patch.object(setup_mod, "setup_model_provider"),
            patch.object(setup_mod, "setup_terminal_backend"),
            patch.object(setup_mod, "setup_agent_settings"),
            patch.object(setup_mod, "setup_gateway"),
            patch.object(setup_mod, "setup_tools"),
            patch.object(setup_mod, "save_config"),
            patch.object(setup_mod, "_print_setup_summary"),
        ):
            setup_mod.run_setup_wizard(args)

        mock_migration.assert_called_once_with(tmp_path)

    def test_migration_reloads_config_on_success(self, tmp_path):
        """When migration returns True, config should be reloaded."""
        args = _first_time_args()
        call_order = []

        def tracking_load_config():
            call_order.append("load_config")
            return {}

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(setup_mod, "load_config", side_effect=tracking_load_config),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(setup_mod, "get_env_value", return_value=""),
            patch.object(setup_mod, "is_interactive_stdin", return_value=True),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            patch("builtins.input", return_value=""),
            patch.object(setup_mod, "prompt_choice", return_value=1),
            patch.object(setup_mod, "_offer_openclaw_migration", return_value=True),
            patch.object(setup_mod, "setup_model_provider"),
            patch.object(setup_mod, "setup_terminal_backend"),
            patch.object(setup_mod, "setup_agent_settings"),
            patch.object(setup_mod, "setup_gateway"),
            patch.object(setup_mod, "setup_tools"),
            patch.object(setup_mod, "save_config"),
            patch.object(setup_mod, "_print_setup_summary"),
        ):
            setup_mod.run_setup_wizard(args)

        # load_config called twice: once at start, once after migration
        assert call_order.count("load_config") == 2

    def test_reloaded_config_flows_into_remaining_setup_sections(self, tmp_path):
        args = _first_time_args()
        initial_config = {}
        reloaded_config = {"model": {"provider": "openrouter"}}

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(
                setup_mod,
                "load_config",
                side_effect=[initial_config, reloaded_config],
            ),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(setup_mod, "get_env_value", return_value=""),
            patch.object(setup_mod, "is_interactive_stdin", return_value=True),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            patch("builtins.input", return_value=""),
            patch.object(setup_mod, "prompt_choice", return_value=1),
            patch.object(setup_mod, "_offer_openclaw_migration", return_value=True),
            patch.object(setup_mod, "setup_model_provider") as setup_model_provider,
            patch.object(setup_mod, "setup_terminal_backend"),
            patch.object(setup_mod, "setup_agent_settings"),
            patch.object(setup_mod, "setup_gateway"),
            patch.object(setup_mod, "setup_tools"),
            patch.object(setup_mod, "save_config"),
            patch.object(setup_mod, "_print_setup_summary"),
        ):
            setup_mod.run_setup_wizard(args)

        setup_model_provider.assert_called_once_with(reloaded_config)

    def test_migration_not_offered_for_existing_install(self, tmp_path):
        """Returning users should not see the migration prompt."""
        args = _first_time_args()

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(setup_mod, "load_config", return_value={}),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(
                setup_mod,
                "get_env_value",
                side_effect=lambda k: "sk-xxx" if k == "OPENROUTER_API_KEY" else "",
            ),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            # Returning user picks "Exit"
            patch.object(setup_mod, "prompt_choice", return_value=9),
            patch.object(
                setup_mod, "_offer_openclaw_migration", return_value=False
            ) as mock_migration,
        ):
            setup_mod.run_setup_wizard(args)

        mock_migration.assert_not_called()


# ---------------------------------------------------------------------------
# _get_section_config_summary / _skip_configured_section — unit tests
# ---------------------------------------------------------------------------


class TestGetSectionConfigSummary:
    """Test the _get_section_config_summary helper."""

    def test_model_returns_none_without_api_key(self):
        with patch.object(setup_mod, "get_env_value", return_value=""):
            result = setup_mod._get_section_config_summary({}, "model")
        assert result is None

    def test_model_returns_summary_with_api_key(self):
        def env_side(key):
            return "sk-xxx" if key == "OPENROUTER_API_KEY" else ""

        with patch.object(setup_mod, "get_env_value", side_effect=env_side):
            result = setup_mod._get_section_config_summary(
                {"model": "openai/gpt-4"}, "model"
            )
        assert result == "openai/gpt-4"

    def test_model_returns_dict_default_key(self):
        def env_side(key):
            return "sk-xxx" if key == "OPENAI_API_KEY" else ""

        with patch.object(setup_mod, "get_env_value", side_effect=env_side):
            result = setup_mod._get_section_config_summary(
                {"model": {"default": "claude-opus-4", "provider": "anthropic"}},
                "model",
            )
        assert result == "claude-opus-4"

    def test_terminal_always_returns(self):
        with patch.object(setup_mod, "get_env_value", return_value=""):
            result = setup_mod._get_section_config_summary(
                {"terminal": {"backend": "docker"}}, "terminal"
            )
        assert result == "backend: docker"

    def test_agent_always_returns(self):
        with patch.object(setup_mod, "get_env_value", return_value=""):
            result = setup_mod._get_section_config_summary(
                {"agent": {"max_turns": 120}}, "agent"
            )
        assert result == "max turns: 120"

    def test_gateway_returns_none_without_tokens(self):
        with patch.object(setup_mod, "get_env_value", return_value=""):
            result = setup_mod._get_section_config_summary({}, "gateway")
        assert result is None

    def test_gateway_lists_platforms(self):
        def env_side(key):
            if key == "TELEGRAM_BOT_TOKEN":
                return "tok123"
            if key == "DISCORD_BOT_TOKEN":
                return "disc456"
            return ""

        # Also patch gateway module's binding since _platform_status()
        # reads from hermes_cli.gateway.get_env_value after the setup
        # flows were unified via platform_registry.
        import hermes_cli.gateway as gateway_mod
        with patch.object(setup_mod, "get_env_value", side_effect=env_side), \
             patch.object(gateway_mod, "get_env_value", side_effect=env_side):
            result = setup_mod._get_section_config_summary({}, "gateway")
        assert "Telegram" in result
        assert "Discord" in result

    def test_tools_returns_none_without_keys(self):
        with patch.object(setup_mod, "get_env_value", return_value=""):
            result = setup_mod._get_section_config_summary({}, "tools")
        assert result is None

    def test_tools_lists_configured(self):
        def env_side(key):
            return "key" if key == "BROWSERBASE_API_KEY" else ""

        with patch.object(setup_mod, "get_env_value", side_effect=env_side):
            result = setup_mod._get_section_config_summary({}, "tools")
        assert "Browser" in result

    # Regression tests for issue #13025: the model / gateway summaries used
    # stale, hardcoded env-var allowlists that drifted from the real setup +
    # status flows.  Every case below would previously return ``None`` and
    # force OpenClaw migration to re-run setup for an already-configured
    # section.

    def test_model_recognises_zai_glm_api_key(self):
        """GLM_API_KEY (zai provider) should count as configured."""
        def env_side(key):
            return "glm-test-key" if key == "GLM_API_KEY" else ""

        with patch.object(setup_mod, "get_env_value", side_effect=env_side):
            result = setup_mod._get_section_config_summary(
                {"model": {"provider": "zai", "default": "glm-5"}}, "model"
            )
        assert result == "glm-5"

    def test_model_recognises_minimax_api_key(self):
        """MINIMAX_API_KEY should count as configured."""
        def env_side(key):
            return "minimax-key" if key == "MINIMAX_API_KEY" else ""

        with patch.object(setup_mod, "get_env_value", side_effect=env_side):
            result = setup_mod._get_section_config_summary(
                {"model": {"provider": "minimax", "default": "MiniMax-M1"}},
                "model",
            )
        assert result == "MiniMax-M1"

    def test_gateway_recognises_whatsapp_enabled(self):
        """WhatsApp uses WHATSAPP_ENABLED (not WHATSAPP_PHONE_NUMBER_ID)."""
        def env_side(key):
            return "true" if key == "WHATSAPP_ENABLED" else ""

        import hermes_cli.gateway as gateway_mod
        with patch.object(setup_mod, "get_env_value", side_effect=env_side), \
             patch.object(gateway_mod, "get_env_value", side_effect=env_side):
            result = setup_mod._get_section_config_summary({}, "gateway")
        assert result is not None
        assert "WhatsApp" in result

    def test_gateway_recognises_signal_http_url(self):
        """Signal uses SIGNAL_HTTP_URL (not SIGNAL_ACCOUNT)."""
        def env_side(key):
            return "http://signal.local" if key == "SIGNAL_HTTP_URL" else ""

        import hermes_cli.gateway as gateway_mod
        with patch.object(setup_mod, "get_env_value", side_effect=env_side), \
             patch.object(gateway_mod, "get_env_value", side_effect=env_side):
            result = setup_mod._get_section_config_summary({}, "gateway")
        assert result is not None
        assert "Signal" in result

    def test_model_ignores_bare_gh_token(self):
        """GH_TOKEN is commonly set for `gh` / git and must NOT count as a
        configured inference provider on its own — mirrors the copilot
        exclusion in resolve_provider()."""
        def env_side(key):
            return "gho_xxx" if key == "GH_TOKEN" else ""

        with patch.object(setup_mod, "get_env_value", side_effect=env_side):
            result = setup_mod._get_section_config_summary({}, "model")
        assert result is None

    def test_model_ignores_bare_github_token(self):
        """GITHUB_TOKEN is commonly set in CI and must not trigger skip."""
        def env_side(key):
            return "ghp_xxx" if key == "GITHUB_TOKEN" else ""

        with patch.object(setup_mod, "get_env_value", side_effect=env_side):
            result = setup_mod._get_section_config_summary({}, "model")
        assert result is None

    def test_model_ignores_claude_code_oauth_token(self):
        """CLAUDE_CODE_OAUTH_TOKEN is set by Claude Code itself and must not
        trigger skip — mirrors the _IMPLICIT_ENV_VARS guard in
        is_provider_explicitly_configured()."""
        def env_side(key):
            return "sk-ant-oat01-xxx" if key == "CLAUDE_CODE_OAUTH_TOKEN" else ""

        with patch.object(setup_mod, "get_env_value", side_effect=env_side):
            result = setup_mod._get_section_config_summary({}, "model")
        assert result is None

    def test_model_copilot_recognised_when_explicitly_chosen(self):
        """If the user picked copilot in config, GH_TOKEN *does* count —
        only the auto-detect path excludes it."""
        def env_side(key):
            return "gho_xxx" if key == "GH_TOKEN" else ""

        cfg = {"model": {"provider": "copilot", "default": "gpt-5"}}
        with patch.object(setup_mod, "get_env_value", side_effect=env_side):
            result = setup_mod._get_section_config_summary(cfg, "model")
        assert result == "gpt-5"

    def test_gateway_matches_platform_registry(self):
        """Every built-in platform should be recognised by its primary
        env-var sentinel — i.e. the summary must not drift from the
        registry used by the setup checklist."""
        from hermes_cli.gateway import _PLATFORMS

        for plat in _PLATFORMS:
            label = plat["label"]
            env_var = plat.get("token_var")
            if not env_var:
                continue
            # Some platforms require a specific value shape (e.g. WhatsApp
            # needs the literal "true"). Use a sentinel that satisfies every
            # real validator _platform_status() currently checks.
            def env_side(key, _target=env_var):
                if key != _target:
                    return ""
                if _target == "WHATSAPP_ENABLED":
                    return "true"
                return "x"
            import hermes_cli.gateway as gateway_mod
            with patch.object(setup_mod, "get_env_value", side_effect=env_side), \
                 patch.object(gateway_mod, "get_env_value", side_effect=env_side):
                result = setup_mod._get_section_config_summary({}, "gateway")
            expected = setup_mod._gateway_platform_short_label(label)
            assert result is not None, f"{label} ({env_var}) not recognised"
            assert expected in result, (
                f"{label} ({env_var}) recognised but label missing from summary: {result!r}"
            )


class TestSkipConfiguredSection:
    """Test the _skip_configured_section helper."""

    def test_returns_false_when_not_configured(self):
        with patch.object(setup_mod, "get_env_value", return_value=""):
            result = setup_mod._skip_configured_section({}, "model", "Model")
        assert result is False

    def test_returns_true_when_user_skips(self):
        def env_side(key):
            return "sk-xxx" if key == "OPENROUTER_API_KEY" else ""

        with (
            patch.object(setup_mod, "get_env_value", side_effect=env_side),
            patch.object(setup_mod, "prompt_yes_no", return_value=False),
        ):
            result = setup_mod._skip_configured_section(
                {"model": "openai/gpt-4"}, "model", "Model"
            )
        assert result is True

    def test_returns_false_when_user_wants_reconfig(self):
        def env_side(key):
            return "sk-xxx" if key == "OPENROUTER_API_KEY" else ""

        with (
            patch.object(setup_mod, "get_env_value", side_effect=env_side),
            patch.object(setup_mod, "prompt_yes_no", return_value=True),
        ):
            result = setup_mod._skip_configured_section(
                {"model": "openai/gpt-4"}, "model", "Model"
            )
        assert result is False


class TestSetupWizardSkipsConfiguredSections:
    """After migration, already-configured sections should offer skip."""

    def test_sections_skipped_when_migration_imported_settings(self, tmp_path):
        """When migration ran and API key exists, model section should be skippable.

        Simulates the real flow: get_env_value returns "" during the is_existing
        check (before migration), then returns a key after migration imported it.
        """
        args = _first_time_args()

        # Track whether migration has "run" — after it does, API key is available
        migration_done = {"value": False}

        def env_side(key):
            if migration_done["value"] and key == "OPENROUTER_API_KEY":
                return "sk-xxx"
            return ""

        def fake_migration(hermes_home):
            migration_done["value"] = True
            return True

        reloaded_config = {"model": "openai/gpt-4"}

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(
                setup_mod, "load_config",
                side_effect=[{}, reloaded_config],
            ),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(setup_mod, "get_env_value", side_effect=env_side),
            patch.object(setup_mod, "is_interactive_stdin", return_value=True),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            patch("builtins.input", return_value=""),
            patch.object(setup_mod, "prompt_choice", return_value=1),
            # Migration succeeds and flips the env_side flag
            patch.object(
                setup_mod, "_offer_openclaw_migration",
                side_effect=fake_migration,
            ),
            # User says No to all reconfig prompts
            patch.object(setup_mod, "prompt_yes_no", return_value=False),
            patch.object(setup_mod, "setup_model_provider") as mock_model,
            patch.object(setup_mod, "setup_terminal_backend") as mock_terminal,
            patch.object(setup_mod, "setup_agent_settings") as mock_agent,
            patch.object(setup_mod, "setup_gateway") as mock_gateway,
            patch.object(setup_mod, "setup_tools") as mock_tools,
            patch.object(setup_mod, "save_config"),
            patch.object(setup_mod, "_print_setup_summary"),
        ):
            setup_mod.run_setup_wizard(args)

        # Model has API key → skip offered, user said No → section NOT called
        mock_model.assert_not_called()
        # Terminal/agent always have a summary → skip offered, user said No
        mock_terminal.assert_not_called()
        mock_agent.assert_not_called()
        # Gateway has no tokens (env_side returns "" for gateway keys) → section runs
        mock_gateway.assert_called_once()
        # Tools have no keys → section runs
        mock_tools.assert_called_once()
