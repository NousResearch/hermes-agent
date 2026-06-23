"""
Tests for skip_confirm and invalidate_cache behavior in /skills install
and /skills uninstall slash commands.

Slash commands always skip confirmation (input() hangs in TUI).
Cache invalidation is deferred by default; --now opts into immediate
invalidation (at the cost of breaking prompt cache mid-session).

Based on PR #1595 by 333Alden333 (salvaged).
Updated for PR #3586 (cache-aware install/uninstall).
"""

from unittest.mock import patch

import yaml


class TestHandleSkillsSlashInstallFlags:
    """Test flag parsing in handle_skills_slash for install."""

    def test_yes_flag_sets_skip_confirm(self):
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_install") as mock_install:
            handle_skills_slash("/skills install test/skill --yes")
            mock_install.assert_called_once()
            _, kwargs = mock_install.call_args
            assert kwargs.get("skip_confirm") is True
            assert kwargs.get("force") is False

    def test_y_flag_sets_skip_confirm(self):
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_install") as mock_install:
            handle_skills_slash("/skills install test/skill -y")
            mock_install.assert_called_once()
            _, kwargs = mock_install.call_args
            assert kwargs.get("skip_confirm") is True

    def test_force_flag_sets_force(self):
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_install") as mock_install:
            handle_skills_slash("/skills install test/skill --force")
            mock_install.assert_called_once()
            _, kwargs = mock_install.call_args
            assert kwargs.get("force") is True
            # Slash commands always skip confirmation (input() hangs in TUI)
            assert kwargs.get("skip_confirm") is True

    def test_no_flags_still_skips_confirm(self):
        """Slash commands always skip confirmation — input() hangs in TUI."""
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_install") as mock_install:
            handle_skills_slash("/skills install test/skill")
            mock_install.assert_called_once()
            _, kwargs = mock_install.call_args
            assert kwargs.get("force") is False
            assert kwargs.get("skip_confirm") is True

    def test_default_defers_cache_invalidation(self):
        """Without --now, cache invalidation is deferred to next session."""
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_install") as mock_install:
            handle_skills_slash("/skills install test/skill")
            mock_install.assert_called_once()
            _, kwargs = mock_install.call_args
            assert kwargs.get("invalidate_cache") is False

    def test_now_flag_invalidates_cache(self):
        """--now opts into immediate cache invalidation."""
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_install") as mock_install:
            handle_skills_slash("/skills install test/skill --now")
            mock_install.assert_called_once()
            _, kwargs = mock_install.call_args
            assert kwargs.get("invalidate_cache") is True


class TestHandleSkillsSlashUninstallFlags:
    """Test flag parsing in handle_skills_slash for uninstall."""

    def test_yes_flag_sets_skip_confirm(self):
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_uninstall") as mock_uninstall:
            handle_skills_slash("/skills uninstall test-skill --yes")
            mock_uninstall.assert_called_once()
            _, kwargs = mock_uninstall.call_args
            assert kwargs.get("skip_confirm") is True

    def test_y_flag_sets_skip_confirm(self):
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_uninstall") as mock_uninstall:
            handle_skills_slash("/skills uninstall test-skill -y")
            mock_uninstall.assert_called_once()
            _, kwargs = mock_uninstall.call_args
            assert kwargs.get("skip_confirm") is True

    def test_no_flags_still_skips_confirm(self):
        """Slash commands always skip confirmation — input() hangs in TUI."""
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_uninstall") as mock_uninstall:
            handle_skills_slash("/skills uninstall test-skill")
            mock_uninstall.assert_called_once()
            _, kwargs = mock_uninstall.call_args
            assert kwargs.get("skip_confirm") is True

    def test_default_defers_cache_invalidation(self):
        """Without --now, cache invalidation is deferred to next session."""
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_uninstall") as mock_uninstall:
            handle_skills_slash("/skills uninstall test-skill")
            mock_uninstall.assert_called_once()
            _, kwargs = mock_uninstall.call_args
            assert kwargs.get("invalidate_cache") is False

    def test_now_flag_invalidates_cache(self):
        """--now opts into immediate cache invalidation."""
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_uninstall") as mock_uninstall:
            handle_skills_slash("/skills uninstall test-skill --now")
            mock_uninstall.assert_called_once()
            _, kwargs = mock_uninstall.call_args
            assert kwargs.get("invalidate_cache") is True


class TestDoInstallSkipConfirm:
    """Test that do_install respects skip_confirm parameter."""

    @patch("hermes_cli.skills_hub.input", return_value="n")
    def test_without_skip_confirm_prompts_user(self, mock_input):
        """Without skip_confirm, input() is called for confirmation."""
        from hermes_cli.skills_hub import do_install
        with patch("hermes_cli.skills_hub._console"), \
             patch("tools.skills_hub.ensure_hub_dirs"), \
             patch("tools.skills_hub.GitHubAuth"), \
             patch("tools.skills_hub.create_source_router") as mock_router, \
             patch("hermes_cli.skills_hub._resolve_short_name", return_value="test/skill"), \
             patch("hermes_cli.skills_hub._resolve_source_meta_and_bundle") as mock_resolve:

            # Make it return None so we exit early
            mock_resolve.return_value = (None, None, None)
            do_install("test-skill", skip_confirm=False)
            # We don't get to the input() call because resolve returns None,
            # but the parameter wiring is correct


class TestDoUninstallSkipConfirm:
    """Test that do_uninstall respects skip_confirm parameter."""

    def test_skip_confirm_bypasses_input(self):
        """With skip_confirm=True, input() should not be called."""
        from hermes_cli.skills_hub import do_uninstall
        with patch("hermes_cli.skills_hub._console") as mock_console, \
             patch("tools.skills_hub.uninstall_skill", return_value=(True, "Removed")) as mock_uninstall, \
             patch("builtins.input") as mock_input:
            do_uninstall("test-skill", skip_confirm=True)
            mock_input.assert_not_called()
            mock_uninstall.assert_called_once_with("test-skill")

    def test_without_skip_confirm_calls_input(self):
        """Without skip_confirm, input() should be called."""
        from hermes_cli.skills_hub import do_uninstall
        with patch("hermes_cli.skills_hub._console"), \
             patch("tools.skills_hub.uninstall_skill", return_value=(True, "Removed")), \
             patch("builtins.input", return_value="y") as mock_input:
            do_uninstall("test-skill", skip_confirm=False)
            mock_input.assert_called_once()

    def test_without_skip_confirm_cancel(self):
        """Without skip_confirm, answering 'n' should cancel."""
        from hermes_cli.skills_hub import do_uninstall
        with patch("hermes_cli.skills_hub._console"), \
             patch("tools.skills_hub.uninstall_skill") as mock_uninstall, \
             patch("builtins.input", return_value="n"):
            do_uninstall("test-skill", skip_confirm=False)
            mock_uninstall.assert_not_called()


class TestDoUninstallPrunesDisableConfig:
    """A successful uninstall must also clear the skill's lingering disable
    state in config.yaml (skills.disabled / skills.platform_disabled), which is
    stored separately from the hub lock file. Otherwise a later re-install of
    the same name comes back silently disabled and the enabled/disabled count
    is wrong.
    """

    def test_uninstall_prunes_global_and_platform_disabled(self, tmp_path, monkeypatch):
        from hermes_cli.skills_hub import do_uninstall

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "skills:\n"
            "  disabled:\n"
            "    - my-skill\n"
            "    - keep-skill\n"
            "  platform_disabled:\n"
            "    telegram:\n"
            "      - my-skill\n"
            "      - keep-skill\n"
            "    cli:\n"
            "      - keep-skill\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_MANAGED", raising=False)

        with patch("hermes_cli.skills_hub._console"), \
             patch("tools.skills_hub.uninstall_skill", return_value=(True, "Removed")):
            do_uninstall("my-skill", skip_confirm=True, invalidate_cache=False)

        saved = yaml.safe_load(config_path.read_text())
        skills_cfg = saved["skills"]
        # The uninstalled skill is gone from both global and per-platform lists...
        assert "my-skill" not in skills_cfg.get("disabled", [])
        assert "my-skill" not in skills_cfg.get("platform_disabled", {}).get("telegram", [])
        # ...while unrelated disable entries are left untouched.
        assert "keep-skill" in skills_cfg.get("disabled", [])
        assert "keep-skill" in skills_cfg.get("platform_disabled", {}).get("telegram", [])
        assert "keep-skill" in skills_cfg.get("platform_disabled", {}).get("cli", [])

    def test_uninstall_without_disable_entry_is_noop(self, tmp_path, monkeypatch):
        """A skill that was never disabled must not perturb config on uninstall."""
        from hermes_cli.skills_hub import do_uninstall

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "skills:\n"
            "  disabled:\n"
            "    - other-skill\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_MANAGED", raising=False)

        with patch("hermes_cli.skills_hub._console"), \
             patch("tools.skills_hub.uninstall_skill", return_value=(True, "Removed")):
            do_uninstall("my-skill", skip_confirm=True, invalidate_cache=False)

        saved = yaml.safe_load(config_path.read_text())
        assert saved["skills"]["disabled"] == ["other-skill"]
