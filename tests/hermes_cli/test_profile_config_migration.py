"""Behavioral coverage for multi-profile config migration entry points.

Covers the two paths identified in PR #62492 review:

1. ``cmd_update`` → ``_migrate_all_profiles()`` in ``update_cmd.py``
   (lists profiles, migrates each, surfaces per-profile failures on stderr)
2. ``cmd_dashboard`` / ``serve`` → ``_migrate_active_profile_on_startup()``
   in ``main.py`` (auto-migrate or warn on stderr for stale config)

Tests call the REAL production helpers with dependencies patched.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# 1. _migrate_profile_config — per-profile migration helper (4 tests)
# ---------------------------------------------------------------------------


class TestMigrateProfileConfig:
    """Unit tests for ``hermes_cli.main._migrate_profile_config``."""

    def test_version_bump_only_migrates_silently(self, capsys):
        """When only a version bump is needed (no missing fields), migration
        is applied silently — no warning on stderr."""
        from hermes_cli.main import _migrate_profile_config

        profile = mock.Mock()
        profile.name = "stale"
        profile.path = "/tmp/nonexistent-profile"

        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(1, 99),
        ), mock.patch(
            "hermes_cli.config.get_missing_env_vars",
            return_value=[],
        ), mock.patch(
            "hermes_cli.config.get_missing_config_fields",
            return_value=[],
        ), mock.patch(
            "hermes_cli.config.migrate_config"
        ) as mock_migrate, mock.patch(
            "hermes_constants.set_hermes_home_override", return_value="token"
        ), mock.patch(
            "hermes_constants.reset_hermes_home_override"
        ):
            _migrate_profile_config(profile)

        mock_migrate.assert_called_once_with(interactive=False, quiet=True)
        captured = capsys.readouterr()
        assert "⚠️" not in captured.err, "Should not warn on version-bump-only"
        assert "⚠️" not in captured.out

    def test_missing_settings_prints_actionable_warning(self, capsys):
        """When missing required settings are detected, an actionable warning
        is printed on stderr naming the profile and remediation command."""
        from hermes_cli.main import _migrate_profile_config

        profile = mock.Mock()
        profile.name = "stale"
        profile.path = "/tmp/nonexistent-profile"

        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(1, 99),
        ), mock.patch(
            "hermes_cli.config.get_missing_env_vars",
            return_value=["OPENAI_API_KEY"],
        ), mock.patch(
            "hermes_cli.config.get_missing_config_fields",
            return_value=[],
        ), mock.patch(
            "hermes_cli.config.migrate_config"
        ) as mock_migrate, mock.patch(
            "hermes_constants.set_hermes_home_override", return_value="token"
        ), mock.patch(
            "hermes_constants.reset_hermes_home_override"
        ):
            _migrate_profile_config(profile)

        mock_migrate.assert_not_called()
        captured = capsys.readouterr()
        assert "⚠️" in captured.err
        assert "stale" in captured.err
        assert "config migrate" in captured.err

    def test_migration_exception_surfaces_warning(self, capsys):
        """If migrate_config raises, the exception propagates to the caller."""
        from hermes_cli.main import _migrate_profile_config

        profile = mock.Mock()
        profile.name = "stale"
        profile.path = "/tmp/nonexistent-profile"

        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(1, 99),
        ), mock.patch(
            "hermes_cli.config.get_missing_env_vars",
            return_value=[],
        ), mock.patch(
            "hermes_cli.config.get_missing_config_fields",
            return_value=[],
        ), mock.patch(
            "hermes_cli.config.migrate_config",
            side_effect=RuntimeError("disk full"),
        ), mock.patch(
            "hermes_constants.set_hermes_home_override", return_value="token"
        ), mock.patch(
            "hermes_constants.reset_hermes_home_override"
        ):
            with pytest.raises(RuntimeError, match="disk full"):
                _migrate_profile_config(profile)

    def test_up_to_date_profile_skips_migration(self, capsys):
        """When config is already at the latest version, nothing happens."""
        from hermes_cli.main import _migrate_profile_config

        profile = mock.Mock()
        profile.name = "active"
        profile.path = "/tmp/nonexistent-profile"

        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(99, 99),
        ), mock.patch("hermes_cli.config.migrate_config") as mock_migrate, mock.patch(
            "hermes_constants.set_hermes_home_override", return_value="token"
        ), mock.patch(
            "hermes_constants.reset_hermes_home_override"
        ):
            _migrate_profile_config(profile)

        mock_migrate.assert_not_called()
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == ""


# ---------------------------------------------------------------------------
# 2. _migrate_all_profiles — update-side entry point (3 tests)
# ---------------------------------------------------------------------------


class TestMigrateAllProfiles:
    """Tests for ``hermes_cli.update_cmd._migrate_all_profiles`` — the
    helper called by ``cmd_update`` to migrate all named profiles."""

    def test_all_profiles_iterated(self, capsys):
        """Every profile returned by ``list_profiles()`` is passed to
        ``_migrate_profile_config``."""
        from hermes_cli.update_cmd import _migrate_all_profiles

        p1 = mock.Mock()
        p1.name = "profile-a"
        p2 = mock.Mock()
        p2.name = "profile-b"
        p3 = mock.Mock()
        p3.name = "profile-c"

        with mock.patch(
            "hermes_cli.profiles.list_profiles",
            return_value=[p1, p2, p3],
        ), mock.patch(
            "hermes_cli.main._migrate_profile_config"
        ) as mock_migrate:
            _migrate_all_profiles()

        assert mock_migrate.call_count == 3
        mock_migrate.assert_any_call(p1)
        mock_migrate.assert_any_call(p2)
        mock_migrate.assert_any_call(p3)
        captured = capsys.readouterr()
        assert "⚠️" not in captured.err  # no failures

    def test_migration_failure_prints_visible_stderr_warning(self, capsys):
        """When ``_migrate_profile_config`` raises for a profile, the helper
        prints a visible stderr warning (not swallowed by logger.debug)."""
        from hermes_cli.update_cmd import _migrate_all_profiles

        p1 = mock.Mock()
        p1.name = "good-profile"
        p2 = mock.Mock()
        p2.name = "bad-profile"

        with mock.patch(
            "hermes_cli.profiles.list_profiles",
            return_value=[p1, p2],
        ), mock.patch(
            "hermes_cli.main._migrate_profile_config",
            side_effect=[None, RuntimeError("permission denied")],
        ):
            _migrate_all_profiles()

        captured = capsys.readouterr()
        assert "⚠️" in captured.err
        assert "bad-profile" in captured.err
        assert "permission denied" in captured.err
        assert "config migrate" in captured.err

    def test_no_profiles_swallows_gracefully(self, capsys):
        """If ``list_profiles`` raises (module not available), the helper
        swallows it gracefully — update must not break."""
        from hermes_cli.update_cmd import _migrate_all_profiles

        with mock.patch(
            "hermes_cli.profiles.list_profiles",
            side_effect=ImportError("no profiles module"),
        ):
            _migrate_all_profiles()  # should not raise

        captured = capsys.readouterr()
        assert captured.err == ""


# ---------------------------------------------------------------------------
# 3. _migrate_active_profile_on_startup — dashboard/serve entry point (3 tests)
# ---------------------------------------------------------------------------


class TestMigrateActiveProfileOnStartup:
    """Tests for ``hermes_cli.main._migrate_active_profile_on_startup`` — the
    helper called by ``cmd_dashboard`` / ``serve`` before startup."""

    def test_stale_config_with_missing_fields_warns_on_stderr(self, capsys):
        """When the active profile's config is stale AND has missing required
        fields, a warning is printed on stderr."""
        from hermes_cli.main import _migrate_active_profile_on_startup

        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(1, 99),
        ), mock.patch(
            "hermes_cli.config.get_missing_env_vars",
            return_value=["OPENAI_API_KEY"],
        ), mock.patch(
            "hermes_cli.config.get_missing_config_fields",
            return_value=["model.provider"],
        ), mock.patch("hermes_cli.config.migrate_config") as mock_migrate:
            _migrate_active_profile_on_startup()

        mock_migrate.assert_not_called()
        captured = capsys.readouterr()
        assert "⚠️" in captured.err
        assert "v1" in captured.err
        assert "v99" in captured.err
        assert "config migrate" in captured.err

    def test_stale_config_version_bump_only_auto_migrates(self, capsys):
        """When the config is stale but only needs a version bump (no missing
        fields), migration is applied silently."""
        from hermes_cli.main import _migrate_active_profile_on_startup

        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(1, 99),
        ), mock.patch(
            "hermes_cli.config.get_missing_env_vars",
            return_value=[],
        ), mock.patch(
            "hermes_cli.config.get_missing_config_fields",
            return_value=[],
        ), mock.patch("hermes_cli.config.migrate_config") as mock_migrate:
            _migrate_active_profile_on_startup()

        mock_migrate.assert_called_once_with(interactive=False, quiet=True)
        captured = capsys.readouterr()
        assert "⚠️" not in captured.err

    def test_current_config_skips_migration(self, capsys):
        """When the config is already at the latest version, nothing happens."""
        from hermes_cli.main import _migrate_active_profile_on_startup

        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(99, 99),
        ), mock.patch("hermes_cli.config.migrate_config") as mock_migrate:
            _migrate_active_profile_on_startup()

        mock_migrate.assert_not_called()
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == ""


# ---------------------------------------------------------------------------
# 4. Wiring assertions — verify entry points call the helpers
# ---------------------------------------------------------------------------


class TestWiring:
    """Verify that production entry points invoke their respective helpers."""

    def test_cmd_update_calls_migrate_all_profiles(self):
        """``_cmd_update_impl`` calls ``_migrate_all_profiles``."""
        import inspect

        from hermes_cli.update_cmd import _cmd_update_impl, _migrate_all_profiles

        source = inspect.getsource(_cmd_update_impl)
        assert "_migrate_all_profiles()" in source, (
            "_cmd_update_impl must call _migrate_all_profiles()"
        )
        # Verify the helper exists and is callable
        assert callable(_migrate_all_profiles)

    def test_cmd_dashboard_calls_migrate_on_startup(self):
        """``cmd_dashboard`` calls ``_migrate_active_profile_on_startup``."""
        import inspect

        from hermes_cli.main import _migrate_active_profile_on_startup

        # cmd_dashboard is a large function; verify the helper is called
        # by checking the source of the module-level function
        from hermes_cli import main as main_mod

        source = inspect.getsource(main_mod.cmd_dashboard)
        assert "_migrate_active_profile_on_startup()" in source, (
            "cmd_dashboard must call _migrate_active_profile_on_startup()"
        )
        assert callable(_migrate_active_profile_on_startup)
