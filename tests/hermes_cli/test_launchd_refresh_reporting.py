"""Regression tests for launchd refresh failure reporting (issue #12882).

refresh_launchd_plist_if_needed() logged the retry failure but still returned
True and printed success. launchd_install() then unconditionally printed
'✓ Service definition updated' even when the service was not registered.

Fix:
- refresh_launchd_plist_if_needed() returns False on persistent registration
  failure (after all retries).
- launchd_install() checks the return value and prints a warning instead of
  the success message when refresh fails.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest


def _make_mock_plist_path(tmp_path, content="<plist/>"):
    p = tmp_path / "com.hermes.plist"
    p.write_text(content)
    return p


class TestRefreshLaunchdPlistReturnsOnFailure:
    """refresh_launchd_plist_if_needed() must return False when retry exhausts."""

    def test_persistent_registration_failure_returns_false(self, tmp_path, monkeypatch, capsys):
        """When _retry_launchctl_bootstrap_until_registered returns False,
        refresh_launchd_plist_if_needed must return False and NOT print success."""
        from hermes_cli import gateway as gw

        plist_path = _make_mock_plist_path(tmp_path, "<old/>")
        monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gw, "launchd_plist_is_current", lambda: False)
        monkeypatch.setattr(gw, "generate_launchd_plist", lambda: "<new/>")
        monkeypatch.setattr(gw, "_refuse_temp_home_service_write", lambda *a: False)
        monkeypatch.setattr(gw, "get_launchd_label", lambda: "com.hermes.agent")
        monkeypatch.setattr(gw, "_launchd_domain", lambda: "gui/501")
        monkeypatch.setattr(gw, "_get_restart_drain_timeout", lambda: 30.0)
        monkeypatch.setattr(gw, "_is_pid_ancestor_of_current_process", lambda p: False)
        monkeypatch.setattr(gw, "_append_launchd_reload_log", lambda msg: None)
        monkeypatch.setattr(gw, "_launchd_reload_log_path", lambda: Path("/tmp/fake.log"))

        import subprocess
        monkeypatch.setattr(gw, "_retry_launchctl_bootstrap_until_registered",
                            lambda *a, **k: False)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: MagicMock(returncode=0))

        result = gw.refresh_launchd_plist_if_needed()

        assert result is False, (
            "refresh_launchd_plist_if_needed() must return False when retry exhausts"
        )
        out = capsys.readouterr().out
        assert "Updated" not in out, (
            "Must not print success message when registration failed"
        )

    def test_successful_registration_returns_true(self, tmp_path, monkeypatch, capsys):
        """When retry succeeds, refresh_launchd_plist_if_needed must return True."""
        from hermes_cli import gateway as gw

        plist_path = _make_mock_plist_path(tmp_path, "<old/>")
        monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gw, "launchd_plist_is_current", lambda: False)
        monkeypatch.setattr(gw, "generate_launchd_plist", lambda: "<new/>")
        monkeypatch.setattr(gw, "_refuse_temp_home_service_write", lambda *a: False)
        monkeypatch.setattr(gw, "get_launchd_label", lambda: "com.hermes.agent")
        monkeypatch.setattr(gw, "_launchd_domain", lambda: "gui/501")
        monkeypatch.setattr(gw, "_get_restart_drain_timeout", lambda: 30.0)
        monkeypatch.setattr(gw, "_is_pid_ancestor_of_current_process", lambda p: False)
        monkeypatch.setattr(gw, "_append_launchd_reload_log", lambda msg: None)

        import subprocess
        monkeypatch.setattr(gw, "_retry_launchctl_bootstrap_until_registered",
                            lambda *a, **k: True)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: MagicMock(returncode=0))

        result = gw.refresh_launchd_plist_if_needed()

        assert result is True
        out = capsys.readouterr().out
        assert "Updated" in out


class TestLaunchdInstallRefreshReporting:
    """launchd_install() must surface refresh failure, not print success."""

    def test_install_repair_failure_prints_warning_not_success(self, tmp_path, monkeypatch, capsys):
        """When refresh_launchd_plist_if_needed returns False, launchd_install
        must print a warning, NOT '✓ Service definition updated'."""
        from hermes_cli import gateway as gw

        plist_path = _make_mock_plist_path(tmp_path, "<old/>")
        monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gw, "launchd_plist_is_current", lambda: False)
        monkeypatch.setattr(gw, "refresh_launchd_plist_if_needed", lambda: False)

        gw.launchd_install(force=False)

        out = capsys.readouterr().out
        assert "Service definition updated" not in out, (
            "Must not print success when refresh failed"
        )
        assert "could not be reloaded" in out or "⚠" in out, (
            "Must print a warning when refresh failed"
        )

    def test_install_repair_success_prints_success(self, tmp_path, monkeypatch, capsys):
        """When refresh_launchd_plist_if_needed returns True, launchd_install
        must print the success message."""
        from hermes_cli import gateway as gw

        plist_path = _make_mock_plist_path(tmp_path, "<old/>")
        monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gw, "launchd_plist_is_current", lambda: False)
        monkeypatch.setattr(gw, "refresh_launchd_plist_if_needed", lambda: True)

        gw.launchd_install(force=False)

        out = capsys.readouterr().out
        assert "Service definition updated" in out
