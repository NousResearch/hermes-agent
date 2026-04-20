"""Regression tests for refresh_launchd_plist_if_needed() (issue #12866).

Before the fix, launchctl bootstrap failures were silently swallowed and
the function always returned True with a success message.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import gateway as g


def _make_fake_run(bootout_rc=0, bootstrap_rc=0):
    """Return a fake subprocess.run that records calls and returns given codes."""
    calls = []

    def fake_run(cmd, check=False, timeout=None):
        calls.append({"cmd": cmd, "check": check, "returncode": bootstrap_rc if "bootstrap" in cmd else bootout_rc})
        rc = bootstrap_rc if "bootstrap" in cmd else bootout_rc
        return SimpleNamespace(returncode=rc)

    return fake_run, calls


def _patch_context(plist, fake_run):
    return (
        patch.object(g, "get_launchd_plist_path", return_value=plist),
        patch.object(g, "launchd_plist_is_current", return_value=False),
        patch.object(g, "generate_launchd_plist", return_value="new-plist-content"),
        patch.object(g, "get_launchd_label", return_value="com.hermes.agent"),
        patch.object(g, "_launchd_domain", return_value="gui/501"),
        patch("subprocess.run", side_effect=fake_run),
    )


class TestRefreshLaunchdPlistIfNeeded:
    def test_bootstrap_success_returns_true(self, tmp_path):
        plist = tmp_path / "com.hermes.agent.plist"
        plist.write_text("old", encoding="utf-8")
        fake_run, calls = _make_fake_run(bootout_rc=0, bootstrap_rc=0)

        printed = []
        patches = _patch_context(plist, fake_run)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patch("builtins.print", side_effect=lambda *a, **k: printed.append(" ".join(str(x) for x in a))):
            result = g.refresh_launchd_plist_if_needed()

        assert result is True
        assert any("Updated" in m for m in printed)
        assert plist.read_text() == "new-plist-content"

    def test_bootstrap_failure_returns_false(self, tmp_path):
        """Core regression: bootstrap rc=5 must return False, not True."""
        plist = tmp_path / "com.hermes.agent.plist"
        plist.write_text("old", encoding="utf-8")
        fake_run, calls = _make_fake_run(bootout_rc=0, bootstrap_rc=5)

        printed = []
        patches = _patch_context(plist, fake_run)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patch("builtins.print", side_effect=lambda *a, **k: printed.append(" ".join(str(x) for x in a))):
            result = g.refresh_launchd_plist_if_needed()

        assert result is False, "bootstrap failure must return False"
        assert not any("Updated" in m for m in printed), "must not print success on failure"
        assert any("Failed" in m or "failed" in m for m in printed)

    def test_bootout_failure_is_tolerated(self, tmp_path):
        """bootout non-zero is harmless (job may not have been loaded)."""
        plist = tmp_path / "com.hermes.agent.plist"
        plist.write_text("old", encoding="utf-8")
        fake_run, calls = _make_fake_run(bootout_rc=3, bootstrap_rc=0)

        patches = _patch_context(plist, fake_run)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patch("builtins.print"):
            result = g.refresh_launchd_plist_if_needed()

        assert result is True

    def test_plist_already_current_returns_false(self, tmp_path):
        plist = tmp_path / "com.hermes.agent.plist"
        plist.write_text("current", encoding="utf-8")

        with patch.object(g, "get_launchd_plist_path", return_value=plist), \
             patch.object(g, "launchd_plist_is_current", return_value=True):
            result = g.refresh_launchd_plist_if_needed()

        assert result is False

    def test_plist_missing_returns_false(self, tmp_path):
        plist = tmp_path / "nonexistent.plist"

        with patch.object(g, "get_launchd_plist_path", return_value=plist), \
             patch.object(g, "launchd_plist_is_current", return_value=False):
            result = g.refresh_launchd_plist_if_needed()

        assert result is False

    def test_bootstrap_called_with_check_false(self, tmp_path):
        """bootstrap must use check=False so we can inspect the returncode."""
        plist = tmp_path / "com.hermes.agent.plist"
        plist.write_text("old", encoding="utf-8")
        fake_run, calls = _make_fake_run(bootstrap_rc=0)

        patches = _patch_context(plist, fake_run)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patch("builtins.print"):
            g.refresh_launchd_plist_if_needed()

        bootstrap_calls = [c for c in calls if "bootstrap" in c["cmd"]]
        assert bootstrap_calls, "bootstrap must be called"
        assert all(c["check"] is False for c in bootstrap_calls)
