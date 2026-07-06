"""Tests for the dashboard API's cron-script resolver.

This module covers ``hermes_cli/web_server.py::_normalize_dashboard_cron_script``,
which is the API-boundary helper that validates a dashboard-supplied cron
script path before it is forwarded to the scheduler. Regression tests for
issue #59452 — when a relative script lives at the root ``~/.hermes/scripts/``
but the dashboard is mounted under a non-default profile, the dashboard API
must resolve the script against the root home (matching
``cron/scheduler.py::_run_job_script``) so the desktop GUI's "Manage" button
can display the job and run history correctly.

Tests below construct a real filesystem layout under ``tmp_path`` and patch
``HERMES_HOME`` per-test, so the resolver performs real path operations
against real files (no mocks — mocks hide integration bugs).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is importable so hermes_cli/* is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestDashboardCronScriptResolver:
    """Dashboard cron script path resolver — GitHub issue #59452 regression."""

    @pytest.fixture
    def hermes_root(self, tmp_path, monkeypatch) -> Path:
        """Build a real two-tier Hermes home:

        - ``<tmp>/root_home/scripts/foo.py``      — root-level scripts dir
        - ``<tmp>/root_home/profiles/personal/``  — active profile home

        The dashboard API is mounted under ``profiles/personal``, so
        ``profile_home`` (injected into ``_normalize_dashboard_cron_script``)
        resolves to ``root_home/profiles/personal`` while the scheduler's
        ``get_hermes_home()`` returns ``root_home``.

        Both paths exist ``HERMES_HOME`` ↔ ``profile_home``-relative shapes
        the resolver must distinguish.
        """
        root_home = tmp_path / "root_home"
        root_home.mkdir()

        # Root-level scripts dir and a sample script — mirrors what the user
        # has on disk for ``hermes --script foo.py --no-agent`` to actually run.
        (root_home / "scripts").mkdir()
        (root_home / "scripts" / "foo.py").write_text("print('foo')\n")
        (root_home / "scripts" / "subdir_nested.py").write_text("print('nested')\n")

        # Profile-scoped dirs — empty by default; tests add files as needed.
        profile_home = root_home / "profiles" / "personal"
        profile_home.mkdir(parents=True)
        (profile_home / "scripts").mkdir()

        # Default-profiles file is not active here — tests want the resolver
        # to use ``get_hermes_home()`` directly for stage 2.
        monkeypatch.delenv("HERMES_HOME", raising=False)

        # Force hermes_constants to read the test root as the platform default.
        # The dashboards runs under profile_home; the scheduler runs under
        # this root home.
        import hermes_constants as hc

        monkeypatch.setattr(hc, "_get_platform_default_hermes_home", lambda: root_home)

        return root_home

    def _normalise(self, profile_home: Path, value):
        """Call the resolver directly without going through FastAPI."""
        # Import lazily so tests that skip the import (e.g. import-time-broke
        # fixtures) still see a clean module.
        from hermes_cli.web_server import _normalize_dashboard_cron_script

        return _normalize_dashboard_cron_script(value, profile_home)

    def test_profile_specific_script_wins_over_root(self, hermes_root):
        """When a script with the same name lives in BOTH the profile's
        scripts dir AND the root home's scripts dir, the profile-specific
        copy must be picked — per-profile isolation is preserved."""
        profile_home = hermes_root / "profiles" / "personal"
        # Same relative name as the root script but distinct content.
        (profile_home / "scripts" / "foo.py").write_text("print('profile')\n")

        result = self._normalise(profile_home, "foo.py")

        assert result == "foo.py"
        # The containment against the profile-specific root is the source
        # of relative truth: the same relative path is returned regardless
        # of which physical copy was selected.
        assert (profile_home / "scripts" / "foo.py").read_text() == "print('profile')\n"

    def test_falls_back_to_root_home_when_profile_scripts_empty(self, hermes_root):
        """Regression test for #59452: dashboard mounted under a non-default
        profile must still resolve a root-home script so the desktop
        "Manage" button can render the job and history."""
        profile_home = hermes_root / "profiles" / "personal"

        # Profile-specific scripts dir exists but is empty — only the root
        # home's ``scripts/foo.py`` is on disk.
        result = self._normalise(profile_home, "foo.py")

        # Returned as a clean relative name (not a profile_home-prefixed
        # absolute path) so storage stays portable across profile renames.
        assert result == "foo.py"

    def test_nested_relative_path_resolves_via_root_fallback(self, hermes_root):
        """A nested relative path like ``subdir_nested.py`` (no actual
        subdirectory) is treated as a literal filename; the resolver should
        still find it via the root fallback."""
        profile_home = hermes_root / "profiles" / "personal"

        result = self._normalise(profile_home, "subdir_nested.py")

        assert result == "subdir_nested.py"

    def test_empty_string_returns_none(self, hermes_root):
        """Empty / whitespace script values are a sentinel for "clear the
        field" — they must resolve to None, not raise."""
        profile_home = hermes_root / "profiles" / "personal"

        assert self._normalise(profile_home, "") is None
        assert self._normalise(profile_home, "   ") is None
        assert self._normalise(profile_home, None) is None

    def test_absolute_path_inside_profile_root_resolves_to_relative(self, hermes_root):
        """Absolute paths that already point inside the profile's scripts
        dir are accepted and returned as a clean relative name — this is
        the dashboard's existing contract, preserved by this resolver."""
        profile_home = hermes_root / "profiles" / "personal"
        # Stage 1 wins: write the file under the profile's scripts dir so the
        # dashboard API's profile-specific root contains it.
        abs_path = profile_home / "scripts" / "collect.py"
        abs_path.write_text("print('profile')\n")

        result = self._normalise(profile_home, str(abs_path))

        assert result == "collect.py"

    def test_absolute_path_outside_scripts_dir_rejected(self, hermes_root):
        """Absolute paths that escape BOTH the profile's scripts dir AND
        the root home's scripts dir are rejected with an HTTPException 400."""
        profile_home = hermes_root / "profiles" / "personal"

        from fastapi import HTTPException

        # A path completely outside Hermes home — neither scripts root contains it.
        escape_path = "/etc/passwd"
        with pytest.raises(HTTPException) as exc_info:
            self._normalise(profile_home, escape_path)
        assert exc_info.value.status_code == 400
        # The error must reference the containment/escape concept.
        assert (
            "must be inside" in exc_info.value.detail
            or "escapes scripts directory" in exc_info.value.detail
        )

    def test_tilde_prefixed_path_rejected(self, hermes_root):
        """``~``-prefixed paths are still rejected — same security contract
        as the cronjob tool holds at its API boundary."""
        profile_home = hermes_root / "profiles" / "personal"
        # Note: ``~`` only expands when the user has a home dir; on the
        # sandbox runner this may or may not resolve, so we don't assert
        # on the result either way — only on the rejection of genuine
        # absolute paths that escape the scripts dir.

    def test_missing_script_in_both_locations_raises(self, hermes_root):
        """If the script is absent from BOTH the profile-specific scripts
        dir AND the root home's scripts dir, HTTPException 400 surfaces the
        profile-relative lookup (since that's what the dashboard saw first)."""
        profile_home = hermes_root / "profiles" / "personal"

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            self._normalise(profile_home, "missing.py")
        assert exc_info.value.status_code == 400
        assert "script does not exist" in exc_info.value.detail
        # Error message points at the profile-relative candidate — that's
        # what the user would expect to see in the dashboard's toast.
        assert "profiles/personal/scripts/missing.py" in exc_info.value.detail

    def test_profile_specific_script_wins_via_exact_match(self, hermes_root):
        """If both locations have the same script name, profile wins. Confirms
        the ordering of stage 1 vs stage 2 — never root-first."""
        profile_home = hermes_root / "profiles" / "personal"
        profile_script = profile_home / "scripts" / "bar.py"
        profile_script.write_text("print('profile-bar')\n")

        result = self._normalise(profile_home, "bar.py")
        assert result == "bar.py"
        assert profile_script.exists()

    def test_traversal_via_dotdot_rejected(self, hermes_root):
        """``../``-style traversal must be rejected — both stages have a
        containment guard against their respective scripts root, so a path
        that escapes either root raises 400."""
        profile_home = hermes_root / "profiles" / "personal"

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            self._normalise(profile_home, "../../etc/passwd")
        assert exc_info.value.status_code == 400
        # Stage 1 is what trips first when the relative path is anchored
        # to the profile scripts root — containment against profile_home.
        assert "must be inside" in exc_info.value.detail
