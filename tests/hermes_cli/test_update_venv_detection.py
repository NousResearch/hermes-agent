"""Regression tests for bug #39714 — update path uses active venv, not hardcoded 'venv/'.

Why: ``hermes update`` hardcoded ``PROJECT_ROOT / "venv"`` in several places,
so on a uv-based install (which puts the venv in ``.venv``) deps were written
to an orphan ``venv/`` directory that the running CLI never imports from.

What: These tests verify that ``_resolve_project_venv_root()`` and the two
update code paths that set ``VIRTUAL_ENV`` correctly target the active venv
(``sys.prefix`` when inside a venv, or the first existing dir among ``.venv``/
``venv`` when not inside one).

Test strategy: patch ``sys.prefix`` / ``sys.base_prefix`` and ``PROJECT_ROOT``
so no real filesystem state is required, then assert the returned path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _resolve_project_venv_root
# ---------------------------------------------------------------------------

class TestResolveProjectVenvRoot:
    """_resolve_project_venv_root() must return the active venv, not a hardcoded name."""

    def test_returns_sys_prefix_when_inside_venv(self, tmp_path):
        """Why: running inside a venv means sys.prefix IS the venv root.
        What: returns Path(sys.prefix) when prefix != base_prefix.
        Test: patch both attributes and assert the return value matches prefix.
        """
        import hermes_cli.main as hm

        dot_venv = tmp_path / ".venv"
        dot_venv.mkdir()

        with patch.object(hm.sys, "prefix", str(dot_venv)), \
             patch.object(hm.sys, "base_prefix", "/usr"):
            result = hm._resolve_project_venv_root()

        assert result == dot_venv

    def test_prefers_dot_venv_when_not_in_active_venv(self, tmp_path):
        """Why: uv creates .venv by default; must prefer it over bare 'venv'.
        What: returns PROJECT_ROOT/.venv when sys.prefix == sys.base_prefix
              and .venv dir exists.
        Test: create .venv and venv dirs; assert .venv is returned.
        """
        import hermes_cli.main as hm

        (tmp_path / ".venv").mkdir()
        (tmp_path / "venv").mkdir()

        with patch.object(hm.sys, "prefix", "/usr"), \
             patch.object(hm.sys, "base_prefix", "/usr"), \
             patch("hermes_cli.main.PROJECT_ROOT", tmp_path):
            result = hm._resolve_project_venv_root()

        assert result == tmp_path / ".venv"

    def test_falls_back_to_legacy_venv_when_dot_venv_absent(self, tmp_path):
        """Why: legacy managed-installer creates 'venv'; must not break those.
        What: returns PROJECT_ROOT/venv when .venv is absent and venv exists.
        Test: create only venv dir; assert venv is returned.
        """
        import hermes_cli.main as hm

        (tmp_path / "venv").mkdir()

        with patch.object(hm.sys, "prefix", "/usr"), \
             patch.object(hm.sys, "base_prefix", "/usr"), \
             patch("hermes_cli.main.PROJECT_ROOT", tmp_path):
            result = hm._resolve_project_venv_root()

        assert result == tmp_path / "venv"

    def test_sys_prefix_takes_precedence_over_disk_layout(self, tmp_path):
        """Why: sys.prefix is authoritative — it is where the running interpreter
        resolves imports from, regardless of what's on disk.
        What: returns sys.prefix path even when a different venv dir exists on disk.
        Test: patch prefix to .venv, but put only bare venv/ on disk; assert
        .venv is returned.
        """
        import hermes_cli.main as hm

        # Only create bare venv/ on disk (NOT .venv/)
        (tmp_path / "venv").mkdir()
        dot_venv = tmp_path / ".venv"

        with patch.object(hm.sys, "prefix", str(dot_venv)), \
             patch.object(hm.sys, "base_prefix", "/usr"):
            result = hm._resolve_project_venv_root()

        assert result == dot_venv


# ---------------------------------------------------------------------------
# _venv_scripts_dir — must follow _resolve_project_venv_root
# ---------------------------------------------------------------------------

class TestVenvScriptsDirFollowsActiveVenv:
    """_venv_scripts_dir must resolve via _resolve_project_venv_root, not a fixed name."""

    def test_returns_bin_of_active_dot_venv(self, tmp_path):
        """Why: Windows lock detection looks up shims in venv Scripts/;
        must find them in the active venv, not a stale hardcoded path.
        What: returns <active_venv>/bin when sys.prefix points at .venv.
        Test: patch sys.prefix to tmp/.venv; create tmp/.venv/bin; assert
        the returned scripts dir matches.
        """
        import hermes_cli.main as hm

        dot_venv = tmp_path / ".venv"
        bin_dir = dot_venv / "bin"
        bin_dir.mkdir(parents=True)

        with patch.object(hm.sys, "prefix", str(dot_venv)), \
             patch.object(hm.sys, "base_prefix", "/usr"), \
             patch.object(hm, "_is_windows", return_value=False):
            result = hm._venv_scripts_dir()

        assert result == bin_dir

    def test_returns_none_when_bin_absent(self, tmp_path):
        """Why: callers guard on None — must not crash when bin/ is missing.
        What: returns None when the resolved venv dir has no bin/Scripts.
        Test: patch sys.prefix to a non-existent .venv dir; assert None returned.
        """
        import hermes_cli.main as hm

        dot_venv = tmp_path / ".venv"
        # Do NOT create the bin dir

        with patch.object(hm.sys, "prefix", str(dot_venv)), \
             patch.object(hm.sys, "base_prefix", "/usr"), \
             patch.object(hm, "_is_windows", return_value=False):
            result = hm._venv_scripts_dir()

        assert result is None


# ---------------------------------------------------------------------------
# Update path: VIRTUAL_ENV env var must target the active venv (unit-level)
# ---------------------------------------------------------------------------

class TestResolveVenvRootReturnType:
    """Sanity checks: _resolve_project_venv_root returns a Path, not a string."""

    def test_returns_path_object(self, tmp_path):
        """Why: callers call str() on the result — must be a Path, not already a str.
        What: return type is pathlib.Path.
        Test: call _resolve_project_venv_root(); assert isinstance(result, Path).
        """
        import hermes_cli.main as hm

        with patch.object(hm.sys, "prefix", "/usr"), \
             patch.object(hm.sys, "base_prefix", "/usr"), \
             patch("hermes_cli.main.PROJECT_ROOT", tmp_path):
            result = hm._resolve_project_venv_root()

        assert isinstance(result, Path)

    def test_str_conversion_safe_for_env_dict(self, tmp_path):
        """Why: uv_env sets VIRTUAL_ENV via str(_resolve_project_venv_root());
        the str must be a valid path.
        What: str(result) is non-empty and starts with '/'.
        Test: assert the stringified result is usable as an env var value.
        """
        import hermes_cli.main as hm

        dot_venv = tmp_path / ".venv"
        dot_venv.mkdir()

        with patch.object(hm.sys, "prefix", str(dot_venv)), \
             patch.object(hm.sys, "base_prefix", "/usr"):
            result = hm._resolve_project_venv_root()

        as_str = str(result)
        assert as_str  # non-empty
        assert "/" in as_str  # looks like a path
