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

import subprocess
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
        """Why: Windows Windows-lock detection looks up shims in venv Scripts/;
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
# Update path: VIRTUAL_ENV env var must target the active venv
# ---------------------------------------------------------------------------

class TestUpdateVirtualEnvTargetsActiveVenv:
    """The VIRTUAL_ENV set in the update dep-install must match the active venv.

    Why: uv refuses to install when VIRTUAL_ENV differs from the interpreter's
    prefix, and in any case writing to the wrong venv leaves the running CLI
    with stale packages.

    What: assert that the VIRTUAL_ENV value passed to
    _install_python_dependencies_with_optional_fallback() equals sys.prefix
    when we're running inside a venv.

    Test: run the update path with subprocess.run stubbed out and inspect the
    env dict that reaches the install helper.
    """

    @pytest.fixture(autouse=True)
    def _patch_managed_uv(self):
        """Stub out managed_uv calls so no network or disk I/O happens."""
        with patch("hermes_cli.managed_uv.resolve_uv", return_value=None), \
             patch("hermes_cli.managed_uv.ensure_uv", return_value=("/usr/bin/uv", False)), \
             patch("hermes_cli.managed_uv.update_managed_uv", return_value=None), \
             patch("hermes_cli.managed_uv.rebuild_venv", return_value=True):
            yield

    def _make_git_side_effect(self, commit_count="1"):
        """Build a subprocess.run side_effect for git + install commands."""

        def side_effect(cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)
            m = MagicMock()
            m.returncode = 0
            m.stdout = ""
            m.stderr = ""
            if "rev-parse" in joined and "--abbrev-ref" in joined:
                m.stdout = "main\n"
            elif "rev-parse" in joined and "--verify" in joined:
                pass
            elif "rev-list" in joined:
                m.stdout = f"{commit_count}\n"
            elif "pull" in joined:
                pass
            elif "py_compile" in joined or "validate" in joined:
                m.stdout = ""
            return m

        return side_effect

    def test_virtual_env_matches_sys_prefix_in_uv_install_path(self, tmp_path, monkeypatch):
        """Why: the git-update path must set VIRTUAL_ENV to the running
        interpreter's prefix so uv writes deps to the correct environment.
        What: capture the env dict passed to the install call and compare
        VIRTUAL_ENV to sys.prefix.
        Test: stub subprocess.run and _install_python_dependencies_with_optional_fallback;
        assert VIRTUAL_ENV == sys.prefix.
        """
        from contextlib import ExitStack
        import hermes_cli.main as hm

        captured_env: list[dict] = []

        def _fake_install(cmd_prefix, *, env=None, group="all"):
            if env is not None:
                captured_env.append(dict(env))

        active_venv = str(tmp_path / ".venv")
        monkeypatch.setattr(hm.sys, "prefix", active_venv)
        monkeypatch.setattr(hm.sys, "base_prefix", "/usr")
        monkeypatch.setenv("VIRTUAL_ENV", active_venv)

        args = SimpleNamespace()

        patches = [
            patch("hermes_cli.main.subprocess.run", side_effect=self._make_git_side_effect("1")),
            patch.object(hm, "_install_python_dependencies_with_optional_fallback", side_effect=_fake_install),
            patch.object(hm, "_is_termux_env", return_value=False),
            patch.object(hm, "_update_node_dependencies"),
            patch.object(hm, "_build_web_ui"),
            patch.object(hm, "_ensure_fhs_path_guard"),
            patch.object(hm, "_print_curator_first_run_notice"),
            patch.object(hm, "_wait_for_interpreter_venv_ready", return_value=True),
            patch.object(hm, "_run_pre_update_backup"),
            patch.object(hm, "_validate_critical_files_syntax", return_value=(True, None, None)),
            patch.object(hm, "_clear_bytecode_cache", return_value=0),
            patch.object(hm, "_invalidate_update_cache"),
            patch.object(hm, "_discard_lockfile_churn"),
            patch.object(hm, "_get_origin_url", return_value="https://github.com/NousResearch/hermes-agent.git"),
            patch.object(hm, "_is_fork", return_value=False),
            patch.object(hm, "_resolve_update_branch", return_value="main"),
            patch.object(hm, "_stash_local_changes_if_needed", return_value=None),
            patch.object(hm, "_clean_managed_worktree", return_value=True),
            patch.object(hm, "_capture_head_sha", return_value="abc123"),
            patch.object(hm, "_install_hangup_protection", return_value=None),
            patch.object(hm, "_finalize_update_output"),
            patch("hermes_cli.config.detect_install_method", return_value="git"),
            patch("hermes_cli.config.is_managed", return_value=False),
            patch("hermes_cli.config.format_docker_update_message", return_value=""),
        ]

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from hermes_cli.main import cmd_update
            cmd_update(args)

        assert captured_env, "install helper was never called — test setup issue"
        for env in captured_env:
            assert env.get("VIRTUAL_ENV") == active_venv, (
                f"VIRTUAL_ENV was {env.get('VIRTUAL_ENV')!r}, "
                f"expected {active_venv!r} (sys.prefix)"
            )
