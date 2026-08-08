"""Tests for _update_node_dependencies — single-pass npm install (#64354/#43564).

The updater used to run two passes (root-only, then workspace-scoped).
Both went through _run_npm_install_deterministic, which prefers ``npm ci``;
``npm ci`` deletes node_modules before reifying the requested tree, so the
second pass wiped the root-only deps (agent-browser, @streamdown) installed
by the first while still exiting 0. The fix collapses the install into one
invocation using --include-workspace-root.

Patches the hermes_cli.main surface (the historical test surface reachable
via update_cmd._m()).
"""

import json
import subprocess
from unittest.mock import patch

import pytest

from hermes_cli.update_cmd import _root_declared_deps_present


@pytest.fixture
def project_root(tmp_path):
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    return tmp_path


def _run_with_mocked_install(project_root):
    result = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    with patch("hermes_cli.main.PROJECT_ROOT", project_root), \
         patch("hermes_cli.update_cmd._record_npm_lockfile_hash", lambda *a, **k: None), \
         patch("hermes_cli.update_cmd._npm_lockfile_changed", lambda *a, **k: True), \
         patch("hermes_cli.main._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._nixos_build_env", return_value={}), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=result) as mock_install:
        from hermes_cli.update_cmd import _update_node_dependencies

        failed = _update_node_dependencies()
    return failed, mock_install


def test_installs_in_a_single_pass(project_root):
    """One npm invocation — a second scoped pass would wipe root deps under npm ci."""
    failed, mock_install = _run_with_mocked_install(project_root)
    assert failed == []
    assert mock_install.call_count == 1


def test_single_pass_uses_include_workspace_root(project_root):
    """The one pass must keep the root's own deps via --include-workspace-root."""
    failed, mock_install = _run_with_mocked_install(project_root)
    assert failed == []
    _, kwargs = mock_install.call_args
    extra = kwargs.get("extra_args") or ()
    assert "--include-workspace-root" in extra
    assert "--workspace" in extra


def test_root_declared_deps_present_true_when_installed(project_root):
    (project_root / "package.json").write_text(
        json.dumps({"dependencies": {"agent-browser": "^1.0"}}), encoding="utf-8"
    )
    (project_root / "node_modules" / "agent-browser").mkdir()
    with patch("hermes_cli.main.PROJECT_ROOT", project_root):
        assert _root_declared_deps_present() is True


def test_root_declared_deps_present_false_when_pruned(project_root):
    """The pruned-tree case: dep declared but node_modules entry missing."""
    (project_root / "package.json").write_text(
        json.dumps({"dependencies": {"agent-browser": "^1.0"}}), encoding="utf-8"
    )
    with patch("hermes_cli.main.PROJECT_ROOT", project_root):
        assert _root_declared_deps_present() is False


def test_root_declared_deps_present_noop_without_manifest(project_root):
    (project_root / "package.json").unlink()
    with patch("hermes_cli.main.PROJECT_ROOT", project_root):
        assert _root_declared_deps_present() is True
