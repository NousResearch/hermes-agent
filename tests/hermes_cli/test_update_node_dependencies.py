"""Tests for _update_node_dependencies — single-pass npm install (#64354).

The updater used to run two passes (root-only, then workspace-scoped).
Both went through _run_npm_install_deterministic, which prefers ``npm ci``;
``npm ci`` deletes node_modules before reifying the requested tree, so the
second pass wiped the root-only deps (agent-browser, @streamdown) installed
by the first while still exiting 0. The fix collapses the install into one
invocation using --include-workspace-root.
"""

import subprocess
from unittest.mock import patch

import pytest

from hermes_cli.main import _update_node_dependencies


@pytest.fixture
def project_root(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    return tmp_path


def _run_with_mocked_install(project_root, returncode=0, stderr=""):
    result = subprocess.CompletedProcess([], returncode, stdout="", stderr=stderr)
    with patch("hermes_cli.main.PROJECT_ROOT", project_root), \
         patch("hermes_constants.find_node_executable", return_value="/usr/bin/npm"), \
         patch("hermes_constants.with_hermes_node_path", side_effect=lambda env=None: env or {}), \
         patch("hermes_cli.main._nixos_build_env", return_value={}), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=result) as mock_install:
        _update_node_dependencies()
    return mock_install


def test_installs_in_a_single_pass(project_root):
    """One npm invocation — a second scoped pass would wipe root deps under npm ci."""
    mock_install = _run_with_mocked_install(project_root)
    assert mock_install.call_count == 1


def test_single_pass_includes_workspace_root_and_selected_workspaces(project_root):
    """Root deps and ui-tui/web must be reified in the same tree; desktop skipped."""
    mock_install = _run_with_mocked_install(project_root)
    extra_args = mock_install.call_args.kwargs["extra_args"]

    assert "--include-workspace-root" in extra_args

    workspaces = [
        extra_args[i + 1]
        for i, arg in enumerate(extra_args[:-1])
        if arg == "--workspace"
    ]
    assert workspaces == ["ui-tui", "web"]

    # The old root-only pass must not come back: scoped npm ci prunes what
    # a --workspaces=false pass installed.
    assert "--workspaces=false" not in extra_args


def test_failure_is_reported(project_root, capsys):
    _run_with_mocked_install(project_root, returncode=1, stderr="npm ERR! boom")
    out = capsys.readouterr().out
    assert "⚠" in out


def test_success_message_mentions_root_and_workspaces(project_root, capsys):
    _run_with_mocked_install(project_root)
    out = capsys.readouterr().out
    assert "repo root + ui-tui, web workspaces" in out


def test_skips_when_npm_missing(project_root):
    with patch("hermes_cli.main.PROJECT_ROOT", project_root), \
         patch("hermes_constants.find_node_executable", return_value=None), \
         patch("hermes_cli.main._run_npm_install_deterministic") as mock_install:
        _update_node_dependencies()
    mock_install.assert_not_called()


def test_skips_without_package_json(tmp_path):
    with patch("hermes_cli.main.PROJECT_ROOT", tmp_path), \
         patch("hermes_constants.find_node_executable", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic") as mock_install:
        _update_node_dependencies()
    mock_install.assert_not_called()
