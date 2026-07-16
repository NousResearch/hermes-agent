"""Tests for the ``pre_plugin_install`` gate hook in ``_install_plugin_core``.

A registered ``pre_plugin_install`` callback can block an install by
returning a reason (str, or list/tuple of strs). The install must abort
fail-closed with ``PluginOperationError`` before the clone is promoted
(``shutil.move``) into ``~/.hermes/plugins``. With no callback registered
(the default), ``invoke_hook`` returns ``[]`` and the install proceeds
unaffected.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import plugins_cmd


@patch("hermes_cli.plugins_cmd.shutil.move")
@patch("hermes_cli.plugins_cmd._plugins_dir")
@patch("hermes_cli.plugins_cmd._read_manifest")
@patch("hermes_cli.plugins_cmd.subprocess.run")
def test_pre_plugin_install_block_aborts(
    mock_run, mock_read_manifest, mock_plugins_dir, mock_move, tmp_path, monkeypatch
):
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    mock_plugins_dir.return_value = plugins_dir
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    mock_read_manifest.return_value = {"name": "evil"}

    monkeypatch.setattr(
        plugins_cmd,
        "invoke_hook",
        lambda hook_name, **kw: (
            [["blocked by test"]] if hook_name == "pre_plugin_install" else []
        ),
    )

    with pytest.raises(plugins_cmd.PluginOperationError, match="blocked by test"):
        plugins_cmd._install_plugin_core("owner/repo", force=False)

    mock_move.assert_not_called()


@patch("hermes_cli.plugins_cmd._copy_example_files")
@patch("hermes_cli.plugins_cmd.shutil.move")
@patch("hermes_cli.plugins_cmd._plugins_dir")
@patch("hermes_cli.plugins_cmd._read_manifest")
@patch("hermes_cli.plugins_cmd.subprocess.run")
def test_no_callback_registered_install_proceeds(
    mock_run,
    mock_read_manifest,
    mock_plugins_dir,
    mock_move,
    mock_copy_example_files,
    tmp_path,
    monkeypatch,
):
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    mock_plugins_dir.return_value = plugins_dir
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    mock_read_manifest.return_value = {"name": "well-behaved"}

    monkeypatch.setattr(plugins_cmd, "invoke_hook", lambda hook_name, **kw: [])

    target, manifest, name = plugins_cmd._install_plugin_core("owner/repo", force=False)

    mock_move.assert_called_once()
    assert name == "well-behaved"
