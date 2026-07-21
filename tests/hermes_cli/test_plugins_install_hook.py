"""Tests for the ``pre_plugin_install`` gate hook in ``_install_plugin_core``.

A registered ``pre_plugin_install`` callback can block an install by
returning a reason (str, or list/tuple of strs). The install must abort
fail-closed with ``PluginOperationError`` before the clone is promoted
(``shutil.move``) into ``~/.hermes/plugins``. With no callback registered
(the default), the gate returns no reasons and the install proceeds
unaffected.

The gate goes through ``collect_hook_block_reasons``, which loads the
trusted plugin registry first — ``plugins install`` is a built-in command
that skips startup discovery, so without that load the gate would fire
against an empty registry (the bug flagged in PR review).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.plugins as plugins_mod
from hermes_cli import plugins_cmd


def _patch_gate(monkeypatch, returns):
    """Patch plugins-level discovery + hook invocation for unit tests."""
    monkeypatch.setattr(plugins_mod, "discover_plugins", lambda force=False: None)
    monkeypatch.setattr(
        plugins_mod,
        "invoke_hook",
        lambda hook_name, **kw: returns if hook_name == "pre_plugin_install" else [],
    )


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

    _patch_gate(monkeypatch, [["blocked by test"]])

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

    _patch_gate(monkeypatch, [])

    target, manifest, name = plugins_cmd._install_plugin_core("owner/repo", force=False)

    mock_move.assert_called_once()
    assert name == "well-behaved"


@patch("hermes_cli.plugins_cmd.shutil.move")
@patch("hermes_cli.plugins_cmd._plugins_dir")
@patch("hermes_cli.plugins_cmd._read_manifest")
@patch("hermes_cli.plugins_cmd.subprocess.run")
def test_real_temp_plugin_gate_blocks_install(
    mock_run, mock_read_manifest, mock_plugins_dir, mock_move, tmp_path, monkeypatch
):
    """End-to-end: a REAL plugin in a temp HERMES_HOME, loaded by the normal
    discovery sweep (no invoke_hook patching), blocks the install.

    Regression for the empty-registry bug: ``plugins install`` never ran
    ``discover_plugins()``, so a registered gate plugin was invisible to the
    hook and the gate silently allowed everything.
    """
    home = tmp_path / "home"
    gate_dir = home / "plugins" / "gatekeeper"
    gate_dir.mkdir(parents=True)
    (gate_dir / "plugin.yaml").write_text(
        "name: gatekeeper\nversion: 1.0.0\nhooks:\n  - pre_plugin_install\n"
    )
    (gate_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_hook(\n"
        "        'pre_plugin_install',\n"
        "        lambda **kw: ['blocked by gatekeeper: ' + kw.get('name', '?')],\n"
        "    )\n"
    )
    (home / "config.yaml").write_text("plugins:\n  enabled:\n    - gatekeeper\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Keep the sweep minimal: no bundled plugins, fresh manager instance.
    empty = tmp_path / "no-bundled"
    empty.mkdir()
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty)
    monkeypatch.setattr(plugins_mod, "_plugin_manager", None)

    plugins_dir = tmp_path / "install-target"
    plugins_dir.mkdir()
    mock_plugins_dir.return_value = plugins_dir
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    mock_read_manifest.return_value = {"name": "evil"}

    with pytest.raises(
        plugins_cmd.PluginOperationError, match="blocked by gatekeeper: evil"
    ):
        plugins_cmd._install_plugin_core("owner/repo", force=False)

    mock_move.assert_not_called()
