"""Tests for entry-point (pip-installed) plugin discovery in the CLI (#53898).

Upstream now ships ``_discover_entrypoint_plugins``; this PR keeps regression
coverage so ``hermes plugins list`` cannot regress to directory-only discovery.
"""

from unittest.mock import patch

import pytest


@pytest.fixture
def empty_dirs(tmp_path):
    """Patch directory sources to empty so only entry points show."""
    with patch(
        "hermes_cli.plugins.get_bundled_plugins_dir", return_value=tmp_path / "nope_bundled"
    ), patch(
        "hermes_cli.plugins_cmd._plugins_dir", return_value=tmp_path / "nope_user"
    ):
        yield


def test_entrypoint_plugin_discovered(empty_dirs):
    from hermes_cli.plugins_cmd import _discover_all_plugins

    with patch(
        "hermes_cli.plugins_cmd._discover_entrypoint_plugins",
        return_value=[("my-pip-plugin", "1.2.3", "A pip plugin", "pkg:plugin")],
    ):
        entries = _discover_all_plugins()

    by_key = {e[5]: e for e in entries}
    assert "my-pip-plugin" in by_key
    name, version, description, source, dir_path, key = by_key["my-pip-plugin"]
    assert name == "my-pip-plugin"
    assert version == "1.2.3"
    assert source == "entrypoint"
    assert dir_path == "pkg:plugin"
