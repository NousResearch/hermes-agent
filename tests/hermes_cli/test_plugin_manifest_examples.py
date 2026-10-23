"""Example configuration must not replace the manifest selected for review."""

from unittest.mock import Mock

import pytest

from hermes_cli.plugins_cmd import _copy_example_files, _read_manifest


@pytest.mark.parametrize("example_name", ["plugin.yaml.example", "plugin.yml.example", "plugin.json.example"])
def test_copy_examples_does_not_create_plugin_manifests(tmp_path, example_name):
    (tmp_path / "plugin.yml").write_text("name: sample\nversion: 1.0.0\nprovides_tools: []\n")
    (tmp_path / example_name).write_text(
        "name: sample\nversion: 1.0.0\nprovides_tools: [unreviewed_marker]\n")
    (tmp_path / "config.yaml.example").write_text("label: sample\n")
    reviewed = _read_manifest(tmp_path)
    _copy_example_files(tmp_path, Mock())
    assert _read_manifest(tmp_path) == reviewed
    assert not (tmp_path / "plugin.yaml").exists()
    assert not (tmp_path / "plugin.json").exists()
    assert (tmp_path / "config.yaml").read_text() == "label: sample\n"
    # Existing user configuration still takes precedence on later preparation.
    (tmp_path / "config.yaml").write_text("label: customized\n")
    _copy_example_files(tmp_path, Mock())
    assert (tmp_path / "config.yaml").read_text() == "label: customized\n"
