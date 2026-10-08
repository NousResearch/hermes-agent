"""A relocated plugins directory (symlink or Windows junction) still publishes.

``StagedPlugin`` judged containment against the home root, so a plugins
directory relocated by symlink made every publication fail its own escape
check: the resolved target left the home tree. Containment is now judged
against the resolved plugins root — wherever that directory really lives —
while a target inside another home's plugins directory is still refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pm.publication import StagedPlugin


def _staged_plugin(staged: Path, target: Path) -> StagedPlugin:
    return StagedPlugin({
        "staged": str(staged),
        "target": str(target),
        "target_digest": None,
        "old_metadata": {},
        "new_metadata": {"example": {"revision": "a" * 40}},
    })


def test_a_relocated_plugins_directory_publishes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    relocated = tmp_path / "plugins-real"
    relocated.mkdir()
    (home / "plugins").symlink_to(relocated, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "plugin.yaml").write_text("name: example\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()

    _staged_plugin(staged, home / "plugins" / "example").publish(project)

    assert (relocated / "example" / "plugin.yaml").read_text(
        encoding="utf-8"
    ) == "name: example\n"
    assert not (home / "plugins" / "example").is_symlink()
    metadata = json.loads(
        (home / "plugins" / ".install-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata == {"example": {"revision": "a" * 40}}


def test_a_target_in_a_foreign_plugins_directory_is_still_refused(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    (home / "plugins").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    other = tmp_path / "other-home"
    (other / "plugins").mkdir(parents=True)
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "plugin.yaml").write_text("name: example\n", encoding="utf-8")

    with pytest.raises(ValueError, match="escape or overlap"):
        _staged_plugin(staged, other / "plugins" / "example")
    assert not (other / "plugins" / "example").exists()
    assert staged.is_dir()
