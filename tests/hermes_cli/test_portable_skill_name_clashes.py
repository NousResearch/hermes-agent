"""Portable plugins whose names slug to the same skill prefix (``foo.bar`` and ``foo-bar``).

Skills take the plugin's slugged name, so two such plugins claim one qualified name. Only one may
own it, a saved hashed name must reach only its own plugin's skill, and disabling one must not
block the other. Real plugin discovery against a temp ``HERMES_HOME``.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins_manifest import _portable_data_namespace
from tools.skills_tool import skill_view


def _home(tmp_path, monkeypatch, *, enabled, disabled=()):
    home = tmp_path / ".hermes"
    (home / "plugins").mkdir(parents=True)
    (home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": list(enabled), "disabled": list(disabled)}}))
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    from hermes_cli.agent_plugins import PLUGIN_SCHEMA_V1
    for name in ("foo.bar", "foo-bar"):
        skill = home / "plugins" / name / "skills" / "workflow"
        skill.mkdir(parents=True)
        (home / "plugins" / name / "plugin.json").write_text(json.dumps({"$schema": PLUGIN_SCHEMA_V1, "name": name}))
        (skill / "SKILL.md").write_text(f"---\nname: workflow\ndescription: {name} workflow.\n---\nBody of {name}.\n")
    plugins_mod._reset_plugin_managers_for_tests()
    plugins_mod.get_plugin_manager().discover_and_load(force=True)


@pytest.fixture(autouse=True)
def _reset_managers():
    yield
    plugins_mod._reset_plugin_managers_for_tests()


def test_a_clashing_portable_skill_has_one_owner_and_old_names_reach_only_their_own(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, enabled=["foo.bar", "foo-bar"])
    owner = plugins_mod.get_plugin_manager()._plugin_skills["foo-bar:workflow"]["plugin_key"]
    loser = ({"foo.bar", "foo-bar"} - {owner}).pop()

    current = json.loads(skill_view("foo-bar:workflow"))
    assert current["success"] is True and f"Body of {owner}." in current["content"]
    assert json.loads(skill_view(f"{_portable_data_namespace(owner)}:workflow"))["success"] is True
    lost = json.loads(skill_view(f"{_portable_data_namespace(loser)}:workflow"))
    assert lost["success"] is False


def test_disabling_one_plugin_does_not_block_the_other_with_the_same_prefix(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, enabled=["foo.bar"], disabled=["foo-bar"])

    result = json.loads(skill_view("foo-bar:workflow"))

    assert result["success"] is True
    assert "Body of foo.bar." in result["content"]
