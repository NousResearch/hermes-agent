"""A plugin Hermes installed whose files were later deleted is visible and removable (deepagents#7278 port).

Before: ``plugins list`` omitted it and ``remove`` answered "No plugin named", while ``plugins.enabled`` and
the installer record still named it.
"""

from __future__ import annotations

import json

import pytest
import hermes_yaml as yaml

from hermes_cli import plugins_cmd
from tui_gateway import server

_SOURCE = "https://github.com/example/plugins#gone"


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    plugins = hermes_home / "plugins"
    (plugins / "present").mkdir(parents=True)
    (plugins / ".install-metadata.json").write_text(json.dumps({
        "gone": {"source": _SOURCE}, "present": {"source": "https://github.com/example/present"},
    }), encoding="utf-8")
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": ["gone", "present", "keep-me"], "disabled": []}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def test_list_shows_an_enabled_plugin_whose_files_are_gone(home, capsys):
    plugins_cmd.cmd_list(type("A", (), {"json": True, "no_bundled": True})())
    rows = {row["name"]: row for row in json.loads(capsys.readouterr().out)}

    assert rows["gone"]["status"] == "missing" and rows["gone"]["source"] == _SOURCE
    assert rows.get("present", {}).get("status") != "missing"


def test_remove_forgets_the_record_and_enable_of_a_plugin_whose_files_are_gone(home):
    resp = server.handle_request({"id": "1", "method": "plugins.manage",
                                  "params": {"action": "remove", "name": "gone"}})

    assert resp["result"]["ok"] is True
    metadata = json.loads((home / "plugins" / ".install-metadata.json").read_text(encoding="utf-8-sig"))
    assert set(metadata) == {"present"}
    enabled = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8-sig"))["plugins"]["enabled"]
    assert enabled == ["keep-me", "present"]
