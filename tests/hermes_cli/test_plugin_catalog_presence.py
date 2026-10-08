"""A catalog entry's app state comes from the pinned plugin.json declaration."""

import pytest

from hermes_cli import plugin_catalog as pc
from hermes_cli import plugin_catalog_presence as presence_mod

SHA = "0" * 40
NEEDS_APP = {"extensions": {"com.nousresearch.hermes": {"servers": {"srv": {
    "app": {"darwin": {"presence": "executable", "location": "/nonexistent/fx-app"},
            "linux": {"presence": "executable", "location": "/nonexistent/fx-app"},
            "win32": {"presence": "executable", "location": "C:/nonexistent/fx-app.exe"}},
    "requires": {"app": True}}}}}}


def _entry(name, *, title=""):
    return pc.entry_from_mapping({"name": name, "repo": f"https://github.com/fx/{name}", "sha": SHA,
                                  "description": f"{name} does things. More.", "maintainer": "fx",
                                  "tier": "official", "category": "tools", "title": title}, name)


@pytest.fixture
def manifests(monkeypatch):
    found = {"everywhere": NEEDS_APP, "here-only": {"name": "here-only"}}
    monkeypatch.setattr(presence_mod, "_pinned_manifest", lambda repo, sha, subdir: found.get(repo.rsplit("/", 1)[-1]))


def test_app_state_comes_from_the_pinned_declaration(manifests):
    missing = presence_mod.presence(_entry("everywhere", title="Everywhere App"))
    undeclared = presence_mod.presence(_entry("here-only"))
    # A declared app that is absent carries a reason; no declaration is unknown, never "present".
    assert missing.state == "missing_app" and "Everywhere App" in missing.sentence
    assert undeclared.state == "unknown" and undeclared.sentence == ""


def test_a_catalog_onboarding_key_is_ignored():
    entry = pc.entry_from_mapping({"name": "old", "repo": "https://github.com/fx/old", "sha": SHA,
                                   "maintainer": "fx", "onboarding": True}, "old")
    assert entry is not None and "onboarding" not in entry.to_dict()
