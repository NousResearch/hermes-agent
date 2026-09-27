"""The onboarding card's catalog plugins: curated flag, platform filter, and the pinned app declaration."""

import pytest

from hermes_cli import plugin_catalog as pc
from hermes_cli import plugin_catalog_presence as presence_mod

SHA = "0" * 40
NEEDS_APP = {"extensions": {"com.nousresearch.hermes": {"servers": {"srv": {
    "app": {"darwin": {"presence": "executable", "location": "/nonexistent/fx-app"},
            "linux": {"presence": "executable", "location": "/nonexistent/fx-app"},
            "win32": {"presence": "executable", "location": "C:/nonexistent/fx-app.exe"}},
    "requires": {"app": True}}}}}}


def _entry(name, *, onboarding=True, platforms=(), title=""):
    return pc.entry_from_mapping({"name": name, "repo": f"https://github.com/fx/{name}", "sha": SHA,
                                  "description": f"{name} does things. More.", "maintainer": "fx",
                                  "tier": "official", "category": "tools", "platforms": list(platforms),
                                  "onboarding": onboarding, "title": title}, name)


@pytest.fixture
def catalog(monkeypatch):
    from hermes_platform.host import facts

    here = {"darwin": "macos", "win32": "windows"}.get(facts.os_family(), "linux")
    other = "windows" if here != "windows" else "macos"
    entries = [_entry("everywhere", title="Everywhere App"), _entry("not-curated", onboarding=False),
               _entry("here-only", platforms=[here]), _entry("elsewhere", platforms=[other])]
    monkeypatch.setattr(pc, "load_catalog_live", lambda: entries)
    manifests = {"everywhere": NEEDS_APP, "here-only": {"name": "here-only"}}
    monkeypatch.setattr(presence_mod, "_pinned_manifest", lambda repo, sha, subdir: manifests.get(repo.rsplit("/", 1)[-1]))
    return entries


def test_onboarding_rows_are_curated_entries_this_os_runs(catalog):
    rows = {r["name"]: r for r in presence_mod.onboarding_entries()}
    assert set(rows) == {"everywhere", "here-only"}
    assert rows["everywhere"]["title"] == "Everywhere App" and rows["here-only"]["title"] == "here-only"


def test_app_state_comes_from_the_pinned_declaration(catalog):
    rows = {r["name"]: r for r in presence_mod.onboarding_entries()}
    # A declared app that is absent greys the row with a reason; no declaration is unknown, never "present".
    assert rows["everywhere"]["app_state"] == "missing_app" and "Everywhere App" in rows["everywhere"]["sentence"]
    assert rows["here-only"]["app_state"] == "unknown" and rows["here-only"]["sentence"] == ""


def test_unreadable_liveness_is_unknown_never_present(tmp_path, monkeypatch):
    """A liveness block that fails validation must not render as a healthy 'present' card:
    the module contract is that an unreadable declaration is unknown."""
    import sys

    from hermes_platform.declaration import parse_declaration

    exe = tmp_path / "app"
    exe.write_text("", encoding="utf-8")
    decl = parse_declaration(
        "App", {sys.platform: {"presence": "executable", "location": str(exe)}},
        {"app": True}, where="test")
    bad = {"kind": "server_json", "path": "relative/no.json"}  # unrooted on every OS
    assert presence_mod._server_presence(decl, bad, "App") is presence_mod.UNKNOWN
    assert presence_mod._server_presence(decl, None, "App") == presence_mod.Presence("present")
