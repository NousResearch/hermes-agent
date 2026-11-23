"""The machine's UI preferences store (#145213): one file beside the ROOT Hermes home, shared by
native Desktop and every browser that opens this machine's Webapp.

Contracts: a profile's HERMES_HOME resolves the same file as the root; the one-time localStorage
import never overwrites an existing store; writes merge per key (concurrent writers keep each
other's keys); only Desktop-prefixed string keys are stored; the route exists only on the Webapp
and only past the session-token gate.
"""
from __future__ import annotations

import json
import multiprocessing
import os

import pytest
from fastapi.testclient import TestClient

from hermes_cli import ui_preferences, web_server


def test_a_profile_home_and_the_root_home_resolve_the_same_store(monkeypatch, tmp_path):
    root = tmp_path / "custom-root"
    (root / "profiles" / "coder").mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(root))
    from_root = ui_preferences.store_path()
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
    from_profile = ui_preferences.store_path()

    assert from_root == from_profile == root / "ui-preferences.json"


def test_a_custom_home_under_the_native_home_is_its_own_root(monkeypatch, tmp_path):
    """Native Desktop keeps an explicit ``HERMES_HOME`` that is not a profile as the root; the
    Webapp's server must resolve the same file, not collapse it to the native home."""
    sandbox = tmp_path / ".hermes" / "sandbox"
    sandbox.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(sandbox))

    assert ui_preferences.store_path() == sandbox.resolve() / "ui-preferences.json"


def test_the_seed_import_only_adds_keys_and_never_creates_an_empty_store(tmp_path):
    """A client's first contact fills what the store lacks: the client that has data fills a
    store an empty client reached first, and nothing already stored is overwritten."""
    path = tmp_path / "ui-preferences.json"
    empty = ui_preferences.write_preferences({}, seed=True, path=path)
    first = ui_preferences.write_preferences({"hermes.desktop.sessionListDensity": "detailed"},
                                             seed=True, path=path)
    second = ui_preferences.write_preferences({"hermes.desktop.sessionListDensity": "compact",
                                               "hermes-desktop-mode-v1": "dark"}, seed=True, path=path)

    assert empty == {"exists": False, "revision": 0, "values": {}}
    assert first["values"] == {"hermes.desktop.sessionListDensity": "detailed"}
    assert second["values"] == {"hermes.desktop.sessionListDensity": "detailed", "hermes-desktop-mode-v1": "dark"}
    assert second["generation"] == first["generation"]
    assert json.loads(path.read_text())["values"] == second["values"]


@pytest.mark.platforms("linux", "macos")
def test_an_unreadable_or_corrupt_store_is_never_merged_over(tmp_path):
    """Only a missing file is no store. An unreadable one refuses the write (the client retries)
    and stays intact; a corrupt one is kept aside and replaced by a NEW generation, which every
    open window follows whatever revision it had seen."""
    path = tmp_path / "ui-preferences.json"
    kept = ui_preferences.write_preferences({"hermes-desktop-theme-v2": "nous", "hermes.desktop.keybinds": "{}"},
                                            path=path)
    before = path.read_bytes()
    path.chmod(0)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("running as a user that reads mode-000 files (root)")
        with pytest.raises(ui_preferences.PreferencesBusy):
            ui_preferences.write_preferences({"hermes-desktop-mode-v1": "dark"}, path=path)
    finally:
        path.chmod(0o600)
    assert path.read_bytes() == before

    torn = before[:-5]
    path.write_bytes(torn)
    assert ui_preferences.read_preferences(path)["exists"] is False
    after = ui_preferences.write_preferences({"hermes-desktop-mode-v1": "dark"}, path=path)

    backups = list(tmp_path.glob("ui-preferences.json.corrupt-*"))
    assert [backup.read_bytes() for backup in backups] == [torn]
    assert after["values"] == {"hermes-desktop-mode-v1": "dark"}
    assert after["generation"] != kept["generation"]


def test_writes_merge_per_key_and_concurrent_writers_keep_each_others_keys(tmp_path):
    path = tmp_path / "ui-preferences.json"
    ui_preferences.write_preferences({"hermes-desktop-mode-v1": "dark", "hermes.desktop.panesFlipped": "true"},
                                     path=path)

    # Separate processes, like native Desktop and the Webapp's server: only the directory lock
    # stands between their read-merge-replace cycles.
    keys = [f"hermes.desktop.k{n}" for n in range(8)]
    spawn = multiprocessing.get_context("spawn")
    writers = [spawn.Process(target=ui_preferences.write_preferences, args=({key: "1"},),
                             kwargs={"path": path}) for key in keys]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(60)
    assert [writer.exitcode for writer in writers] == [0] * len(keys)
    after = ui_preferences.write_preferences({"hermes.desktop.panesFlipped": None}, path=path)

    assert after["values"] == {"hermes-desktop-mode-v1": "dark", **{key: "1" for key in keys}}
    assert after["revision"] == 2 + len(keys)
    assert not path.with_name(path.name + ".lock").exists()


@pytest.mark.parametrize("changes", [{"hermes.agent.secret": "x"}, {"hermes.desktop.n": 3}, {"": "x"}])
def test_only_desktop_string_keys_are_stored(tmp_path, changes):
    path = tmp_path / "ui-preferences.json"

    with pytest.raises(ui_preferences.PreferencesError):
        ui_preferences.write_preferences(changes, path=path)
    assert not path.exists()


@pytest.fixture
def webapp_state(monkeypatch):
    for key in ("ui_surface", "gateway_runner", "listener_public_url", "webapp_home"):
        if hasattr(web_server.app.state, key):
            monkeypatch.delattr(web_server.app.state, key)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)


def _client(token: bool) -> TestClient:
    client = TestClient(web_server.app, raise_server_exceptions=False)
    if token:
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client


def test_the_webapp_reads_and_writes_the_machine_store_behind_the_session_gate(webapp_state):
    from hermes_constants import get_default_hermes_root

    store = get_default_hermes_root() / "ui-preferences.json"
    anonymous = _client(token=False)

    assert anonymous.get("/api/ui-preferences").status_code == 401
    assert anonymous.patch("/api/ui-preferences", json={"values": {"hermes-desktop-mode-v1": "dark"}}).status_code == 401
    assert not store.exists()

    tab = _client(token=True)
    assert tab.get("/api/ui-preferences").json() == {"exists": False, "revision": 0, "values": {}}
    written = tab.patch("/api/ui-preferences", json={"values": {"hermes-desktop-mode-v1": "dark"}})
    assert written.status_code == 200
    assert json.loads(store.read_text())["values"] == {"hermes-desktop-mode-v1": "dark"}
    assert tab.patch("/api/ui-preferences", json={"values": {"hermes.other": "x"}}).status_code == 400


@pytest.mark.parametrize("surface", [None, "serve", "dashboard"])
def test_the_route_exists_only_on_the_webapp(monkeypatch, webapp_state, surface):
    if surface is None:
        monkeypatch.delattr(web_server.app.state, "ui_surface")
    else:
        monkeypatch.setattr(web_server.app.state, "ui_surface", surface, raising=False)

    assert _client(token=True).get("/api/ui-preferences").status_code == 404
