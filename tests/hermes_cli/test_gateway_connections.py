"""The machine's saved gateways list: one file both native Desktop and the Webapp write.

Contracts: the Python normalizer is the TS one's twin (one shared fixture, both sides); saves keep
Electron's semantics (duplicates refused with the colliding label, edits inherit what the editor
omits, a token-mode remote needs a token); the route speaks the IPC shapes, never returns a secret,
exists only on the Webapp, and only past the session gate; concurrent writers keep each other's saves.
"""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hermes_cli import gateway_connections as store
from hermes_cli import web_server
from hermes_cli.gateway_connections_normalize import (
    merge_connection_input,
    normalize_connection_input,
    normalize_registry,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "gateway_connections" / "normalize_cases.json"


@pytest.fixture
def home(monkeypatch, tmp_path):
    root = tmp_path / "root"
    (root / "profiles" / "coder").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
    return root.resolve()


def test_the_python_normalizer_matches_the_typescript_one_on_the_shared_fixture():
    cases = json.loads(FIXTURE.read_text())
    for name, raw in cases["inputs"].items():
        assert normalize_registry(raw) == cases["expected"][name], name
    registry = normalize_registry(cases["saveRegistry"])
    for payload, expected in zip(cases["saves"], cases["expectedSaves"], strict=True):
        existing = next((c for c in registry["connections"] if payload.get("id") and c["id"] == payload["id"]), None)
        try:
            got = {"entry": normalize_connection_input(merge_connection_input(payload, existing), registry)}
        except Exception as exc:  # the fixture records TS's thrown message
            got = {"error": str(exc)}
        assert got == expected, payload


def test_a_profile_home_names_the_root_store_and_writes_land_owner_only(home):
    assert store.store_path() == home / "gateway-connections.json"
    assert store.tokens_path() == home / "gateway-connection-tokens.json"
    assert store.read_registry()["connections"] == [{"id": "local", "kind": "local", "label": "This device"}]

    connection, registry = store.save_connection(
        {"kind": "remote", "label": "Homelab", "url": "10.0.0.5:9119/", "token": " tok-1234567890 "})

    on_disk = json.loads(store.store_path().read_text())
    assert (store.store_path().stat().st_mode & 0o777) == 0o600
    assert on_disk["revision"] == registry["revision"] == 1
    assert on_disk["connections"][1]["url"] == "http://10.0.0.5:9119"
    assert on_disk["connections"][1]["token"] == {"encoding": "plain", "value": "tok-1234567890"}
    assert connection == {"id": "homelab", "kind": "remote", "label": "Homelab", "url": "http://10.0.0.5:9119",
                          "authMode": "token", "tokenSet": True, "tokenPreview": "...567890", "headerNames": []}


def test_saves_keep_electrons_rules(home):
    store.save_connection({"kind": "remote", "label": "Homelab", "url": "http://a:1", "token": "t",
                           "headers": {"CF-Access-Client-Id": "id", "Cookie": "dropped"}})
    with pytest.raises(store.DuplicateConnection) as dup:
        store.save_connection({"kind": "remote", "label": "HOMELAB", "url": "http://b:1", "token": "t"})
    assert dup.value.label == "Homelab"
    with pytest.raises(store.DuplicateConnection):
        store.save_connection({"kind": "cloud", "label": "Other", "url": "HTTP://a:1/"})
    with pytest.raises(store.ConnectionStoreError, match="session token is required"):
        store.save_connection({"kind": "remote", "label": "Tokenless", "url": "http://c:1"})

    # An edit that omits the token and headers keeps them; a null header value keeps that one.
    store.save_connection({"id": "homelab", "kind": "remote", "label": "Renamed"})
    store.save_connection({"id": "homelab", "kind": "remote", "label": "Renamed",
                           "headers": {"CF-Access-Client-Id": None, "X-Extra": "v"}})
    assert store.connection_secret("homelab") == {"token": "t", "headers": {"CF-Access-Client-Id": "id", "X-Extra": "v"}}

    registry = store.set_primary("homelab")
    assert (registry["primary"], registry["lastUsed"]) == ("homelab", "local")
    registry = store.remove_connection("homelab")
    assert registry["primary"] == "local" and [c["id"] for c in registry["connections"]] == ["local"]
    with pytest.raises(store.NoSuchConnection):
        store.remove_connection("homelab")
    with pytest.raises(store.ConnectionStoreError):
        store.remove_connection("local")
    with pytest.raises(store.ConnectionStoreError):
        store.set_launch_mode("sometimes")


def test_an_undecrypted_desktop_envelope_reads_as_unset_and_survives_python_writes(home):
    """Desktop keeps a secret its keychain would not open as the original envelope; the Webapp
    neither uses nor drops it."""
    blob = {"encoding": "safeStorage", "value": "YmxvYg=="}
    store.store_path().write_text(json.dumps({"version": 2, "revision": 4, "importedFrom": {"/ud": {"at": 1}},
                                              "connections": [{"id": "r", "kind": "remote", "label": "R",
                                                               "url": "http://r", "token": blob}]}))
    assert store.connection_secret("r")["token"] is None
    assert store.public_registry(store.read_registry())["connections"][1]["tokenSet"] is False

    store.set_last_used("r")
    on_disk = json.loads(store.store_path().read_text())
    assert on_disk["connections"][1]["token"] == blob
    assert on_disk["importedFrom"] == {"/ud": {"at": 1}} and on_disk["revision"] == 5


def test_a_corrupt_store_is_kept_aside_and_an_unreadable_one_refuses(home, tmp_path):
    store.store_path().write_text("{not json")
    store.set_launch_mode("last-used")
    assert list(home.glob("gateway-connections.json.corrupt-*"))
    assert json.loads(store.store_path().read_text())["launchMode"] == "last-used"

    if os.geteuid() != 0:
        store.store_path().chmod(0)
        with pytest.raises(store.ConnectionStoreBusy):
            store.set_launch_mode("primary")


def _save_one(root: str, index: int) -> None:
    os.environ["HERMES_HOME"] = root
    store.save_connection({"kind": "remote", "label": f"Box {index}", "url": f"http://box{index}:1", "token": "t"})


def test_concurrent_writers_keep_each_others_saves(home):
    ctx = multiprocessing.get_context("spawn")
    workers = [ctx.Process(target=_save_one, args=(str(home), i)) for i in range(6)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(30)
    labels = {c["label"] for c in store.read_registry()["connections"]}
    assert labels == {"This device"} | {f"Box {i}" for i in range(6)}


def test_oauth_token_sets_are_keyed_by_connection_and_fall_back_to_an_unbound_url(home):
    store.save_connection({"kind": "remote", "label": "Gw", "url": "https://gw.example", "authMode": "oauth"})
    first = {"accessToken": "a1", "refreshToken": "r1", "expiresAt": 1, "provider": "p", "userId": "u"}
    store.tokens_path().write_text(json.dumps({"version": 1, "revision": 1, "tokens": {},
                                              "unbound": {"https://gw.example": first}}))
    assert store.read_tokens("gw") == {**first, "baseUrl": "https://gw.example"}

    second = dict(first, accessToken="a2")
    store.write_tokens("gw", second)
    doc = json.loads(store.tokens_path().read_text())
    assert doc["tokens"] == {"gw": {**second, "baseUrl": "https://gw.example"}}
    assert doc["unbound"] == {} and doc["revision"] == 2
    store.write_tokens("gw", None)
    assert store.read_tokens("gw") is None


@pytest.fixture
def webapp_state(monkeypatch, home):
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


def test_the_route_speaks_the_ipc_shapes_behind_the_gate_and_never_returns_a_secret(webapp_state):
    anonymous = _client(token=False)
    assert anonymous.get("/api/gateway-connections").status_code == 401
    assert anonymous.post("/api/gateway-connections", json={"kind": "remote"}).status_code == 401
    assert not store.store_path().exists()

    tab = _client(token=True)
    listed = tab.get("/api/gateway-connections").json()
    assert listed["secureTokenStorage"] is True and listed["quarantined"] == []

    saved = tab.post("/api/gateway-connections", json={
        "kind": "remote", "label": "Homelab", "url": "http://h:1", "token": "secret-token-value",
        "headers": {"CF-Access-Client-Secret": "header-secret"}})
    body = saved.json()
    assert saved.status_code == 200 and body["ok"] is True
    assert body["connection"]["headerNames"] == ["CF-Access-Client-Secret"]
    assert "secret-token-value" not in saved.text and "header-secret" not in saved.text

    dup = tab.post("/api/gateway-connections", json={"kind": "remote", "label": "homelab", "url": "http://x:1", "token": "t"})
    assert dup.status_code == 400 and "Homelab" in dup.json()["detail"]
    assert tab.post("/api/gateway-connections/homelab/primary").json()["registry"]["primary"] == "homelab"
    assert tab.post("/api/gateway-connections/launch-mode", json={"mode": "last-used"}).json()["registry"]["launchMode"] == "last-used"
    assert tab.post("/api/gateway-connections/last-used", json={"id": "homelab"}).json()["registry"]["lastUsed"] == "homelab"
    assert tab.post("/api/gateway-connections/nope/primary").status_code == 404
    assert tab.delete("/api/gateway-connections/nope").status_code == 404
    removed = tab.delete("/api/gateway-connections/homelab").json()
    assert removed["ok"] and removed["registry"]["primary"] == "local"


@pytest.mark.parametrize("surface", [None, "serve", "dashboard", "gateway"])
def test_the_route_exists_only_on_the_webapp(monkeypatch, webapp_state, surface):
    if surface is None:
        monkeypatch.delattr(web_server.app.state, "ui_surface")
    else:
        monkeypatch.setattr(web_server.app.state, "ui_surface", surface, raising=False)
    assert _client(token=True).get("/api/gateway-connections").status_code == 404
