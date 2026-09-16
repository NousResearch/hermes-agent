"""Catalog HTTP contract: opt-in backend signals with real A/B/A profile config I/O."""

import asyncio
from pathlib import Path
import sys

import pytest
import yaml
from fastapi.testclient import TestClient


@pytest.fixture
def catalog_client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    other = home / "profiles" / "b"
    other.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", "")
    for directory, servers in ((home, {"demo": {"command": "unused", "enabled": False}}),
                               (other, {"demo": {"command": "unused", "enabled": True},
                                        "legacy": {"command": "unused"}})):
        (directory / "config.yaml").write_text(yaml.safe_dump(
            {"mcp_servers": servers, "terminal": {"backend": "docker"}}), encoding="utf-8")
    catalog = tmp_path / "catalog"
    for name, suggest in (("demo", {"keywords": ["demo"], "applications": ["Fixture Paint"],
                                   "examples": ["Draw a picture."], "requires_app": True}),
                          ("legacy", {"keywords": ["legacy"]})):
        entry = catalog / name
        entry.mkdir(parents=True)
        (entry / "manifest.yaml").write_text(yaml.safe_dump({
            "manifest_version": 1, "name": name, "description": "Fixture entry",
            "transport": {"type": "stdio", "command": "must-not-run"}, "suggest": suggest,
        }), encoding="utf-8")
    monkeypatch.setenv("HERMES_OPTIONAL_MCPS", str(catalog))

    from agent import secret_scope
    from tui_gateway import launch_profile_policy
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client, home, other


def test_catalog_detection_is_opt_in_and_preserves_profile_state(catalog_client, tmp_path, monkeypatch):
    from hermes_cli import mcp_app_detection as detection
    from hermes_constants import get_hermes_home

    client, home, other = catalog_client
    import socket
    import subprocess

    def forbidden(*_args, **_kwargs):
        pytest.fail("Catalog discovery must not execute apps, install or probe the network")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    app_root = tmp_path / "Applications"
    app_root.mkdir()
    if sys.platform == "darwin":
        (app_root / "Fixture Paint.app").mkdir()
    elif sys.platform == "win32":
        (app_root / "Fixture Paint").mkdir()
    else:
        (app_root / "fixture.desktop").write_text("[Desktop Entry]\nName=Fixture Paint\nExec=must-not-run\n")
    monkeypatch.setattr(detection, "_application_roots", lambda: [app_root])
    calls = []
    discover = detection.discover_catalog_apps

    def checked_discover(criteria):
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()  # Scans stay off the event loop.
        assert get_hermes_home() == home  # No named-profile scope/skills lock spans discovery.
        calls.append(criteria)
        return discover(criteria)

    monkeypatch.setattr(detection, "discover_catalog_apps", checked_discover)
    snapshots = {directory: (directory / "config.yaml").read_bytes() for directory in (home, other)}
    for profile, enabled, legacy_installed in ((None, False, False), ("b", True, True), (None, False, False)):
        params = {"profile": profile} if profile else {}
        before = len(calls)
        old = client.get("/api/mcp/catalog", params=params)
        assert old.status_code == 200, old.text
        assert len(calls) == before
        assert "discovery" not in old.json()
        assert all("detected_apps" not in entry for entry in old.json()["entries"])
        new = client.get("/api/mcp/catalog", params={**params, "detect_apps": "true", "future_hint": "ignored"})
        assert new.status_code == 200, new.text
        data = new.json()
        assert data["discovery"] == {"scope": "backend", "status": "ok", "platform": sys.platform}
        entries = {entry["name"]: entry for entry in data["entries"]}
        assert entries["demo"]["detected_apps"] == ["Fixture Paint"]
        assert entries["demo"]["installed"] is True and entries["demo"]["enabled"] is enabled
        assert entries["legacy"]["detected_apps"] == []
        assert entries["legacy"]["installed"] is legacy_installed
        assert entries["legacy"]["suggest"]["applications"] == []
        assert entries["legacy"]["suggest"]["requires_app"] is False
        for entry in data["entries"]:
            entry.pop("detected_apps")
        data.pop("discovery")
        assert data == old.json()
        assert len(calls) == before + 1
    assert {directory: (directory / "config.yaml").read_bytes() for directory in snapshots} == snapshots
    future = tmp_path / "catalog" / "fixture-paint"
    future.mkdir()
    manifest = future / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({
        "manifest_version": 1, "name": "fixture-paint", "description": "Create illustrations in Fixture Paint",
        "transport": {"type": "stdio", "command": "must-not-run"},
    }), encoding="utf-8")
    fresh = client.get("/api/mcp/catalog?detect_apps=true").json()
    added = next(entry for entry in fresh["entries"] if entry["name"] == "fixture-paint")
    assert added["suggest"] is None
    assert added["detected_apps"] == ["fixture paint"]
    manifest.unlink()
    assert all(entry["name"] != "fixture-paint" for entry in client.get("/api/mcp/catalog?detect_apps=true").json()["entries"])
    assert client.get("/api/mcp/catalog", params={"profile": "missing", "detect_apps": True}).status_code == 404


def test_catalog_discovery_failure_retains_entries_as_unknown(catalog_client, monkeypatch):
    from hermes_cli import mcp_app_detection as detection

    client, _home, _other = catalog_client
    calls = []

    def failed_discovery(criteria):
        calls.append(criteria)
        raise OSError("private path must not be returned")

    monkeypatch.setattr(detection, "discover_catalog_apps", failed_discovery)
    old = client.get("/api/mcp/catalog?detect_apps=false").json()
    assert calls == []
    response = client.get("/api/mcp/catalog?detect_apps=true")
    assert response.status_code == 200
    data = response.json()
    assert data["discovery"] == {"scope": "backend", "status": "unavailable", "platform": sys.platform}
    assert all(entry.pop("detected_apps") == [] for entry in data["entries"])
    data.pop("discovery")
    assert data == old
    assert "private path" not in response.text
    assert client.get("/api/mcp/catalog?detect_apps=not-a-bool").status_code == 422
