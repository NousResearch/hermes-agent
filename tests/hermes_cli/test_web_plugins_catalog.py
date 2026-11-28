"""Dashboard plugin-catalog surface: GET /api/dashboard/plugins/catalog merges installed state via the
installer-owned install-metadata ``catalog`` record; the install endpoint has no kill-list bypass."""

from __future__ import annotations

import pytest
import hermes_yaml as yaml

from hermes_cli import plugin_catalog as pc_cat

VALID_SHA = "38fe0fb53eff98d477f807432e965429e665ca33"
OTHER_SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


@pytest.fixture
def client(monkeypatch, tmp_path, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    (catalog_dir / "alpha-plugin.yaml").write_text(yaml.safe_dump({
        "name": "alpha-plugin", "repo": "https://github.com/example/alpha-plugin", "sha": VALID_SHA,
        "description": "d", "maintainer": "Example", "tier": "official",
        "capabilities": {"provides_tools": ["tool_a"], "requires_env": ["EXAMPLE_API_KEY"]}}))
    (catalog_dir / "removed.yaml").write_text(yaml.safe_dump({"removed": [
        {"name": "bad-plugin", "repo": "https://github.com/evil/bad-plugin", "reason": "exfiltrated env vars"}]}))
    monkeypatch.setattr(pc_cat, "get_catalog_dir", lambda: catalog_dir)
    monkeypatch.setattr(pc_cat, "fetch_live_catalog", lambda **_: None)

    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _install(name: str, sidecar: dict | None):
    """A user-dir plugin; *sidecar* records catalog provenance the way the installer does — on the
    installer-owned ``.install-metadata.json`` record (an in-tree ``.hermes-catalog.json`` is inert)."""
    from hermes_constants import get_hermes_home
    from hermes_cli.plugins_cmd import _read_install_metadata, _write_install_metadata
    d = get_hermes_home() / "plugins" / name
    d.mkdir(parents=True)
    (d / "plugin.yaml").write_text(yaml.safe_dump({"name": name, "version": "1.0", "description": "x"}))
    if sidecar:
        block = {"name": sidecar["catalog_name"], "sha": sidecar["sha"], "tier": sidecar.get("tier", "community")}
        _write_install_metadata({**_read_install_metadata(), name: {
            "pinned": True, "revision": sidecar["sha"], "source": "https://github.com/example/alpha-plugin.git",
            "catalog": block}})


def test_catalog_endpoint_merges_installed_state_from_sidecar(client):
    from starlette.testclient import TestClient
    from hermes_cli.web_server import app
    assert TestClient(app).get("/api/dashboard/plugins/catalog").status_code == 401

    # Manifest name differs from the catalog name (the common case): matched through the sidecar.
    _install("alpha", {"catalog_name": "alpha-plugin", "sha": OTHER_SHA, "tier": "official"})
    data = client.get("/api/dashboard/plugins/catalog").json()
    [entry] = data["entries"]
    assert (entry["name"], entry["sha_short"], entry["capabilities"]["provides_tools"]) == ("alpha-plugin", VALID_SHA[:7], ["tool_a"])
    assert "tool_a" in entry["capability_summary"]
    assert (entry["installed"], entry["installed_sha"], entry["update_available"]) == (True, OTHER_SHA, True)
    assert entry["runtime_status"] == "inactive"
    assert data["removed"][0]["reason"] == "exfiltrated env vars"


def test_catalog_endpoint_does_not_match_bundled_or_entrypoint_plugins_by_name(client, tmp_path, monkeypatch):
    """A bundled plugin (e.g. ``plugins/cron_providers/chronos``) or pip entry-point plugin that
    shares a bare name with an unrelated catalog entry must never mark that catalog entry as
    installed — only user-dir installs can satisfy catalog entries."""
    from hermes_cli import plugins_cmd

    catalog_dir = tmp_path / "catalog_collision"
    catalog_dir.mkdir()
    for name in ("chronos", "pip-twin"):
        (catalog_dir / f"{name}.yaml").write_text(yaml.safe_dump({
            "name": name, "repo": f"https://github.com/example/{name}", "sha": VALID_SHA,
            "description": f"community {name}", "maintainer": "Example", "tier": "community"}))
    monkeypatch.setattr(pc_cat, "get_catalog_dir", lambda: catalog_dir)

    bundled_dir = tmp_path / "bundled" / "cron_providers" / "chronos"
    bundled_dir.mkdir(parents=True)
    ep_dir = tmp_path / "site-packages" / "pip_twin"
    ep_dir.mkdir(parents=True)
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: [
        ("chronos", "1.0.0", "Bundled cron provider", "bundled", bundled_dir, "cron_providers/chronos"),
        ("pip-twin", "1.0.0", "Entry-point plugin", "entrypoint", ep_dir, "pip-twin"),
    ])

    by_name = {e["name"]: e for e in client.get("/api/dashboard/plugins/catalog").json()["entries"]}
    assert by_name["chronos"]["installed"] is False
    assert by_name["chronos"]["runtime_status"] is None
    assert by_name["pip-twin"]["installed"] is False
    assert by_name["pip-twin"]["runtime_status"] is None

    # Direct calls to installed_catalog_state also ignore non-catalog sources if passed.
    from hermes_cli.plugins_cmd_catalog import installed_catalog_state
    direct = {e["name"]: e for e in installed_catalog_state({
        "chronos": {"dir": str(bundled_dir), "source": "bundled", "runtime_status": "inactive"},
        "pip-twin": {"dir": str(ep_dir), "source": "entrypoint", "runtime_status": "enabled"},
    })["entries"]}
    assert direct["chronos"]["installed"] is False
    assert direct["pip-twin"]["installed"] is False

    # Installing the actual catalog plugin in the user plugins dir marks it installed cleanly.
    _install("chronos", {"catalog_name": "chronos", "sha": VALID_SHA, "tier": "community"})
    from hermes_constants import get_hermes_home
    user_dir = get_hermes_home() / "plugins" / "chronos"
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: [
        ("chronos", "1.0.0", "Bundled cron provider", "bundled", bundled_dir, "cron_providers/chronos"),
        ("chronos", "1.0.0", "Community chronos", "git", user_dir, "chronos"),
    ])
    by_name = {e["name"]: e for e in client.get("/api/dashboard/plugins/catalog").json()["entries"]}
    assert by_name["chronos"]["installed"] is True
    assert by_name["chronos"]["installed_sha"] == VALID_SHA
    assert by_name["chronos"]["runtime_status"] == "inactive"


def test_catalog_install_with_different_manifest_name_does_not_claim_second_entry(client, tmp_path, monkeypatch):
    """If catalog entry ``alpha-plugin`` installs with manifest name ``beta``, its installer-owned
    sidecar pins it to ``alpha-plugin`` so it must not also fall back to claim an unrelated catalog
    entry named ``beta``."""
    catalog_dir = tmp_path / "catalog_two"
    catalog_dir.mkdir()
    for name in ("alpha-plugin", "beta"):
        (catalog_dir / f"{name}.yaml").write_text(yaml.safe_dump({
            "name": name, "repo": f"https://github.com/example/{name}", "sha": VALID_SHA,
            "description": name, "maintainer": "Example", "tier": "community"}))
    monkeypatch.setattr(pc_cat, "get_catalog_dir", lambda: catalog_dir)

    _install("beta", {"catalog_name": "alpha-plugin", "sha": VALID_SHA, "tier": "community"})
    by_name = {e["name"]: e for e in client.get("/api/dashboard/plugins/catalog").json()["entries"]}
    assert by_name["alpha-plugin"]["installed"] is True
    assert by_name["beta"]["installed"] is False
    assert by_name["beta"]["runtime_status"] is None


def test_install_endpoint_refuses_removed_plugins_with_no_bypass(client, monkeypatch):
    from hermes_cli import plugins_cmd
    monkeypatch.setattr(plugins_cmd, "_install_plugin_core", lambda *a, **k: pytest.fail("kill-listed install ran"))
    for body in ({"identifier": "https://github.com/evil/bad-plugin.git"},
                 {"identifier": "", "catalog_name": "bad-plugin"}, {"identifier": "evil/bad-plugin"}):
        resp = client.post("/api/dashboard/agent-plugins/install", json=body)
        assert resp.status_code == 400, body
        assert "removed" in resp.json()["detail"] or "not in the Hermes plugin catalog" in resp.json()["detail"]
    assert client.post("/api/dashboard/agent-plugins/install", json={"identifier": ""}).status_code == 400
