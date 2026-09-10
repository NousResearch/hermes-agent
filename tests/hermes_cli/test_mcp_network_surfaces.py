"""Server network intent survives real config/API/picker/catalog boundaries.

Only actual connection probing and interactive selection are replaced here;
wire-level HTTP, SSE and OAuth behavior is exercised in the transport tests.
"""
import argparse
import json
from types import SimpleNamespace

import pytest
import yaml


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = set_hermes_home_override(tmp_path)
    yield tmp_path
    reset_hermes_home_override(token)


def parse(*args):
    from hermes_cli.mcp_config import mcp_command
    from hermes_cli.subcommands.mcp import build_mcp_parser
    parser = argparse.ArgumentParser()
    build_mcp_parser(parser.add_subparsers(dest="command"), cmd_mcp=mcp_command)
    return parser.parse_args(["mcp", *args])


def read_config(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["mcp_servers"]


def test_cli_add_configure_and_runtime_loader_preserve_network_and_sse(monkeypatch, home):
    from hermes_cli import mcp_config
    seen = []
    monkeypatch.setattr(mcp_config, "_configure_http_auth", lambda *_: True)
    monkeypatch.setattr(mcp_config, "_probe_single_server", lambda name, cfg: seen.append(dict(cfg)) or [("ping", "fixture")])
    monkeypatch.setattr(mcp_config, "_choose_tools", lambda *_: 1)
    args = parse("add", "unity", "--url", "http://localhost:8080/sse", "--network", "windows", "--transport", "sse")
    mcp_config.mcp_command(args)
    assert seen[0]["network"] == "windows"
    assert seen[0]["transport"] == "sse"
    assert read_config(home)["unity"]["network"] == "windows"
    mcp_config.mcp_command(parse("add", "new-default", "--url", "http://localhost:8080/mcp"))
    assert seen[-1]["network"] == "auto"
    assert read_config(home)["new-default"]["network"] == "auto"
    mcp_config.mcp_command(parse("configure", "unity", "--network", "local"))
    from tools.mcp_tool_config import _load_mcp_config
    loaded = _load_mcp_config()["unity"]
    assert loaded["network"] == "local"
    assert loaded["transport"] == "sse"
    assert loaded["url"] == "http://localhost:8080/sse"


def test_picker_changes_only_http_network_without_dropping_other_fields(monkeypatch, home):
    from hermes_cli import mcp_config, mcp_picker
    original = {"url": "http://localhost:8080/mcp", "transport": "sse", "headers": {"X-Project": "scene"},
                "tools": {"exclude": ["delete"]}, "enabled": False}
    assert mcp_config._save_mcp_server("unity", original)
    assert mcp_config._save_mcp_server("stdio", {"command": "python", "args": ["server.py"]})
    monkeypatch.setattr(mcp_picker, "_run_submenu", lambda _title, options: options[2][1]())
    actions = mcp_picker._network_actions("unity")
    assert len(actions) == 1
    actions[0][1]()
    assert read_config(home)["unity"] == {**original, "network": "windows"}
    assert mcp_picker._network_actions("stdio") == []


@pytest.mark.parametrize("invalid", [
    {"url": "http://localhost:8080/mcp", "network": "typo"},
    {"url": "http://localhost:8080/mcp", "network": []},
    {"command": "python", "network": "windows"},
    {"url": "https://example.com/mcp", "network": "windows"},
])
def test_invalid_save_and_bulk_replace_are_atomic(home, invalid):
    from hermes_cli.mcp_config import _save_mcp_server, _replace_mcp_servers
    assert _save_mcp_server("existing", {"command": "python"})
    before = (home / "config.yaml").read_bytes()
    ok, errors = _replace_mcp_servers({"good": {"url": "http://localhost/mcp"}, "bad": invalid})
    assert not ok and errors
    assert (home / "config.yaml").read_bytes() == before
    assert not _save_mcp_server("bad", invalid)
    assert (home / "config.yaml").read_bytes() == before


def test_portable_env_url_can_be_saved_but_runtime_validates_resolved_target(monkeypatch, home):
    from hermes_cli.mcp_config import _save_mcp_server
    from tools.mcp_tool_config import _load_mcp_config
    from tools.mcp_windows import InvalidMcpNetworkError, validate_mcp_network
    monkeypatch.setenv("FIXTURE_MCP_URL", "https://example.com/mcp")
    assert _save_mcp_server("portable", {"url": "${FIXTURE_MCP_URL}", "network": "windows"})
    cfg = _load_mcp_config()["portable"]
    with pytest.raises(InvalidMcpNetworkError, match="loopback"):
        validate_mcp_network(cfg["url"], cfg["network"])


@pytest.fixture
def client(monkeypatch):
    from starlette.testclient import TestClient
    from hermes_cli import web_server
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    c = TestClient(web_server.app)
    c.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    yield c
    c.close()


def test_rest_create_list_and_bulk_save_preserve_network_and_transport(client, home):
    body = {"name": "unity", "url": "http://localhost:8080/sse", "transport": "sse", "network": "windows"}
    response = client.post("/api/mcp/servers", json=body)
    assert response.status_code == 200, response.text
    saved = read_config(home)["unity"]
    assert saved["network"] == "windows" and saved["transport"] == "sse"
    rows = client.get("/api/mcp/servers").json()["servers"]
    row = next(row for row in rows if row["name"] == "unity")
    assert row["network"] == "windows" and row["transport"] == "sse"
    response = client.post("/api/mcp/servers", json={"name": "new-default", "url": "http://localhost:8080/mcp"})
    assert response.status_code == 200, response.text
    assert read_config(home)["new-default"]["network"] == "auto"
    response = client.put("/api/mcp/servers", json={"servers": {
        "unity": {**saved, "network": "local"},
        "legacy": {"url": "http://localhost:8081/mcp"},
    }})
    assert response.status_code == 200, response.text
    replaced = read_config(home)
    assert replaced["unity"]["network"] == "local"
    assert "network" not in replaced["legacy"]
    rows = client.get("/api/mcp/servers").json()["servers"]
    assert next(row for row in rows if row["name"] == "legacy")["network"] == "local"
    before = (home / "config.yaml").read_bytes()
    bad = client.post("/api/mcp/servers", json={**body, "name": "bad", "network": "typo"})
    assert bad.status_code == 422
    assert (home / "config.yaml").read_bytes() == before


def test_profile_builder_uses_same_contract_without_touching_foreground_profile(home):
    from hermes_cli.mcp_config import _save_mcp_server
    from hermes_cli.web_models import MCPServerCreate
    from hermes_cli.web_server_profiles import _write_profile_mcp_servers
    assert _save_mcp_server("foreground", {"command": "python"})
    before = (home / "config.yaml").read_bytes()
    target = home / "profiles" / "unity"
    target.mkdir(parents=True)
    count = _write_profile_mcp_servers(target, [
        MCPServerCreate(name="unity", url="http://localhost:8080/sse", transport="sse"),
    ])
    assert count == 1
    saved = read_config(target)["unity"]
    assert saved["network"] == "auto" and saved["transport"] == "sse"
    assert (home / "config.yaml").read_bytes() == before


def catalog_entry():
    from hermes_cli.mcp_catalog import AuthSpec, CatalogEntry, TransportSpec
    return CatalogEntry(name="fixture-local", description="Fixture", source="fixture",
                        transport=TransportSpec(type="http", url="http://localhost:8080/mcp"), auth=AuthSpec(type="none"))


def test_catalog_cli_reinstall_and_api_keep_or_override_network(monkeypatch, home, client):
    from hermes_cli import mcp_catalog, mcp_config
    entry = catalog_entry()
    monkeypatch.setattr(mcp_catalog, "get_entry", lambda name: entry)
    monkeypatch.setattr(mcp_catalog, "_apply_tool_selection", lambda *_a, **_kw: None)

    # Fresh installs explicitly persist the new-entry automatic choice.
    mcp_catalog.install_entry(entry)
    assert read_config(home)[entry.name]["network"] == "auto"

    # Reinstalling a pre-feature entry without the field preserves omission/local.
    legacy = {"url": entry.transport.url, "enabled": True}
    assert mcp_config._save_mcp_server(entry.name, legacy)
    mcp_catalog.install_entry(entry)
    assert "network" not in read_config(home)[entry.name]

    mcp_config.mcp_command(parse("install", entry.name, "--network", "windows"))
    assert read_config(home)[entry.name]["network"] == "windows"
    mcp_catalog.install_entry(entry)
    assert read_config(home)[entry.name]["network"] == "windows"
    response = client.post("/api/mcp/catalog/install", json={"name": entry.name, "network": "local"})
    assert response.status_code == 200, response.text
    assert read_config(home)[entry.name]["network"] == "local"


def test_catalog_install_action_forwards_network_and_owner(monkeypatch, client, home):
    from hermes_cli import mcp_catalog
    from hermes_cli.web_routers import mcp as router
    entry = catalog_entry()
    entry.install = SimpleNamespace()
    (home / "profiles" / "unity").mkdir(parents=True)
    monkeypatch.setattr(mcp_catalog, "get_entry", lambda name: entry)
    calls = []
    monkeypatch.setattr(router, "_spawn_hermes_action", lambda args, action: calls.append((args, action)))
    response = client.post("/api/mcp/catalog/install", json={"name": entry.name, "network": "windows", "profile": "unity"})
    assert response.status_code == 200, response.text
    argv = calls[0][0]
    assert argv[argv.index("--network") + 1] == "windows"
    assert "unity" in argv and entry.name in argv


def test_import_keeps_routing_and_sse_but_does_not_import_bearer_tokens():
    from hermes_cli.agent_import import _translate_mcp_server
    translated, stripped = _translate_mcp_server("unity", {
        "url": "http://localhost:8080/sse", "network": "windows", "type": "sse",
        "headers": {"Authorization": "Bearer fixture-secret", "X-Project": "scene"}})
    assert translated == {"url": "http://localhost:8080/sse", "network": "windows", "transport": "sse", "headers": {"X-Project": "scene"}}
    assert stripped == ["mcp_servers.unity.headers.Authorization"]
    assert "fixture-secret" not in json.dumps(translated)
    legacy, _ = _translate_mcp_server("legacy", {"url": "http://localhost:8080/mcp"})
    assert "network" not in legacy


def test_disk_schema_cache_cannot_replay_another_networks_tools():
    from tools import mcp_schema_cache as cache
    cfg = {"url": "http://localhost:8080/mcp", "network": "local", "lazy": True}
    fp = cache.config_fingerprint(cfg)
    cache.write_cache_entry("unity", fp, tools=[{"name": "linux_only", "inputSchema": {}}], utility_tools=[])
    assert cache.get_cached_entry("unity", fp) is not None
    assert cache.get_cached_entry("unity", cache.config_fingerprint({**cfg, "network": "windows"})) is None
    assert cache.config_fingerprint({"url": cfg["url"]}) == cache.config_fingerprint({"url": cfg["url"], "network": "local"})
    assert cache.config_fingerprint({"url": cfg["url"]}) != cache.config_fingerprint({"url": cfg["url"], "network": "auto"})


@pytest.mark.parametrize("kind", ["http", "sse"])
def test_acp_session_servers_keep_transport_and_auth_headers(kind):
    from acp.schema import McpServerHttp, McpServerSse
    from acp_adapter.server import _mcp_server_config
    model = McpServerSse if kind == "sse" else McpServerHttp
    server = model(name="fixture", type=kind, url=f"http://localhost:8080/{kind}", headers=[{"name": "X-Project", "value": "scene"}])
    config = _mcp_server_config(server)
    assert config["headers"] == {"X-Project": "scene"}
    assert config.get("transport", "http") == kind
    assert config["network"] == "local"
