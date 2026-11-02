"""Profile-builder query-auth persistence for MCP servers (#61562)."""

from starlette.testclient import TestClient

import hermes_yaml as yaml


def test_profiles_create_builder_query_auth_is_profile_scoped(monkeypatch, _isolate_hermes_home):
    import hermes_state
    import hermes_cli.profiles as profiles_mod
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    root = get_hermes_home()
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", root / "state.db")
    monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)
    query_secret = "profile-builder-query-secret"
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    response = client.post("/api/profiles", json={
        "name": "builder-auth",
        "mcp_servers": [{
            "name": "Query Server",
            "url": "https://example.com/query-mcp",
            "auth": "query",
            "bearer_token": query_secret,
        }],
    })

    assert response.status_code == 200
    assert response.json()["mcp_written"] == 1
    profile_dir = root / "profiles" / "builder-auth"
    config_text = (profile_dir / "config.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    assert config["mcp_servers"]["Query Server"] == {
        "url": "https://example.com/query-mcp",
        "auth": "query",
        "token": "${MCP_QUERY_SERVER_API_KEY}",
    }
    assert query_secret not in config_text
    profile_env = (profile_dir / ".env").read_text(encoding="utf-8")
    assert f"MCP_QUERY_SERVER_API_KEY={query_secret}" in profile_env
    assert not (root / ".env").exists()
