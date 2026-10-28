"""POST /api/dashboard/agent-plugins/install clones over https:// only, like plugins.manage install."""

import pytest


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_install_endpoint_refuses_non_https_sources_and_passes_https(client, monkeypatch):
    from hermes_cli import plugins_cmd
    seen = []
    monkeypatch.setattr(plugins_cmd, "dashboard_install_plugin",
                        lambda ident, **k: seen.append(ident) or {"ok": False, "error": "stub"})

    for identifier in ("file:///srv/plugin", "ssh://git@example.com/a/p.git", "git@example.com:a/p.git",
                       "http://example.com/a/p.git"):
        resp = client.post("/api/dashboard/agent-plugins/install", json={"identifier": identifier})
        assert resp.status_code == 400 and "https://" in resp.json()["detail"], identifier
    assert seen == []

    client.post("/api/dashboard/agent-plugins/install", json={"identifier": "https://example.com/a/p.git"})
    assert seen == ["https://example.com/a/p.git"]
