"""Behavior contracts for the remote Streamable HTTP MCP server."""

import pytest


def test_remote_http_requires_auth_and_host_allowlist(monkeypatch):
    import mcp_serve

    monkeypatch.delenv("TEST_MCP_TOKEN", raising=False)
    with pytest.raises(SystemExit) as missing_token:
        mcp_serve.run_mcp_server(
            transport="http", host="0.0.0.0", token_env="TEST_MCP_TOKEN",
            allowed_hosts=["mcp.example.com:*"],
        )
    assert missing_token.value.code == 2

    monkeypatch.setenv("TEST_MCP_TOKEN", "secret")
    with pytest.raises(SystemExit) as missing_host:
        mcp_serve.run_mcp_server(
            transport="http", host="0.0.0.0", token_env="TEST_MCP_TOKEN",
        )
    assert missing_host.value.code == 2

    pytest.importorskip("mcp")
    from starlette.testclient import TestClient

    server = mcp_serve.create_mcp_server(
        bearer_token="secret",
        resource_url="https://mcp.example.com/mcp",
    )
    app = server.streamable_http_app(host="127.0.0.1")
    request = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2026-07-28", "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }
    headers = {"Accept": "application/json, text/event-stream", "Host": "localhost:8000"}
    with TestClient(app) as client:
        assert client.post("/mcp", json=request, headers=headers).status_code == 401
        authorized = client.post(
            "/mcp", json=request,
            headers={**headers, "Authorization": "Bearer secret"},
        )
    assert authorized.status_code == 200


def test_remote_http_uses_bearer_auth_and_preserves_transport_settings(monkeypatch):
    import mcp_serve

    calls = {}

    class Bridge:
        def start(self):
            calls["started"] = True

        def stop(self):
            calls["stopped"] = True

    class Server:
        async def run_streamable_http_async(self, **kwargs):
            calls["http"] = kwargs

    def create_server(**kwargs):
        calls["create"] = kwargs
        return Server()

    monkeypatch.setenv("TEST_MCP_TOKEN", "secret")
    monkeypatch.setattr(mcp_serve, "EventBridge", Bridge)
    monkeypatch.setattr(mcp_serve, "create_mcp_server", create_server)

    mcp_serve.run_mcp_server(
        transport="http",
        host="0.0.0.0",
        port=9000,
        path="/remote-mcp",
        token_env="TEST_MCP_TOKEN",
        allowed_hosts=["mcp.example.com:*"],
        public_url="https://mcp.example.com/remote-mcp",
    )

    assert calls["started"] is True
    assert calls["stopped"] is True
    assert calls["create"]["bearer_token"] == "secret"
    assert calls["create"]["resource_url"] == "https://mcp.example.com/remote-mcp"
    assert calls["http"]["host"] == "0.0.0.0"
    assert calls["http"]["port"] == 9000
    assert calls["http"]["streamable_http_path"] == "/remote-mcp"
    assert calls["http"]["transport_security"].allowed_hosts == ["mcp.example.com:*"]