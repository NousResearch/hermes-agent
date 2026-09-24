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

    monkeypatch.delenv("TEST_MCP_TOKEN", raising=False)
    with pytest.raises(SystemExit) as proxied_without_token:
        mcp_serve.run_mcp_server(
            transport="http",
            host="127.0.0.1",
            token_env="TEST_MCP_TOKEN",
            allowed_hosts=["mcp.example.com:*"],
            public_url="https://mcp.example.com/mcp",
        )
    assert proxied_without_token.value.code == 2

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


def test_wildcard_http_bind_requires_client_reachable_public_url(monkeypatch, capsys):
    import mcp_serve

    monkeypatch.setenv("TEST_MCP_TOKEN", "secret")

    with pytest.raises(SystemExit) as missing_public_url:
        mcp_serve.run_mcp_server(
            transport="http",
            host="0.0.0.0",
            token_env="TEST_MCP_TOKEN",
            allowed_hosts=["mcp.example.com:*"],
        )

    assert missing_public_url.value.code == 2
    assert "wildcard MCP HTTP binds require --public-url" in capsys.readouterr().err


@pytest.mark.parametrize(
    "path",
    [
        "mcp",
        "/m cp",
        "/mcp?x=1",
        "/mcp#part",
        "/mcp/{id}",
        "/mcp%20x",
        "/mcp%3Fx",
        "/mcp%2Fchild",
    ],
)
def test_http_rejects_unusable_route_path_before_starting_bridge(
    monkeypatch, capsys, path,
):
    import mcp_serve

    starts = []

    class Bridge:
        def start(self):
            starts.append(True)

    monkeypatch.setattr(mcp_serve, "EventBridge", Bridge)
    with pytest.raises(SystemExit) as invalid_path:
        mcp_serve.run_mcp_server(transport="http", path=path)

    assert invalid_path.value.code == 2
    assert "absolute literal path" in capsys.readouterr().err
    assert starts == []


@pytest.mark.parametrize("port", [-1, 0, 65536])
def test_http_rejects_invalid_bind_port_before_starting_bridge(
    monkeypatch, capsys, port,
):
    import mcp_serve

    starts = []

    class Bridge:
        def start(self):
            starts.append(True)

    monkeypatch.setattr(mcp_serve, "EventBridge", Bridge)
    with pytest.raises(SystemExit) as invalid_port:
        mcp_serve.run_mcp_server(transport="http", port=port)

    assert invalid_port.value.code == 2
    assert "port must be between 1 and 65535" in capsys.readouterr().err
    assert starts == []


@pytest.mark.parametrize(
    "public_url",
    [
        "mcp.example.com/mcp",
        "http://[bad/mcp",
        "https://mcp.example.com:bad/mcp",
        "https://mcp example.com/mcp",
        "https://mcp.example.com:99999/mcp",
    ],
)
def test_http_rejects_invalid_public_url_before_starting_bridge(
    monkeypatch, capsys, public_url,
):
    import mcp_serve

    starts = []

    class Bridge:
        def start(self):
            starts.append(True)

    monkeypatch.setattr(mcp_serve, "EventBridge", Bridge)
    monkeypatch.setenv("TEST_MCP_TOKEN", "secret")
    with pytest.raises(SystemExit) as invalid_url:
        mcp_serve.run_mcp_server(
            transport="http",
            host="0.0.0.0",
            token_env="TEST_MCP_TOKEN",
            allowed_hosts=["mcp.example.com:*"],
            public_url=public_url,
        )

    assert invalid_url.value.code == 2
    assert "absolute http:// or https:// URL" in capsys.readouterr().err
    assert starts == []


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


def test_authenticated_http_brackets_ipv6_resource_host(monkeypatch):
    import mcp_serve

    calls = {}

    class Bridge:
        def start(self):
            pass

        def stop(self):
            pass

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
        host="::1",
        port=9000,
        token_env="TEST_MCP_TOKEN",
    )

    assert calls["create"]["bearer_token"] == "secret"
    assert calls["create"]["resource_url"] == "http://[::1]:9000/mcp"
    assert calls["http"]["host"] == "::1"


@pytest.mark.parametrize(
    ("host", "allowed_hosts"),
    [
        ("LOCALHOST", ["127.0.0.1:*", "localhost:*", "[::1]:*"]),
        ("127.0.0.1", ["mcp.example.com:*"]),
    ],
)
def test_loopback_http_enforces_explicit_transport_security(
    monkeypatch, host, allowed_hosts,
):
    import mcp_serve

    calls = {}

    class Bridge:
        def start(self):
            pass

        def stop(self):
            pass

    class Server:
        async def run_streamable_http_async(self, **kwargs):
            calls["http"] = kwargs

    monkeypatch.setattr(mcp_serve, "EventBridge", Bridge)
    monkeypatch.setattr(mcp_serve, "create_mcp_server", lambda **kwargs: Server())

    supplied_hosts = None if host == "LOCALHOST" else allowed_hosts
    if supplied_hosts:
        monkeypatch.setenv("TEST_MCP_TOKEN", "secret")
    mcp_serve.run_mcp_server(
        transport="http",
        host=host,
        allowed_hosts=supplied_hosts,
        token_env="TEST_MCP_TOKEN",
    )

    assert calls["http"]["host"] == host.lower()
    security = calls["http"]["transport_security"]
    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == allowed_hosts
    assert security.allowed_origins == []


@pytest.mark.asyncio
async def test_static_token_verifier_handles_unicode_as_unauthorized():
    import mcp_serve

    verifier, _ = mcp_serve._http_auth("secret", "https://mcp.example.com/mcp")

    assert await verifier.verify_token("sëcret") is None
