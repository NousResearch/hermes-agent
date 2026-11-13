"""The PayPal catalog entry must use its Streamable HTTP endpoint with OAuth."""

import asyncio
from unittest.mock import AsyncMock, Mock


def test_paypal_catalog_connects_with_streamable_http_oauth(monkeypatch):
    from hermes_cli.mcp_catalog import _build_server_config, get_entry
    from tools import mcp_oauth_manager, mcp_tool

    monkeypatch.delenv("HERMES_OPTIONAL_MCPS", raising=False)
    entry = get_entry("paypal")
    assert entry is not None
    config = _build_server_config(entry, None)
    assert config == {"url": "https://mcp.paypal.com/mcp", "auth": "oauth"}

    server = mcp_tool.MCPServerTask("paypal")
    server._auth_type = config["auth"]
    provider = object()
    manager = Mock()
    manager.get_or_build_provider.return_value = provider
    monkeypatch.setattr(mcp_oauth_manager, "get_manager", lambda: manager)
    transport = Mock(return_value=object())
    monkeypatch.setattr(server, "_streamable_http_transport", transport)
    serve = AsyncMock(return_value="shutdown")
    monkeypatch.setattr(server, "_serve_transport", serve)
    assert asyncio.run(server._run_http(config)) == "shutdown"
    manager.get_or_build_provider.assert_called_once_with("paypal", config["url"], None)
    assert transport.call_args.args[0] == config["url"]
    assert transport.call_args.args[5] is provider
    serve.assert_awaited_once()
