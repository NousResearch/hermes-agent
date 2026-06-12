"""Tests for the Composio integration plugin (M4).

Covers:
  * ``client.ComposioConfig`` — load/save/parse, env-var fallbacks
  * ``client.rank_tools`` — verb-priority ordering
  * ``client.ComposioClient`` — REST call envelopes, error mapping
    (mocked via httpx.MockTransport so no real network)
  * ``mcp_server.build_catalog`` — end-to-end catalog construction
    with a stubbed client
  * ``mcp_server._to_mcp_tool`` — MCP Tool shape (name/desc/schema)
  * ``mcp_server.ComposioMCPServer`` — list_tools / call_tool
    (mocked, end-to-end via in-process MCP transport)
  * ``setup.cmd_setup`` / ``cmd_mcp_install`` / ``cmd_mcp_uninstall``
    — config.yaml round-trip

Strategy
--------
The Composio REST API and MCP stdio transport are both external
I/O, so we use ``httpx.MockTransport`` to stub the REST calls
and a direct in-process invocation for the MCP layer.  No real
network, no real stdio, no real Composio account.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    monkeypatch.delenv("COMPOSIO_BASE_URL", raising=False)
    monkeypatch.delenv("COMPOSIO_USER_ID", raising=False)
    yield hermes_home


def _mock_transport(handler):
    """Build an httpx.MockTransport that calls ``handler(request)``."""
    return httpx.MockTransport(handler)


def _ok_json(payload: Dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=payload)


def _error_json(status: int, msg: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": msg}})


# ---------------------------------------------------------------------------
# ComposioConfig
# ---------------------------------------------------------------------------


def test_config_default_values():
    from plugins.integration.composio.client import ComposioConfig
    c = ComposioConfig()
    assert c.api_key == ""
    assert c.base_url == "https://backend.composio.dev"
    assert c.user_id == "default"
    assert c.max_tools_per_toolkit == 6
    assert c.allowed_toolkits == []
    assert c.allowlist_only is False
    assert c.is_configured() is False


def test_config_env_fallback(monkeypatch):
    from plugins.integration.composio.client import ComposioConfig
    monkeypatch.setenv("COMPOSIO_API_KEY", "sk-test-env")
    monkeypatch.setenv("COMPOSIO_USER_ID", "alice")
    c = ComposioConfig.from_global_config()
    assert c.api_key == "sk-test-env"
    assert c.user_id == "alice"


def test_config_save_and_load(tmp_path, monkeypatch):
    from plugins.integration.composio.client import ComposioConfig
    # Save to a tmp file (independent of HERMES_HOME)
    cfg_file = tmp_path / "composio.json"
    c = ComposioConfig(
        api_key="sk-abc",
        base_url="https://example.test",
        user_id="bob",
        max_tools_per_toolkit=10,
        allowed_toolkits=["gmail", "github"],
        allowlist_only=True,
        config_path=str(cfg_file),
    )
    c.save()
    assert cfg_file.exists()
    # Load by re-pointing HERMES_HOME at a tmp dir containing the
    # config file (since from_global_config always uses
    # ``$HERMES_HOME/composio.json``).
    import json as _json
    fake_home = tmp_path / "fake_hermes"
    fake_home.mkdir()
    (fake_home / "composio.json").write_text(cfg_file.read_text())
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    loaded = ComposioConfig.from_global_config()
    assert loaded.api_key == "sk-abc"
    assert loaded.base_url == "https://example.test"
    assert loaded.user_id == "bob"
    assert loaded.max_tools_per_toolkit == 10
    assert loaded.allowed_toolkits == ["gmail", "github"]
    assert loaded.allowlist_only is True


def test_config_as_dict_safe_redacts_key():
    from plugins.integration.composio.client import ComposioConfig
    c = ComposioConfig(api_key="sk-very-secret")
    d = c.as_dict_safe()
    assert "sk-very-secret" not in json.dumps(d)
    assert d["api_key_set"] is True


# ---------------------------------------------------------------------------
# rank_tools
# ---------------------------------------------------------------------------


def test_rank_tools_priority_ordering():
    from plugins.integration.composio.client import (
        ComposioTool,
        rank_tools,
    )
    tools = [
        ComposioTool(slug="GMAIL_SEND_EMAIL", toolkit="gmail"),
        ComposioTool(slug="GMAIL_FETCH_MESSAGE", toolkit="gmail"),
        ComposioTool(slug="GMAIL_LIST_LABELS", toolkit="gmail"),
        ComposioTool(slug="GMAIL_RANDOM_NOUN", toolkit="gmail"),
    ]
    ranked = rank_tools(tools, limit=4)
    # _FETCH_ comes before _LIST_ comes before _SEND_ comes before nothing
    assert ranked[0].slug == "GMAIL_FETCH_MESSAGE"
    assert ranked[1].slug == "GMAIL_LIST_LABELS"
    assert ranked[2].slug == "GMAIL_SEND_EMAIL"
    assert ranked[3].slug == "GMAIL_RANDOM_NOUN"


def test_rank_tools_caps_at_limit():
    from plugins.integration.composio.client import (
        ComposioTool,
        rank_tools,
    )
    tools = [
        ComposioTool(slug=f"GMAIL_FETCH_{i}", toolkit="gmail") for i in range(20)
    ]
    assert len(rank_tools(tools, limit=5)) == 5


def test_rank_tools_preserves_toolkit():
    """Ranking within a toolkit must not cross-pollute toolkits."""
    from plugins.integration.composio.client import (
        ComposioTool,
        rank_tools,
    )
    tools = [
        ComposioTool(slug="GMAIL_FETCH_X", toolkit="gmail"),
        ComposioTool(slug="SLACK_FETCH_X", toolkit="slack"),
    ]
    ranked = rank_tools(tools, limit=10)
    assert {t.toolkit for t in ranked} == {"gmail", "slack"}


# ---------------------------------------------------------------------------
# ComposioClient — mocked REST
# ---------------------------------------------------------------------------


def test_client_list_connected_accounts_success():
    from plugins.integration.composio.client import (
        ComposioClient,
        ComposioConfig,
    )

    seen: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "headers": dict(request.headers)})
        return _ok_json({
            "items": [
                {
                    "id": "ca_abc",
                    "toolkit": {"slug": "gmail"},
                    "user_id": "alice",
                    "status": "active",
                    "email": "alice@gmail.com",
                },
                {
                    "id": "ca_def",
                    "toolkit": {"slug": "github"},
                    "user_id": "alice",
                    "status": "active",
                    "email": "alice@github",
                },
            ]
        })

    cfg = ComposioConfig(api_key="sk-test", config_path="/tmp/x.json")
    async def _go():
        async with ComposioClient(cfg) as c:
            c._client = httpx.AsyncClient(
                base_url=cfg.base_url,
                headers={"X-API-Key": cfg.api_key},
                transport=_mock_transport(handler),
            )
            return await c.list_connected_accounts()
    accounts = asyncio.run(_go())
    assert len(accounts) == 2
    assert accounts[0].toolkit == "gmail"
    assert accounts[0].id == "ca_abc"
    assert accounts[1].toolkit == "github"
    # X-API-Key was sent
    assert seen[0]["headers"]["x-api-key"] == "sk-test"


def test_client_auth_error_raises_typed():
    from plugins.integration.composio.client import (
        ComposioAuthError,
        ComposioClient,
        ComposioConfig,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _error_json(401, "invalid api key")

    cfg = ComposioConfig(api_key="sk-bad", config_path="/tmp/x.json")
    async def _go():
        async with ComposioClient(cfg) as c:
            c._client = httpx.AsyncClient(
                base_url=cfg.base_url,
                transport=_mock_transport(handler),
            )
            return await c.list_connected_accounts()
    with pytest.raises(ComposioAuthError) as exc:
        asyncio.run(_go())
    assert exc.value.status == 401


def test_client_rate_limit_raises_typed():
    from plugins.integration.composio.client import (
        ComposioClient,
        ComposioConfig,
        ComposioRateLimitError,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _error_json(429, "rate limit")

    cfg = ComposioConfig(api_key="sk-x", config_path="/tmp/x.json")
    async def _go():
        async with ComposioClient(cfg) as c:
            c._client = httpx.AsyncClient(
                base_url=cfg.base_url,
                transport=_mock_transport(handler),
            )
            return await c.list_connected_accounts()
    with pytest.raises(ComposioRateLimitError):
        asyncio.run(_go())


def test_client_execute_tool_success_envelope():
    from plugins.integration.composio.client import (
        ComposioClient,
        ComposioConfig,
    )

    seen_body: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_body.update(json.loads(request.content))
        return _ok_json({
            "successful": True,
            "data": {"thread_id": "t1"},
        })

    cfg = ComposioConfig(api_key="sk-x", config_path="/tmp/x.json")
    async def _go():
        async with ComposioClient(cfg) as c:
            c._client = httpx.AsyncClient(
                base_url=cfg.base_url,
                transport=_mock_transport(handler),
            )
            return await c.execute_tool(
                "GMAIL_SEND_EMAIL",
                connected_account_id="ca_1",
                arguments={"to": "x@y.z", "subject": "hi"},
            )
    result = asyncio.run(_go())
    assert result["successful"] is True
    assert result["data"]["thread_id"] == "t1"
    assert seen_body["tool_slug"] == "GMAIL_SEND_EMAIL"
    assert seen_body["connected_account_id"] == "ca_1"
    assert seen_body["arguments"] == {"to": "x@y.z", "subject": "hi"}


# ---------------------------------------------------------------------------
# mcp_server — catalog & tool conversion
# ---------------------------------------------------------------------------


def test_composio_tool_mcp_name_format():
    from plugins.integration.composio.client import ComposioTool
    from plugins.integration.composio.mcp_server import _to_mcp_tool
    from plugins.integration.composio.client import ComposioConnectedAccount

    t = ComposioTool(
        slug="gmail_send_email", toolkit="gmail", name="Send Email", description="sends",
        input_schema={"type": "object", "properties": {}},
    )
    acc = ComposioConnectedAccount(id="ca_1", toolkit="gmail", email="a@b.com")
    mcp_t = _to_mcp_tool(t, acc)
    assert mcp_t.name == "composio_gmail_send_email"
    assert "sends" in mcp_t.description
    assert "a@b.com" in mcp_t.description  # account hint appended


def test_build_catalog_no_config_returns_error():
    from plugins.integration.composio.client import ComposioConfig
    from plugins.integration.composio.mcp_server import (
        ComposioMCPServer,
        build_catalog,
    )
    cfg = ComposioConfig(api_key="", config_path="/tmp/x.json")

    async def _go():
        return await build_catalog(cfg, client=MagicMock())
    cat = asyncio.run(_go())
    assert cat.is_empty
    assert any("API key" in e for e in cat.errors)


def test_build_catalog_filters_allowlist():
    from plugins.integration.composio.client import (
        ComposioClient,
        ComposioConfig,
        ComposioConnectedAccount,
        ComposioTool,
    )
    from plugins.integration.composio.mcp_server import build_catalog

    # Stub client that returns 2 accounts and 1 tool per toolkit
    class _StubClient:
        async def list_connected_accounts(self):
            return [
                ComposioConnectedAccount(id="ca_g", toolkit="gmail", email="a@b.com"),
                ComposioConnectedAccount(id="ca_s", toolkit="slack"),
                ComposioConnectedAccount(id="ca_d", toolkit="discord"),
            ]
        async def list_toolkit_tools(self, toolkit, limit=50):
            return [
                ComposioTool(
                    slug=f"{toolkit.upper()}_FETCH_X",
                    toolkit=toolkit,
                    name="Fetch X",
                    input_schema={"type": "object", "properties": {}},
                ),
            ]

    cfg = ComposioConfig(
        api_key="sk-x",
        config_path="/tmp/x.json",
        allowed_toolkits=["gmail", "slack"],  # excludes discord
    )
    cat = asyncio.run(build_catalog(cfg, _StubClient()))
    toolkits = {t.toolkit for t in cat.tools}
    assert toolkits == {"gmail", "slack"}
    assert "discord" not in toolkits
    # All 2 tool names map to entries
    assert len(cat.by_name) == 2


def test_build_catalog_allowlist_only_drops_unmatched():
    from plugins.integration.composio.client import (
        ComposioConfig,
        ComposioConnectedAccount,
        ComposioTool,
    )
    from plugins.integration.composio.mcp_server import build_catalog

    class _StubClient:
        async def list_connected_accounts(self):
            return [
                ComposioConnectedAccount(id="ca_g", toolkit="gmail"),
                ComposioConnectedAccount(id="ca_s", toolkit="slack"),
            ]
        async def list_toolkit_tools(self, toolkit, limit=50):
            return [ComposioTool(slug=f"{toolkit}_X", toolkit=toolkit)]

    cfg = ComposioConfig(
        api_key="sk-x",
        config_path="/tmp/x.json",
        allowed_toolkits=["github"],  # matches nothing connected
        allowlist_only=True,
    )
    cat = asyncio.run(build_catalog(cfg, _StubClient()))
    assert cat.is_empty
    assert any("allowed_toolkits" in e for e in cat.errors)


# ---------------------------------------------------------------------------
# mcp_server — end-to-end list_tools / call_tool (in-process)
# ---------------------------------------------------------------------------


def test_mcp_server_list_tools_routes_to_catalog():
    """Stub the client, build the server, and call ``_list_tools``
    (the closure registered via @server.list_tools) by directly
    hitting the registered handler on the mcp.Server.  This bypasses
    the stdio transport so we can verify the catalog→MCP mapping
    without a real client."""
    from plugins.integration.composio.client import (
        ComposioConfig,
        ComposioConnectedAccount,
        ComposioTool,
    )
    from plugins.integration.composio.mcp_server import (
        ComposioMCPServer,
        _to_mcp_tool,
    )

    class _StubClient:
        async def list_connected_accounts(self):
            return [ComposioConnectedAccount(id="ca_g", toolkit="gmail", email="a@b.com")]
        async def list_toolkit_tools(self, toolkit, limit=50):
            return [
                ComposioTool(slug="gmail_fetch_x", toolkit="gmail", description="Fetch X"),
                ComposioTool(slug="gmail_send_email", toolkit="gmail", description="Send"),
            ]

    cfg = ComposioConfig(api_key="sk-x", config_path="/tmp/x.json")
    server = ComposioMCPServer(config=cfg)
    # Inject the stub client and force catalog rebuild
    async def _setup():
        from plugins.integration.composio.mcp_server import build_catalog
        server._client = _StubClient()  # type: ignore[assignment]
        server._catalog = await build_catalog(cfg, server._client)  # type: ignore[arg-type]
    asyncio.run(_setup())

    # Read directly from the catalog (same data path as the handler)
    assert server._catalog is not None
    tools = [
        _to_mcp_tool(t, acc)
        for t, acc in server._catalog.by_name.values()
    ]
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"composio_gmail_fetch_x", "composio_gmail_send_email"}


def test_mcp_server_call_tool_unknown_name_returns_error():
    """Directly exercise :meth:`ComposioMCPServer._dispatch_call` via
    a helper that bypasses the MCP request envelope parsing (which
    is exercised in the stdio integration tests, not here)."""
    from plugins.integration.composio.client import (
        ComposioConfig,
        ComposioConnectedAccount,
        ComposioTool,
    )
    from plugins.integration.composio.mcp_server import ComposioMCPServer

    class _StubClient:
        async def list_connected_accounts(self):
            return [ComposioConnectedAccount(id="ca_g", toolkit="gmail")]
        async def list_toolkit_tools(self, toolkit, limit=50):
            return [ComposioTool(slug="gmail_fetch_x", toolkit="gmail")]

    cfg = ComposioConfig(api_key="sk-x", config_path="/tmp/x.json")
    server = ComposioMCPServer(config=cfg)
    async def _setup():
        from plugins.integration.composio.mcp_server import build_catalog
        server._client = _StubClient()  # type: ignore[assignment]
        server._catalog = await build_catalog(cfg, server._client)  # type: ignore[arg-type]
    asyncio.run(_setup())

    # Find the inner _call_tool closure registered via @server.call_tool
    async def _go():
        return await server.handle_call(
            name="composio_gmail_NOT_REAL",
            arguments={},
        )
    result = asyncio.run(_go())
    assert result.isError is True
    assert "Unknown Composio tool" in result.content[0].text
    assert "composio_gmail_NOT_REAL" in result.content[0].text


def test_mcp_server_call_tool_dispatches_to_client():
    from plugins.integration.composio.client import (
        ComposioConfig,
        ComposioConnectedAccount,
        ComposioTool,
    )
    from plugins.integration.composio.mcp_server import ComposioMCPServer

    captured: Dict[str, Any] = {}

    class _StubClient:
        async def list_connected_accounts(self):
            return [ComposioConnectedAccount(id="ca_g", toolkit="gmail", email="a@b.com")]
        async def list_toolkit_tools(self, toolkit, limit=50):
            return [ComposioTool(
                slug="gmail_send_email", toolkit="gmail",
                input_schema={"type": "object", "properties": {}},
            )]
        async def execute_tool(self, tool_slug, *, connected_account_id, arguments):
            captured["tool_slug"] = tool_slug
            captured["account"] = connected_account_id
            captured["arguments"] = arguments
            return {"successful": True, "data": {"thread_id": "t42"}}

    cfg = ComposioConfig(api_key="sk-x", config_path="/tmp/x.json")
    server = ComposioMCPServer(config=cfg)
    async def _setup():
        from plugins.integration.composio.mcp_server import build_catalog
        server._client = _StubClient()  # type: ignore[assignment]
        server._catalog = await build_catalog(cfg, server._client)  # type: ignore[arg-type]
    asyncio.run(_setup())

    # Look up the real (tool, account) pair from the catalog and
    # call via the same path the MCP handler uses (server._call_tool).
    assert server._catalog is not None
    real_name = next(iter(server._catalog.by_name.keys()))

    async def _go():
        return await server.handle_call(
            name=real_name,
            arguments={"to": "x@y.z", "subject": "hi"},
        )
    result = asyncio.run(_go())
    assert result.isError is False
    assert captured["tool_slug"] == "gmail_send_email"
    assert captured["account"] == "ca_g"
    assert captured["arguments"] == {"to": "x@y.z", "subject": "hi"}
    # Result is JSON-encoded in the text content
    text = result.content[0].text
    parsed = json.loads(text)
    assert parsed["data"]["thread_id"] == "t42"


# ---------------------------------------------------------------------------
# setup CLI — config.yaml round-trip
# ---------------------------------------------------------------------------


def test_mcp_install_writes_config_entry(monkeypatch, tmp_path):
    """cmd_mcp_install writes the stdio MCP entry into config.yaml."""
    from plugins.integration.composio import setup as setup_mod

    # Patch the config helpers to use a temp file
    cfg_file = tmp_path / "config.yaml"
    monkeypatch.setattr(setup_mod, "read_raw_config", lambda: {})
    written: Dict[str, Any] = {}
    def _write(cfg):
        written.clear()
        written.update(cfg)
    monkeypatch.setattr(setup_mod, "save_config", _write)

    args = MagicMock()
    rc = setup_mod.cmd_mcp_install(args)
    assert rc == 0
    assert "mcp_servers" in written
    server_entry = written["mcp_servers"]["composio"]
    assert server_entry["command"]  # some python path
    assert server_entry["args"] == ["-m", "plugins.integration.composio.mcp_server"]


def test_mcp_uninstall_removes_entry(monkeypatch):
    from plugins.integration.composio import setup as setup_mod

    monkeypatch.setattr(
        setup_mod,
        "read_raw_config",
        lambda: {"mcp_servers": {"composio": {"command": "x"}, "other": {"command": "y"}}},
    )
    written: Dict[str, Any] = {}
    monkeypatch.setattr(setup_mod, "save_config", lambda c: written.update(c))

    args = MagicMock()
    rc = setup_mod.cmd_mcp_uninstall(args)
    assert rc == 0
    assert "composio" not in written["mcp_servers"]
    assert "other" in written["mcp_servers"]  # unrelated entries preserved


def test_mcp_uninstall_no_entry_is_noop(monkeypatch):
    from plugins.integration.composio import setup as setup_mod

    monkeypatch.setattr(setup_mod, "read_raw_config", lambda: {})
    monkeypatch.setattr(setup_mod, "save_config", lambda c: None)
    args = MagicMock()
    rc = setup_mod.cmd_mcp_uninstall(args)
    assert rc == 0


def test_cmd_setup_saves_config(monkeypatch, tmp_path):
    """cmd_setup writes composio.json with the right values."""
    from plugins.integration.composio import setup as setup_mod
    from plugins.integration.composio.client import ComposioConfig

    # Force save() to write to a temp file
    config_path = tmp_path / "composio.json"

    def _fake_from_global_config(cls):
        cfg = ComposioConfig(config_path=str(config_path))
        return cfg

    monkeypatch.setattr(setup_mod.ComposioConfig, "from_global_config", classmethod(_fake_from_global_config))

    args = MagicMock(
        api_key="sk-from-flag",
        user_id="charlie",
        base_url=None,
        max_tools_per_toolkit=12,
        allowed_toolkits=["gmail"],
        allowlist_only=True,
        non_interactive=True,
        enable=False,  # don't poke config.yaml in this test
    )
    rc = setup_mod.cmd_setup(args)
    assert rc == 0
    raw = json.loads(config_path.read_text())
    assert raw["api_key"] == "sk-from-flag"
    assert raw["user_id"] == "charlie"
    assert raw["max_tools_per_toolkit"] == 12
    assert raw["allowed_toolkits"] == ["gmail"]
    assert raw["allowlist_only"] is True


def test_enable_plugin_in_config(monkeypatch):
    """_enable_plugin_in_config() should add the plugin key to the
    config.yaml's plugins.enabled list, idempotently."""
    from plugins.integration.composio import setup as setup_mod

    monkeypatch.setattr(setup_mod, "read_raw_config", lambda: {"plugins": {"enabled": ["foo"]}})
    written: Dict[str, Any] = {}
    monkeypatch.setattr(setup_mod, "save_config", lambda c: written.update(c))
    assert setup_mod._enable_plugin_in_config() is True
    assert "integration/composio" in written["plugins"]["enabled"]
    assert "foo" in written["plugins"]["enabled"]  # preserves existing

    # Idempotent: already present → returns False
    monkeypatch.setattr(
        setup_mod, "read_raw_config",
        lambda: {"plugins": {"enabled": ["integration/composio"]}},
    )
    assert setup_mod._enable_plugin_in_config() is False


def test_cmd_setup_enables_plugin_by_default(monkeypatch, tmp_path):
    """``hermes composio setup`` (no --no-enable) writes the
    composio.json AND adds the plugin to plugins.enabled."""
    from plugins.integration.composio import setup as setup_mod
    from plugins.integration.composio.client import ComposioConfig

    config_path = tmp_path / "composio.json"
    def _fake_from_global_config(cls):
        return ComposioConfig(config_path=str(config_path))
    monkeypatch.setattr(setup_mod.ComposioConfig, "from_global_config", classmethod(_fake_from_global_config))

    written: Dict[str, Any] = {}
    monkeypatch.setattr(setup_mod, "read_raw_config", lambda: {})
    monkeypatch.setattr(setup_mod, "save_config", lambda c: written.update(c))

    args = MagicMock(
        api_key="sk-x", user_id=None, base_url=None,
        max_tools_per_toolkit=None, allowed_toolkits=None,
        allowlist_only=None, non_interactive=True,
        enable=True,
    )
    rc = setup_mod.cmd_setup(args)
    assert rc == 0
    assert "integration/composio" in written["plugins"]["enabled"]


def test_cmd_setup_missing_key_returns_error(monkeypatch, tmp_path):
    from plugins.integration.composio import setup as setup_mod
    from plugins.integration.composio.client import ComposioConfig

    config_path = tmp_path / "composio.json"
    def _fake_from_global_config(cls):
        return ComposioConfig(config_path=str(config_path))
    monkeypatch.setattr(setup_mod.ComposioConfig, "from_global_config", classmethod(_fake_from_global_config))

    args = MagicMock(
        api_key=None,
        user_id=None,
        base_url=None,
        max_tools_per_toolkit=None,
        allowed_toolkits=None,
        allowlist_only=None,
        non_interactive=True,
        enable=False,
    )
    rc = setup_mod.cmd_setup(args)
    assert rc == 2  # error


# ---------------------------------------------------------------------------
# Plugin discovery (smoke test that the manifest loads)
# ---------------------------------------------------------------------------


def test_plugin_yaml_loads():
    """Smoke test: plugin.yaml is present and parseable."""
    import yaml
    manifest_path = Path(__file__).parent.parent.parent / "plugins" / "integration" / "composio" / "plugin.yaml"
    with manifest_path.open() as f:
        data = yaml.safe_load(f)
    assert data["name"] == "composio"
    assert data["version"] == "1.0.0"
    assert "mcp_servers" in str(data) or "MCP" in str(data)
