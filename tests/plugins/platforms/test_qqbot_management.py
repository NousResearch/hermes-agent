"""Menu/panel REST behavior with a real HTTP client and fake transport."""
import asyncio

import httpx
import pytest

from plugins.platforms.qqbot.management import QQBotManagementClient, read_credentials


def test_authenticated_endpoints_and_one_page_passthrough():
    requests = []
    page = {"panels": [], "next_cursor": "next", "is_end": False, "future": True}

    def respond(request):
        requests.append(request)
        if request.url.host == "bots.qq.com":
            return httpx.Response(200, json={"access_token": "test-token"})
        return httpx.Response(200, json=page)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = QQBotManagementClient(http, "app", "secret")
            assert await client.list_panels("future scope", cursor="a&b", limit=73) == page
            await client.get_global_menu()
            await client.update_global_menu({"menu": {}, "future": [1]})
            await client.create_panel({})
            await client.get_panel("a/b?#")
            await client.update_panel("a/b?#", {"panel": {}})
            await client.update_panel_targets("a/b?#", {"op": "future"})
            await client.delete_panel("a/b?#")
    asyncio.run(exercise())
    assert len(requests) == 9  # one authentication, eight explicit operations; no pagination
    assert dict(requests[1].url.params) == {"scope": "future scope", "cursor": "a&b", "limit": "73"}
    assert requests[5].url.raw_path == b"/v2/panels/a%2Fb%3F%23"
    assert requests[-2].url.raw_path.endswith(b"/target")
    assert requests[-1].method == "DELETE"
    assert requests[2].headers["Authorization"] == "QQBot test-token"
    assert requests[2].headers["X-Union-Appid"] == "app"
    assert requests[3].content == b'{"menu":{},"future":[1]}'


@pytest.mark.parametrize("status,body", [(403, {}), (200, {"code": 123}), (200, [1])])
@pytest.mark.parametrize("authentication", [True, False])
def test_failures_are_not_retried(status, body, authentication):
    calls = []
    def respond(request):
        calls.append(request)
        if not authentication and request.url.host == "bots.qq.com":
            return httpx.Response(200, json={"access_token": "token"})
        return httpx.Response(status, json=body)
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            with pytest.raises((RuntimeError, httpx.HTTPStatusError)):
                await QQBotManagementClient(http, "app", "secret").get_global_menu()
    asyncio.run(exercise())
    assert len(calls) == (1 if authentication else 2)


@pytest.mark.parametrize("panel_id", ["", ".", ".."])
@pytest.mark.parametrize("operation", ["get_panel", "delete_panel"])
def test_invalid_panel_ids_never_reach_http(panel_id, operation):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"access_token": "token"})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = QQBotManagementClient(http, "app", "secret")
            with pytest.raises(ValueError, match="panel_id"):
                await getattr(client, operation)(panel_id)

    asyncio.run(exercise())
    assert calls == []


def test_empty_delete_response():
    def respond(request):
        if request.url.host == "bots.qq.com":
            return httpx.Response(200, json={"access_token": "token"})
        return httpx.Response(204)
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            assert await QQBotManagementClient(http, "app", "secret").delete_panel("id") == {}
    asyncio.run(exercise())


def test_credentials_read_real_profile_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "platforms:\n  qqbot:\n    extra:\n      app_id: configured-app\n      client_secret: configured-secret\n", encoding="utf-8")
    assert read_credentials() == ("configured-app", "configured-secret")


def test_credentials_use_profile_scope(monkeypatch):
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    from gateway.config import GatewayConfig
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: GatewayConfig())
    monkeypatch.setenv("QQ_APP_ID", "wrong")
    monkeypatch.setenv("QQ_CLIENT_SECRET", "wrong")
    token = set_secret_scope({"QQ_APP_ID": "owner", "QQ_CLIENT_SECRET": "owner-secret"})
    try:
        assert read_credentials() == ("owner", "owner-secret")
    finally:
        reset_secret_scope(token)
