"""Pipedream Apps backend: the Desktop Apps tab's server door.

Every test drives the router through a mocked Pipedream transport, so the suite
never reaches the network and no real credential is required. The assertions pin
the parts the renderer depends on: cursor pagination, credential stripping,
ownership-checked deletes, and the Connect Link's app parameter.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PLUGIN_FILE = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "pipedream-apps"
    / "dashboard"
    / "plugin_api.py"
)
PREFIX = "/api/plugins/pipedream-apps"


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "hermes_pipedream_apps_api_test", PLUGIN_FILE
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AccountTransport:
    """Pipedream transport whose account list changes on delete, like the real API."""

    def __init__(self, accounts: list[dict]):
        self.requests: list[httpx.Request] = []
        self.deletes: list[str] = []
        self.accounts = [dict(account) for account in accounts]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/oauth/token":
            self.requests.append(request)
            return httpx.Response(
                200, json={"access_token": "at_test", "expires_in": 3600}
            )
        if request.method == "GET" and path.endswith("/accounts"):
            self.requests.append(request)
            return httpx.Response(200, json=self.accounts)
        if request.method == "DELETE" and "/accounts/" in path:
            self.requests.append(request)
            account_id = path.rsplit("/", 1)[-1]
            self.deletes.append(account_id)
            self.accounts = [
                account for account in self.accounts if account.get("id") != account_id
            ]
            return httpx.Response(204)
        self.requests.append(request)
        return httpx.Response(404, json={"error": "not found"})


class RecordedTransport:
    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.deletes: list[str] = []
        self.token_calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        if path == "/v1/oauth/token":
            self.token_calls += 1
            return httpx.Response(
                200, json={"access_token": "at_test", "expires_in": 3600}
            )

        if path == "/v1/connect/apps":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "app_1",
                            "name_slug": "slack",
                            "name": "Slack",
                            "auth_type": "oauth",
                            "description": "Team chat",
                            "img_src": "https://example.com/slack.png",
                            "categories": ["Chat"],
                            "featured_weight": 10,
                            "unexpected": "drop",
                        }
                    ],
                    "page_info": {
                        "count": 1,
                        "total_count": 42,
                        "start_cursor": None,
                        "end_cursor": "cursor_2",
                    },
                },
            )

        if path == "/v1/connect/app_categories":
            return httpx.Response(
                200,
                json=[{"id": "appcat_1", "name": "Chat", "description": "Chat apps"}],
            )

        if path == "/v1/connect/apps/slack":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "name_slug": "slack",
                        "name": "Slack",
                        "unexpected": "drop",
                    }
                },
            )

        if path.endswith("/tokens"):
            return httpx.Response(
                200,
                json={
                    "token": "ctok_test",
                    "connect_link_url": (
                        "https://pipedream.com/_static/connect.html"
                        "?token=ctok_test&connectLink=true"
                    ),
                    "expires_at": "2030-01-01T00:00:00Z",
                },
            )

        if request.method == "GET" and path.endswith("/accounts"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "apn_1",
                        "name": "me@example.com",
                        "external_id": "user-1",
                        "healthy": True,
                        "dead": False,
                        "created_at": "2030-01-01T00:00:00Z",
                        "app": {
                            "name_slug": "slack",
                            "name": "Slack",
                            "unexpected": "drop",
                        },
                        "credentials": {
                            "email": "me@example.com",
                            "oauth_access_token": "secret",
                        },
                        "unexpected": "drop",
                    }
                ],
            )

        if request.method == "DELETE" and "/accounts/" in path:
            self.deletes.append(path.rsplit("/", 1)[-1])
            return httpx.Response(204)

        return httpx.Response(404, json={"error": "not found"})


@pytest.fixture
def plugin(monkeypatch):
    module = _load_plugin()
    module._TOKEN.update({"access_token": "", "expires_at": 0.0})
    yield module
    module._transport = None


@pytest.fixture
def client(plugin):
    app = FastAPI()
    app.include_router(plugin.router, prefix=PREFIX)
    return TestClient(app)


def _install(plugin, transport: RecordedTransport) -> None:
    plugin._transport = httpx.MockTransport(transport)


def _request_json(request: httpx.Request) -> dict:
    return json.loads(request.content.decode("utf-8"))


def test_config_reports_readiness_without_secrets(plugin, client):
    body = client.get(f"{PREFIX}/config").json()

    assert body["configured"] is True
    assert body["environment"] == "production"
    assert body["mcpBaseUrl"] == "https://agent.unitalk.ai/api/pipedream/mcp"


def test_config_unconfigured_blocks_api(plugin, client, monkeypatch):
    for key in ("_CLIENT_ID", "_CLIENT_SECRET", "_PROJECT_ID"):
        monkeypatch.setattr(plugin, key, "")

    body = client.get(f"{PREFIX}/config").json()
    assert body["configured"] is False
    assert body["mcpBaseUrl"] is None
    assert client.get(f"{PREFIX}/apps").status_code == 503


def test_config_builds_the_mcp_proxy_prefix(plugin, client, monkeypatch):
    monkeypatch.setattr(plugin, "_MCP_BASE", "https://app.example")

    assert (
        client.get(f"{PREFIX}/config").json()["mcpBaseUrl"]
        == "https://app.example/api/pipedream/mcp"
    )


def test_apps_map_the_official_payload_and_cache_the_token(plugin, client):
    recorder = RecordedTransport()
    _install(plugin, recorder)

    response = client.get(
        f"{PREFIX}/apps",
        params={"q": "slack", "limit": 10, "category": "appcat_1", "after": "cursor_1"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["apps"][0]["name"] == "Slack"
    assert body["apps"][0]["categories"] == ["Chat"]
    assert "unexpected" not in body["apps"][0]
    assert body["nextCursor"] == "cursor_2"
    assert body["totalCount"] == 42

    upstream = next(r for r in recorder.requests if r.url.path == "/v1/connect/apps")
    assert "has_components" not in upstream.url.params
    assert upstream.url.params.get("q") == "slack"
    assert upstream.url.params.get("limit") == "10"
    assert upstream.url.params.get("category_ids") == "appcat_1"
    assert upstream.url.params.get("after") == "cursor_1"
    assert upstream.headers.get("authorization") == "Bearer at_test"
    assert upstream.headers.get("x-pd-environment") == "production"

    token_request = next(
        r for r in recorder.requests if r.url.path == "/v1/oauth/token"
    )
    assert "connect:tokens:create" in _request_json(token_request)["scope"]

    assert client.get(f"{PREFIX}/apps", params={"limit": 10}).status_code == 200
    assert recorder.token_calls == 1


def test_categories(plugin, client):
    _install(plugin, RecordedTransport())

    body = client.get(f"{PREFIX}/categories").json()

    assert body["categories"] == [
        {"id": "appcat_1", "name": "Chat", "description": "Chat apps"}
    ]


def test_retrieve_app_strips_unknown_fields(plugin, client):
    _install(plugin, RecordedTransport())

    body = client.get(f"{PREFIX}/apps/slack").json()

    assert body == {"name_slug": "slack", "name": "Slack"}


def test_connect_appends_the_app_to_the_connect_link(plugin, client):
    recorder = RecordedTransport()
    _install(plugin, recorder)

    response = client.post(
        f"{PREFIX}/connect",
        json={"externalUserId": "user-1", "appSlug": "slack"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    query = dict(parse_qsl(urlsplit(body["connectLinkUrl"]).query))
    assert query["app"] == "slack"
    assert query["token"] == "ctok_test"

    upstream = next(r for r in recorder.requests if r.url.path.endswith("/tokens"))
    payload = _request_json(upstream)
    assert payload["external_user_id"] == "user-1"
    assert "connect:accounts:write" in payload["scope"]
    assert "connect:tokens:create" not in payload["scope"]


def test_accounts_strip_credentials_and_enrich_email(plugin, client):
    recorder = RecordedTransport()
    _install(plugin, recorder)

    response = client.get(f"{PREFIX}/accounts", params={"externalUserId": "user-1"})

    assert response.status_code == 200, response.text
    accounts = response.json()
    assert accounts[0]["id"] == "apn_1"
    assert accounts[0]["email"] == "me@example.com"
    assert "credentials" not in accounts[0]
    assert "unexpected" not in accounts[0]
    assert accounts[0]["app"] == {"name_slug": "slack", "name": "Slack"}

    upstream = next(r for r in recorder.requests if r.url.path.endswith("/accounts"))
    assert upstream.url.params.get("include_credentials") == "true"


def test_delete_checks_ownership_before_deleting(plugin, client):
    recorder = RecordedTransport()
    _install(plugin, recorder)

    not_owned = client.delete(
        f"{PREFIX}/accounts/apn_other", params={"externalUserId": "user-1"}
    )
    assert not_owned.status_code == 404
    assert recorder.deletes == []

    owned = client.delete(
        f"{PREFIX}/accounts/apn_1", params={"externalUserId": "user-1"}
    )
    assert owned.status_code == 200, owned.text
    assert recorder.deletes == ["apn_1"]


def test_input_validation(plugin, client):
    _install(plugin, RecordedTransport())

    assert client.get(f"{PREFIX}/accounts").status_code == 400
    assert (
        client.get(
            f"{PREFIX}/accounts", params={"externalUserId": "x" * 251}
        ).status_code
        == 400
    )
    assert (
        client.post(
            f"{PREFIX}/connect",
            json={"externalUserId": "user-1", "appSlug": "bad slug"},
        ).status_code
        == 400
    )
    assert (
        client.delete(
            f"{PREFIX}/accounts/not-an-account", params={"externalUserId": "user-1"}
        ).status_code
        == 400
    )


def test_upstream_credential_failure_is_reported_without_leaking(plugin, client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth/token":
            return httpx.Response(401, json={"error": "invalid_client"})
        return httpx.Response(500)

    plugin._transport = httpx.MockTransport(handler)

    response = client.get(f"{PREFIX}/apps")

    assert response.status_code == 502
    assert "credentials" in response.json()["detail"].lower()


def _set_mcp_base(plugin, monkeypatch, base: str = "https://app.example") -> str:
    monkeypatch.setattr(plugin, "_MCP_BASE", base)
    return f"{base}/api/pipedream/mcp"


def _mcp_servers() -> dict:
    from hermes_cli.config import load_config

    return load_config().get("mcp_servers") or {}


def _seed_mcp_server(name: str, url: str, *, enabled: bool = True) -> None:
    from hermes_cli.config import load_config, save_config

    config = load_config()
    config.setdefault("mcp_servers", {})[name] = {"url": url, "enabled": enabled}
    save_config(config)


def _account(account_id: str, app_slug: str = "slack") -> dict:
    return {
        "id": account_id,
        "name": "me@example.com",
        "external_id": "user-1",
        "healthy": True,
        "dead": False,
        "app": {"name_slug": app_slug, "name": app_slug.title()},
    }


def test_mcp_enable_requires_the_proxy_base(plugin, client, monkeypatch):
    monkeypatch.setattr(plugin, "_MCP_BASE", "")
    response = client.post(
        f"{PREFIX}/mcp/enable",
        json={"externalUserId": "user-1", "appSlug": "slack"},
    )

    assert response.status_code == 503
    assert _mcp_servers() == {}


def test_mcp_enable_adds_an_enabled_server(plugin, client, monkeypatch):
    base = _set_mcp_base(plugin, monkeypatch)

    response = client.post(
        f"{PREFIX}/mcp/enable",
        json={"externalUserId": "user-1", "appSlug": "slack"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["action"] == "added"
    entry = _mcp_servers()["pipedream_slack"]
    assert entry["url"] == f"{base}/user-1/slack"
    assert entry.get("enabled", True) is not False


def test_mcp_enable_is_idempotent_and_reenables(plugin, client, monkeypatch):
    base = _set_mcp_base(plugin, monkeypatch)
    url = f"{base}/user-1/slack"
    _seed_mcp_server("pipedream_slack", url, enabled=False)

    response = client.post(
        f"{PREFIX}/mcp/enable",
        json={"externalUserId": "user-1", "appSlug": "slack"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["action"] == "enabled"
    assert _mcp_servers()["pipedream_slack"]["enabled"] is True

    again = client.post(
        f"{PREFIX}/mcp/enable",
        json={"externalUserId": "user-1", "appSlug": "slack"},
    )
    assert again.status_code == 200, again.text
    assert again.json()["action"] == "unchanged"


def test_mcp_enable_replaces_a_stale_owned_url(plugin, client, monkeypatch):
    base = _set_mcp_base(plugin, monkeypatch)
    _seed_mcp_server("pipedream_slack", f"{base}/old-user/slack", enabled=True)

    response = client.post(
        f"{PREFIX}/mcp/enable",
        json={"externalUserId": "user-1", "appSlug": "slack"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["action"] == "replaced"
    assert _mcp_servers()["pipedream_slack"]["url"] == f"{base}/user-1/slack"


def test_mcp_enable_leaves_a_foreign_url_untouched(plugin, client, monkeypatch):
    _set_mcp_base(plugin, monkeypatch)
    foreign = "https://elsewhere.example/mcp"
    _seed_mcp_server("pipedream_slack", foreign, enabled=True)

    response = client.post(
        f"{PREFIX}/mcp/enable",
        json={"externalUserId": "user-1", "appSlug": "slack"},
    )

    assert response.status_code == 409
    assert "different URL" in response.json()["detail"]
    assert _mcp_servers()["pipedream_slack"]["url"] == foreign


def test_disconnect_disables_the_mcp_server_when_last_account_leaves(
    plugin, client, monkeypatch
):
    base = _set_mcp_base(plugin, monkeypatch)
    _seed_mcp_server("pipedream_slack", f"{base}/user-1/slack", enabled=True)
    _install(plugin, AccountTransport([_account("apn_1")]))

    response = client.delete(
        f"{PREFIX}/accounts/apn_1", params={"externalUserId": "user-1"}
    )

    assert response.status_code == 200, response.text
    assert _mcp_servers()["pipedream_slack"]["enabled"] is False


def test_disconnect_keeps_the_mcp_server_while_another_account_remains(
    plugin, client, monkeypatch
):
    base = _set_mcp_base(plugin, monkeypatch)
    _seed_mcp_server("pipedream_slack", f"{base}/user-1/slack", enabled=True)
    _install(plugin, AccountTransport([_account("apn_1"), _account("apn_2")]))

    response = client.delete(
        f"{PREFIX}/accounts/apn_1", params={"externalUserId": "user-1"}
    )

    assert response.status_code == 200, response.text
    assert _mcp_servers()["pipedream_slack"]["enabled"] is True


def test_disconnect_never_disables_a_foreign_entry(plugin, client, monkeypatch):
    _set_mcp_base(plugin, monkeypatch)
    foreign = "https://elsewhere.example/mcp"
    _seed_mcp_server("pipedream_slack", foreign, enabled=True)
    _install(plugin, AccountTransport([_account("apn_1")]))

    response = client.delete(
        f"{PREFIX}/accounts/apn_1", params={"externalUserId": "user-1"}
    )

    assert response.status_code == 200, response.text
    assert _mcp_servers()["pipedream_slack"]["enabled"] is True
