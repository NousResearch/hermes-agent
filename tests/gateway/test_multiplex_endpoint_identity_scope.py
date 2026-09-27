"""Invariant: an adapter's endpoint/identity resolves through the SAME profile scope as its secret.

Under ``gateway.multiplex_profiles`` ``os.environ`` holds the DEFAULT profile's values; a secondary
profile's ``.env`` exists only in its secret scope. Every adapter here already read its *secret*
scope-aware but read the paired homeserver / URL / client_id / from-number / webhook URL raw from
``os.environ`` — so the secondary's credential was sent to the default profile's host or identity.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import secret_scope as ss
from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

DEFAULT_ENV = {
    "MATRIX_HOMESERVER": "https://default.matrix.example", "MATRIX_USER_ID": "@default:example",
    "MATRIX_DEVICE_ID": "DEFAULTDEV", "MATRIX_ACCESS_TOKEN": "default-matrix-token",
    "HASS_URL": "http://default-ha.example:8123", "HASS_TOKEN": "default-ha-token",
    "TEAMS_CLIENT_ID": "default-client-id", "TEAMS_TENANT_ID": "default-tenant",
    "TEAMS_CLIENT_SECRET": "default-teams-secret", "TEAMS_HOME_CHANNEL": "default-conv",
    "DINGTALK_CLIENT_ID": "default-ding-id", "DINGTALK_CLIENT_SECRET": "default-ding-secret",
    "DINGTALK_WEBHOOK_URL": "https://oapi.dingtalk.com/robot/send?access_token=DEFAULT",
    "TELEGRAM_WEBHOOK_URL": "https://default.example/tg",
}
SECONDARY = {
    "MATRIX_HOMESERVER": "https://bot2.matrix.example", "MATRIX_USER_ID": "@bot2:example",
    "MATRIX_DEVICE_ID": "BOT2DEV", "MATRIX_ACCESS_TOKEN": "bot2-matrix-token",
    "HASS_URL": "http://bot2-ha.example:8123", "HASS_TOKEN": "bot2-ha-token",
    "TEAMS_CLIENT_ID": "bot2-client-id", "TEAMS_TENANT_ID": "bot2-tenant",
    "TEAMS_CLIENT_SECRET": "bot2-teams-secret", "TEAMS_HOME_CHANNEL": "bot2-conv",
    "DINGTALK_CLIENT_ID": "bot2-ding-id", "DINGTALK_CLIENT_SECRET": "bot2-ding-secret",
    "DINGTALK_WEBHOOK_URL": "https://oapi.dingtalk.com/robot/send?access_token=BOT2",
    "TELEGRAM_WEBHOOK_URL": "https://bot2.example/tg",
}


@pytest.fixture
def secondary_scope(monkeypatch):
    """Default profile in os.environ, multiplex on, secondary profile's scope installed."""
    for key, value in DEFAULT_ENV.items():
        monkeypatch.setenv(key, value)
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope(SECONDARY)
    yield
    ss.reset_secret_scope(token)
    ss.set_multiplex_active(False)


def test_matrix_homeserver_identity_follow_the_scoped_token(secondary_scope):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(enabled=True))
    assert adapter._access_token == SECONDARY["MATRIX_ACCESS_TOKEN"]
    assert (adapter._homeserver, adapter._user_id, adapter._device_id) == (
        SECONDARY["MATRIX_HOMESERVER"], SECONDARY["MATRIX_USER_ID"], SECONDARY["MATRIX_DEVICE_ID"])


@pytest.mark.asyncio
async def test_matrix_standalone_send_posts_scoped_token_to_scoped_homeserver(secondary_scope, monkeypatch):
    import aiohttp
    from plugins.platforms.matrix import adapter as mx

    seen = {}

    class _Resp:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def json(self): return {"event_id": "$x"}
        async def text(self): return ""

    class _Sess:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def put(self, url, **kw): seen["url"] = url; seen["headers"] = kw.get("headers", {}); return _Resp()

    monkeypatch.setattr(aiohttp, "ClientSession", _Sess)
    await mx._standalone_send(PlatformConfig(enabled=True), "!room:example", "hi")
    assert seen["url"].startswith(SECONDARY["MATRIX_HOMESERVER"] + "/")
    assert DEFAULT_ENV["MATRIX_HOMESERVER"] not in seen["url"]


def test_homeassistant_url_follows_the_scoped_token(secondary_scope):
    from plugins.platforms.homeassistant.adapter import HomeAssistantAdapter

    adapter = HomeAssistantAdapter(PlatformConfig(enabled=True))
    assert (adapter._hass_url, adapter._hass_token) == (SECONDARY["HASS_URL"], SECONDARY["HASS_TOKEN"])


@pytest.mark.asyncio
async def test_homeassistant_standalone_send_targets_scoped_url(secondary_scope, monkeypatch):
    import aiohttp
    from plugins.platforms.homeassistant import adapter as ha

    seen = {}

    class _Resp:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def text(self): return ""

    class _Sess:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def post(self, url, **kw): seen["url"] = url; return _Resp()

    monkeypatch.setattr(aiohttp, "ClientSession", _Sess)
    await ha._standalone_send(PlatformConfig(enabled=True), "x", "hi")
    assert seen["url"].startswith(SECONDARY["HASS_URL"] + "/")


def test_teams_credentials_pair_scoped_secret_with_scoped_app_identity(secondary_scope):
    teams = load_plugin_adapter("teams")
    assert teams._credentials(PlatformConfig(enabled=True)) == (
        SECONDARY["TEAMS_CLIENT_ID"], SECONDARY["TEAMS_CLIENT_SECRET"], SECONDARY["TEAMS_TENANT_ID"])
    # extra (per-profile config.yaml) must win over any env value.
    assert teams._credentials(PlatformConfig(enabled=True, extra={"client_id": "yaml-id"}))[0] == "yaml-id"


def test_teams_env_enablement_seeds_scoped_identity_and_home_channel(secondary_scope):
    teams = load_plugin_adapter("teams")
    seed = teams._env_enablement()
    assert seed["client_id"] == SECONDARY["TEAMS_CLIENT_ID"]
    assert seed["home_channel"]["chat_id"] == SECONDARY["TEAMS_HOME_CHANNEL"]


def test_dingtalk_client_id_follows_the_scoped_secret(secondary_scope):
    from plugins.platforms.dingtalk.adapter import _credentials

    assert _credentials(None) == (SECONDARY["DINGTALK_CLIENT_ID"], SECONDARY["DINGTALK_CLIENT_SECRET"])


@pytest.mark.asyncio
async def test_dingtalk_standalone_send_posts_to_scoped_robot_webhook(secondary_scope, monkeypatch):
    import httpx
    from plugins.platforms.dingtalk import adapter as ding

    posted = []

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"errcode": 0}

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, **kw): posted.append(url); return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    result = await ding._standalone_send(PlatformConfig(enabled=True), "c", "hi")
    assert result.get("success") is True
    assert posted == [SECONDARY["DINGTALK_WEBHOOK_URL"]]


@pytest.mark.asyncio
async def test_telegram_connect_registers_scoped_webhook_url(secondary_scope, monkeypatch):
    from plugins.platforms.telegram import adapter as tg

    class _Builder:
        def __getattr__(self, name):
            return lambda *a, **k: self
        def build(self):
            app = MagicMock(); app.bot = MagicMock(); app.start = AsyncMock(); app.initialize = AsyncMock()
            return app

    monkeypatch.setattr(tg, "Application", type("_App", (), {"builder": staticmethod(_Builder)}))
    adapter = tg.TelegramAdapter(PlatformConfig(enabled=True, token="bot2-token"))
    for name in ("_wire_plugin_handlers", "_register_handlers", "_mark_connected", "_start_post_connect_housekeeping"):
        monkeypatch.setattr(adapter, name, MagicMock())
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a, **k: True)
    monkeypatch.setattr(adapter, "_build_ptb_requests", AsyncMock(return_value=(MagicMock(), MagicMock())))
    monkeypatch.setattr(adapter, "_initialize_app_with_retries", AsyncMock())
    seen = {}

    async def _webhook(url, *, is_reconnect): seen["url"] = url
    async def _polling(*, is_reconnect): seen["url"] = "<polling>"

    monkeypatch.setattr(adapter, "_start_webhook_mode", _webhook)
    monkeypatch.setattr(adapter, "_start_polling_mode", _polling)
    assert await adapter.connect() is True
    assert seen["url"] == SECONDARY["TELEGRAM_WEBHOOK_URL"]
    assert os.environ["TELEGRAM_WEBHOOK_URL"] == DEFAULT_ENV["TELEGRAM_WEBHOOK_URL"]  # env untouched; scope won
