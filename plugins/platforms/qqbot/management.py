"""QQ-owned menu/panel REST client; no gateway connection or automatic writes."""
from __future__ import annotations

from contextlib import asynccontextmanager
from urllib.parse import quote, urlencode

from .constants import API_BASE, TOKEN_URL


def read_credentials() -> tuple[str, str]:
    """Resolve only the active profile's QQ configuration and secrets."""
    from gateway.platforms._shared import get_scoped_secret
    from gateway.config import Platform, PlatformConfig, load_gateway_config

    platform = load_gateway_config().platforms.get(Platform.QQBOT, PlatformConfig())
    extra = platform.extra or {}
    app_id = extra.get("app_id") or get_scoped_secret("QQ_APP_ID", "")
    secret = platform.token or extra.get("client_secret") or get_scoped_secret("QQ_CLIENT_SECRET", "")
    if not app_id or not secret:
        raise ValueError("QQ credentials are not configured")
    return str(app_id), str(secret)


class QQBotManagementClient:
    """Pass complete JSON bodies and one-page responses through unchanged.

    The caller owns the injected HTTP client's lifetime. ``open_management_client``
    supplies the production lifetime boundary. No retries or adapter internals.
    """

    def __init__(self, http, app_id: str, client_secret: str):
        self._http = http
        self._app_id = app_id
        self._client_secret = client_secret
        self._access_token = None

    async def request(self, method: str, path: str, payload=None):
        if self._access_token is None:
            response = await self._http.post(
                TOKEN_URL, json={"appId": self._app_id, "clientSecret": self._client_secret})
            data = self._decode(response)
            self._access_token = data.get("access_token")
            if not self._access_token:
                raise RuntimeError("QQ authentication returned no access token")
        response = await self._http.request(
            method, API_BASE + path, json=payload,
            headers={"Authorization": f"QQBot {self._access_token}", "X-Union-Appid": self._app_id})
        return self._decode(response)

    @staticmethod
    def _decode(response):
        response.raise_for_status()
        data = response.json() if response.content else {}
        if not isinstance(data, dict):
            raise RuntimeError("QQ returned a non-object response")
        if data.get("code") not in (None, 0, "0"):
            raise RuntimeError("QQ rejected the management request")
        return data

    async def get_global_menu(self):
        return await self.request("GET", "/v2/menu")

    async def update_global_menu(self, payload):
        return await self.request("PUT", "/v2/menu", payload)

    async def list_panels(self, scope, *, cursor=None, limit=None):
        params = {"scope": scope}
        if cursor is not None:
            params["cursor"] = cursor
        if limit is not None:
            params["limit"] = limit
        return await self.request("GET", "/v2/panels?" + urlencode(params))

    async def create_panel(self, payload):
        return await self.request("POST", "/v2/panels", payload)

    async def get_panel(self, panel_id):
        return await self.request("GET", panel_path(panel_id))

    async def update_panel(self, panel_id, payload):
        return await self.request("PUT", panel_path(panel_id), payload)

    async def update_panel_targets(self, panel_id, payload):
        return await self.request("PUT", panel_path(panel_id) + "/target", payload)

    async def delete_panel(self, panel_id):
        return await self.request("DELETE", panel_path(panel_id))


def panel_path(panel_id: str) -> str:
    if panel_id in ("", ".", ".."):
        raise ValueError("panel_id must be a nonempty identifier, not a dot segment")
    return "/v2/panels/" + quote(panel_id, safe="")


@asynccontextmanager
async def open_management_client():
    """Open authenticated REST resources only when the caller executes a request."""
    import httpx

    app_id, secret = read_credentials()
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as http:
        yield QQBotManagementClient(http, app_id, secret)
