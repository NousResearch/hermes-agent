"""Pipedream Apps backend, mounted at ``/api/plugins/pipedream-apps/``.

The Desktop Apps tab never sees the Pipedream client secret or the server OAuth
access token. It calls this router through ``ctx.rest``; every Pipedream request
is made here, server-side, with the project's client-credentials token. The only
credential the renderer receives is the short-lived Connect Link it opens in the
browser.

Configuration is baked into the module constants below, so the plugin works the
same in a source checkout and inside the production Docker image (no gateway
environment or ``.env`` file is required).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, HTTPException, Query

log = logging.getLogger(__name__)

router = APIRouter()

_API_BASE = "https://api.pipedream.com"
# Baked-in project configuration (shared by source and the production Docker image).
_CLIENT_ID = "e-Z0DaSW794F-dO1Mprd7lXSbNWmKpG8joXKd8p4V5E"
_CLIENT_SECRET = "rimYTA12SVExkWOe9N1YopDoANfmIFt9S60jKQlieKo"
_PROJECT_ID = "proj_XZsbovO"
_ENVIRONMENT = "production"
_MCP_BASE = "https://agent.unitalk.ai"
# Server-side OAuth token: covers the catalog/accounts proxy and token minting.
_OAUTH_SCOPE = (
    "connect:apps:* connect:accounts:read connect:accounts:write connect:tokens:create"
)
# Connect Links only accept the client-side subset; connect:tokens:create is
# OAuth-only and would be rejected here.
_CONNECT_TOKEN_SCOPE = "connect:apps:* connect:accounts:read connect:accounts:write"
_APP_SLUG_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_ACCOUNT_ID_RE = re.compile(r"^apn_[A-Za-z0-9_-]{1,120}$")
_APP_FIELDS = (
    "id",
    "name_slug",
    "name",
    "auth_type",
    "description",
    "img_src",
    "categories",
    "featured_weight",
)
_ACCOUNT_FIELDS = (
    "id",
    "name",
    "external_id",
    "healthy",
    "dead",
    "app",
    "created_at",
    "updated_at",
    "authorized_scopes",
    "expires_at",
    "error",
    "last_refreshed_at",
    "next_refresh_at",
)
_EMAIL_KEYS = (
    "email",
    "emailAddress",
    "email_address",
    "user_email",
    "username",
    "userEmail",
)

# Test seam: a MockTransport can be installed without touching production code.
_transport: httpx.AsyncBaseTransport | None = None
_TOKEN: dict[str, Any] = {"access_token": "", "expires_at": 0.0}


def _config() -> tuple[str, str, str]:
    return (_CLIENT_ID, _CLIENT_SECRET, _PROJECT_ID)


def _configured() -> bool:
    return all(_config())


def _required_config() -> tuple[str, str, str]:
    client_id, client_secret, project_id = _config()
    if not (client_id and client_secret and project_id):
        raise HTTPException(status_code=503, detail="Pipedream is not configured")
    return client_id, client_secret, project_id


def _project_id() -> str:
    return _required_config()[2]


def _environment() -> str:
    value = _ENVIRONMENT.lower()
    return value if value in ("development", "production") else "production"


def _mcp_base() -> str | None:
    """Full MCP proxy prefix (``…/api/pipedream/mcp``) the Desktop plugin appends
    ``/<external_user_id>/<app_slug>`` to. Both a bare app URL and an already
    complete prefix are accepted."""
    value = _MCP_BASE.rstrip("/")
    if not value:
        return None
    return (
        value if value.endswith("/api/pipedream/mcp") else f"{value}/api/pipedream/mcp"
    )


def _mcp_server_name(app_slug: str) -> str:
    return f"pipedream_{app_slug}"


def _mcp_entry(servers: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((entry for entry in servers if entry.get("name") == name), None)


def _mcp_entry_owned(entry: dict[str, Any], base: str) -> bool:
    """Whether a config entry is one this plugin may replace or disable.

    Pipedream entries are ordinary ``mcp_servers`` rows, so ownership is the name
    pattern plus a URL under the configured MCP proxy base. A user-authored row
    that happens to share the name but points elsewhere is left untouched.
    """
    if entry.get("source") != "config":
        return False
    url = entry.get("url")
    return isinstance(url, str) and url.startswith(f"{base}/")


async def _enable_pipedream_mcp(external_user_id: str, app_slug: str) -> dict[str, Any]:
    """Create, enable or replace ``pipedream_<slug>`` through the Dashboard MCP API."""
    base = _mcp_base()
    if not base:
        raise HTTPException(
            status_code=503, detail="Pipedream MCP proxy base is not configured"
        )
    from hermes_cli.web_models import MCPEnabledToggle, MCPServerCreate
    from hermes_cli.web_routers.mcp import (
        add_mcp_server,
        list_mcp_servers,
        remove_mcp_server,
        set_mcp_server_enabled,
    )

    name = _mcp_server_name(app_slug)
    expected_url = (
        f"{base}/{quote(external_user_id, safe='')}/{quote(app_slug, safe='')}"
    )

    for attempt in range(2):
        listed = await list_mcp_servers()
        entry = _mcp_entry(listed.get("servers") or [], name)

        if entry is None:
            try:
                await add_mcp_server(MCPServerCreate(name=name, url=expected_url))
            except HTTPException as error:
                # Another connect won the add race; re-list and enable that entry.
                if error.status_code == 409 and attempt == 0:
                    continue
                raise
            return {"ok": True, "name": name, "action": "added", "url": expected_url}

        if entry.get("source") == "plugin":
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Server '{name}' is provided by plugin "
                    f"'{entry.get('plugin')}' and cannot be modified"
                ),
            )

        if entry.get("url") == expected_url:
            if entry.get("enabled"):
                return {
                    "ok": True,
                    "name": name,
                    "action": "unchanged",
                    "url": expected_url,
                }
            await set_mcp_server_enabled(name, MCPEnabledToggle(enabled=True))
            return {
                "ok": True,
                "name": name,
                "action": "enabled",
                "url": expected_url,
            }

        if _mcp_entry_owned(entry, base):
            await remove_mcp_server(name)
            await add_mcp_server(MCPServerCreate(name=name, url=expected_url))
            return {
                "ok": True,
                "name": name,
                "action": "replaced",
                "url": expected_url,
            }

        raise HTTPException(
            status_code=409,
            detail=(
                f"Server '{name}' exists with a different URL; "
                "leaving it untouched"
            ),
        )

    raise HTTPException(
        status_code=409,
        detail=f"Server '{name}' is already being configured; try again",
    )


async def _disable_pipedream_mcp(app_slug: str) -> dict[str, Any]:
    """Disable ``pipedream_<slug>`` via the Dashboard MCP API when we own it."""
    base = _mcp_base()
    if not base:
        return {"ok": True, "action": "absent", "reason": "no MCP proxy base"}
    from hermes_cli.web_models import MCPEnabledToggle
    from hermes_cli.web_routers.mcp import list_mcp_servers, set_mcp_server_enabled

    name = _mcp_server_name(app_slug)
    listed = await list_mcp_servers()
    entry = _mcp_entry(listed.get("servers") or [], name)

    if entry is None:
        return {"ok": True, "name": name, "action": "absent"}
    if not _mcp_entry_owned(entry, base):
        return {
            "ok": False,
            "name": name,
            "action": "ignored",
            "error": "MCP server is not owned by Pipedream",
        }
    if not entry.get("enabled"):
        return {"ok": True, "name": name, "action": "unchanged"}

    await set_mcp_server_enabled(name, MCPEnabledToggle(enabled=False))
    return {"ok": True, "name": name, "action": "disabled"}


async def _disable_pipedream_mcp_after_disconnect(
    external_user_id: str, account: dict[str, Any]
) -> None:
    """Best-effort disable after the account delete; never fail the delete itself."""
    if not _mcp_base():
        return
    app = account.get("app")
    app_slug = app.get("name_slug") if isinstance(app, dict) else None
    if not isinstance(app_slug, str) or not _APP_SLUG_RE.match(app_slug):
        return

    try:
        remaining = await _fetch_accounts(external_user_id, include_credentials=False)
        deleted_id = account.get("id")
        if any(
            entry.get("id") != deleted_id
            and isinstance(entry.get("app"), dict)
            and entry["app"].get("name_slug") == app_slug
            for entry in remaining
        ):
            return
        result = await _disable_pipedream_mcp(app_slug)
        if not result.get("ok"):
            log.warning(
                "Pipedream MCP disable after disconnect failed: %s",
                result.get("error"),
            )
    except Exception as error:  # health: allow BLE001 -- account delete succeeded
        log.warning(
            "Pipedream MCP disable after disconnect failed: %s",
            type(error).__name__,
        )


def _api_base() -> str:
    return _API_BASE.rstrip("/")


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=_api_base(),
        timeout=httpx.Timeout(20.0, connect=10.0),
        transport=_transport,
    )


def _upstream_error(status: int) -> HTTPException:
    if status == 404:
        return HTTPException(status_code=404, detail="Pipedream resource not found")
    if status in (401, 403):
        return HTTPException(
            status_code=502, detail="Pipedream rejected the configured credentials"
        )
    if status == 429:
        return HTTPException(
            status_code=503, detail="Pipedream rate limit reached; try again shortly"
        )
    return HTTPException(status_code=502, detail="Pipedream request failed")


async def _oauth_token() -> str:
    now = time.monotonic()
    cached = str(_TOKEN.get("access_token") or "")
    expires_at = float(_TOKEN.get("expires_at") or 0.0)
    if cached and expires_at > now + 30:
        return cached

    client_id, client_secret, _ = _required_config()
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": _OAUTH_SCOPE,
    }

    try:
        async with _make_client() as client:
            response = await client.post("/v1/oauth/token", json=payload)
    except httpx.HTTPError as error:
        log.warning("Pipedream token transport failure: %s", type(error).__name__)
        raise HTTPException(
            status_code=503, detail="Could not reach Pipedream"
        ) from None

    if response.status_code >= 400:
        log.warning("Pipedream token request failed: %s", response.status_code)
        raise _upstream_error(response.status_code)

    try:
        data = response.json()
    except ValueError:
        raise HTTPException(
            status_code=502, detail="Pipedream returned an invalid token response"
        ) from None

    token = data.get("access_token") if isinstance(data, dict) else None
    if not isinstance(token, str) or not token:
        raise HTTPException(
            status_code=502, detail="Pipedream did not return an access token"
        )

    lifetime = data.get("expires_in", 3600) if isinstance(data, dict) else 3600
    try:
        lifetime_seconds = float(lifetime)
    except (TypeError, ValueError):
        lifetime_seconds = 3600.0

    _TOKEN["access_token"] = token
    _TOKEN["expires_at"] = now + max(60.0, lifetime_seconds - 60.0)
    return token


async def _request(
    method: str,
    path: str,
    *,
    params: list[tuple[str, str]] | None = None,
    json: dict[str, Any] | None = None,
    token: str | None = None,
) -> Any:
    access_token = token or await _oauth_token()
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "X-PD-Environment": _environment(),
    }

    try:
        async with _make_client() as client:
            response = await client.request(
                method, path, params=params, json=json, headers=headers
            )
    except httpx.HTTPError as error:
        log.warning("Pipedream %s transport failure: %s", method, type(error).__name__)
        raise HTTPException(
            status_code=502, detail="Could not reach Pipedream"
        ) from None

    if response.status_code >= 400:
        log.warning(
            "Pipedream %s %s -> %s", method, path.split("?", 1)[0], response.status_code
        )
        raise _upstream_error(response.status_code)

    if response.status_code == 204 or not response.content:
        return None

    try:
        return response.json()
    except ValueError:
        raise HTTPException(
            status_code=502, detail="Pipedream returned an invalid response"
        ) from None


def _external_user_id(value: object) -> str:
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="externalUserId is required")
    clean = value.strip()
    if (
        not clean
        or len(clean) > 250
        or any(ord(char) < 32 or ord(char) == 127 for char in clean)
    ):
        raise HTTPException(status_code=400, detail="externalUserId is invalid")
    return clean


def _app_slug(value: object) -> str:
    if not isinstance(value, str) or not _APP_SLUG_RE.match(value.strip()):
        raise HTTPException(status_code=400, detail="Invalid app slug")
    return value.strip()


def _safe_app(app: dict[str, Any]) -> dict[str, Any]:
    safe = {key: app[key] for key in _APP_FIELDS if key in app}
    categories = safe.get("categories")
    if isinstance(categories, list):
        safe["categories"] = [
            category for category in categories if isinstance(category, str)
        ]
    return safe


def _pick_email(credentials: object, name: object) -> str | None:
    if isinstance(credentials, dict):
        for key in _EMAIL_KEYS:
            value = credentials.get(key)
            if isinstance(value, str) and "@" in value:
                return value.strip()
    if isinstance(name, str) and "@" in name:
        return name.strip()
    return None


def _safe_account(account: object) -> dict[str, Any] | None:
    if not isinstance(account, dict):
        return None
    safe = {key: account[key] for key in _ACCOUNT_FIELDS if key in account}
    safe["email"] = _pick_email(account.get("credentials"), account.get("name"))
    app = safe.get("app")
    if isinstance(app, dict):
        safe["app"] = _safe_app(app)
    return safe


def _with_app(link: str, app_slug: str) -> str:
    parts = urlsplit(link)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["app"] = app_slug
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


async def _fetch_accounts(
    external_user_id: str, *, include_credentials: bool
) -> list[dict[str, Any]]:
    params = [("include_credentials", "true")] if include_credentials else []
    path = f"/v1/connect/{quote(_project_id(), safe='')}/users/{quote(external_user_id, safe='')}/accounts"

    try:
        data = await _request("GET", path, params=params)
    except HTTPException as error:
        # Credential enrichment is best-effort: retry once without it only when
        # the parameter itself was rejected. Auth and rate-limit failures stand.
        if not include_credentials or error.status_code not in (400, 422):
            raise
        data = await _request("GET", path)

    if isinstance(data, dict):
        data = data.get("data")
    return [entry for entry in data or [] if isinstance(entry, dict)]


@router.get("/config")
async def config() -> dict[str, Any]:
    """Non-secret readiness for the Apps tab. Never includes credentials."""
    return {
        "configured": _configured(),
        "environment": _environment(),
        "mcpBaseUrl": _mcp_base() if _configured() else None,
    }


@router.get("/apps")
async def list_apps(
    q: str = "",
    limit: int = 24,
    after: str | None = None,
    before: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    _required_config()
    safe_limit = max(1, min(60, limit or 24))
    params: list[tuple[str, str]] = [("limit", str(safe_limit))]

    if q.strip():
        params.append(("q", q.strip()[:200]))
    if category and category.strip():
        params.append(("category_ids", category.strip()))
    if after and after.strip():
        params.append(("after", after.strip()))
    elif before and before.strip():
        params.append(("before", before.strip()))

    data = await _request("GET", "/v1/connect/apps", params=params)
    apps = data.get("data") if isinstance(data, dict) else None
    page = data.get("page_info") if isinstance(data, dict) else None
    page = page if isinstance(page, dict) else {}
    normalized = [_safe_app(app) for app in apps or [] if isinstance(app, dict)]

    return {
        "apps": normalized,
        "nextCursor": page.get("end_cursor"),
        "prevCursor": page.get("start_cursor"),
        "totalCount": page.get("total_count", len(normalized)),
    }


@router.get("/categories")
async def list_categories() -> dict[str, Any]:
    _required_config()
    data = await _request("GET", "/v1/connect/app_categories")
    raw = data.get("categories") if isinstance(data, dict) else data

    categories = []
    for entry in raw or []:
        if not isinstance(entry, dict) or not entry.get("id") or not entry.get("name"):
            continue
        categories.append(
            {
                "id": entry.get("id"),
                "name": entry.get("name"),
                "description": entry.get("description"),
            }
        )

    return {"categories": categories}


@router.get("/apps/{slug}")
async def retrieve_app(slug: str) -> dict[str, Any]:
    _required_config()
    data = await _request("GET", f"/v1/connect/apps/{quote(_app_slug(slug), safe='')}")
    app = (
        data.get("data")
        if isinstance(data, dict) and isinstance(data.get("data"), dict)
        else data
    )
    if not isinstance(app, dict):
        raise HTTPException(status_code=404, detail="Pipedream app not found")
    return _safe_app(app)


@router.post("/connect")
async def create_connect_token(payload: dict[str, Any]) -> dict[str, Any]:
    _required_config()
    external_user_id = _external_user_id(
        payload.get("externalUserId") or payload.get("external_user_id")
    )
    slug = _app_slug(payload.get("appSlug") or payload.get("app_slug"))

    data = await _request(
        "POST",
        f"/v1/connect/{quote(_project_id(), safe='')}/tokens",
        json={
            "external_user_id": external_user_id,
            "expires_in": 3600,
            "scope": _CONNECT_TOKEN_SCOPE,
        },
    )

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=502, detail="Pipedream returned an invalid token response"
        )

    link = data.get("connect_link_url")
    token = data.get("token")
    if not isinstance(link, str) or not link or not isinstance(token, str) or not token:
        raise HTTPException(
            status_code=502, detail="Pipedream did not return a Connect Link"
        )

    return {
        "token": token,
        "connectLinkUrl": _with_app(link, slug),
        "expiresAt": data.get("expires_at"),
    }


@router.post("/mcp/enable")
async def enable_pipedream_mcp(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure the connected app has an enabled MCP proxy entry.

    Uses the Dashboard MCP API in-process, so the entry is profile-scoped by the
    plugin route dependency and appears in the same MCP list as every other
    server. No live reload: the change applies on the next session/gateway start.
    """
    external_user_id = _external_user_id(
        payload.get("externalUserId") or payload.get("external_user_id")
    )
    app_slug = _app_slug(payload.get("appSlug") or payload.get("app_slug"))
    return await _enable_pipedream_mcp(external_user_id, app_slug)


@router.get("/accounts")
async def list_accounts(
    external_user_id: str | None = Query(default=None, alias="externalUserId"),
) -> list[dict[str, Any]]:
    _required_config()
    user = _external_user_id(external_user_id)
    raw = await _fetch_accounts(user, include_credentials=True)
    return [
        account
        for account in (_safe_account(entry) for entry in raw)
        if account is not None
    ]


@router.delete("/accounts/{account_id}")
async def delete_account(
    account_id: str,
    external_user_id: str | None = Query(default=None, alias="externalUserId"),
) -> dict[str, bool]:
    _required_config()
    user = _external_user_id(external_user_id)
    if not _ACCOUNT_ID_RE.match(account_id):
        raise HTTPException(status_code=400, detail="Invalid account id")

    accounts = await _fetch_accounts(user, include_credentials=False)
    target = next(
        (account for account in accounts if account.get("id") == account_id),
        None,
    )
    if target is None:
        raise HTTPException(status_code=404, detail="Account not found or not owned")

    await _request(
        "DELETE",
        f"/v1/connect/{quote(_project_id(), safe='')}/accounts/{quote(account_id, safe='')}",
    )
    # Disable the app's MCP proxy only when this was the last account for that
    # app. The account delete is authoritative; the MCP step is best-effort.
    await _disable_pipedream_mcp_after_disconnect(user, target)
    return {"ok": True}
