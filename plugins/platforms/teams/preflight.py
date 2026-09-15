"""Side-effect-free Microsoft Teams credential preflight."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ALLOWED_SERVICE_HOSTS = {"smba.trafficmanager.net", "smba.infra.gov.teams.microsoft.us"}
_TOKEN_SCOPE = "https://api.botframework.com/.default"
_REQUIRED_FIELDS = ("tenant_id", "client_id", "client_secret")

HttpPost = Callable[[str, dict[str, str], float], Awaitable[Any]]


def _valid_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(_IDENTIFIER.fullmatch(value))


def _malformed(config: Any) -> dict[str, Any] | None:
    if not isinstance(config, dict):
        return {"ok": False, "category": "malformed_configuration", "message": "Teams configuration must be an object."}
    for field, env_name in (("tenant_id", "TEAMS_TENANT_ID"), ("client_id", "TEAMS_CLIENT_ID"), ("client_secret", "TEAMS_CLIENT_SECRET")):
        if field not in config and env_name in config:
            config[field] = config[env_name]
    missing = [name for name in _REQUIRED_FIELDS if not isinstance(config.get(name), str) or not config[name].strip()]
    if missing:
        return {"ok": False, "category": "malformed_configuration", "message": "Missing required Teams fields: " + ", ".join(missing) + "."}
    for name in ("tenant_id", "client_id"):
        if not _valid_identifier(config[name]):
            return {"ok": False, "category": "malformed_configuration", "message": f"Teams {name} has an unsafe format."}
    if "service_url" in config and config["service_url"]:
        parsed = urlparse(str(config["service_url"]))
        if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_SERVICE_HOSTS or parsed.port not in (None, 443):
            return {"ok": False, "category": "malformed_configuration", "message": "Teams service URL is not an allowed HTTPS endpoint."}
    if "port" in config and config["port"] not in (None, ""):
        try:
            if not 1 <= int(config["port"]) <= 65535:
                raise ValueError
        except (TypeError, ValueError):
            return {"ok": False, "category": "malformed_configuration", "message": "Teams port must be between 1 and 65535."}
    return None


async def _default_post(url: str, data: dict[str, str], timeout: float) -> Any:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            return response


async def preflight_teams_config(config: dict[str, Any], *, http_post: HttpPost | None = None, timeout: float = 15.0) -> dict[str, Any]:
    """Validate credentials and request a Bot Framework token; never writes or starts anything."""
    config = dict(config) if isinstance(config, dict) else config
    invalid = _malformed(config)
    if invalid:
        return invalid
    tenant_id = config["tenant_id"].strip()
    client_id = config["client_id"].strip()
    client_secret = config["client_secret"]
    post = http_post or _default_post
    try:
        response = await post(
            f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
            {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret, "scope": _TOKEN_SCOPE},
            timeout,
        )
        status = int(response.status)
        if status == 401:
            return {"ok": False, "category": "invalid_credentials", "message": "Teams credentials were rejected."}
        if status == 403:
            return {"ok": False, "category": "permission_denied", "message": "Microsoft Entra admin consent or required permission is missing."}
        if status >= 400:
            return {"ok": False, "category": "token_acquisition_failed", "message": "Bot Framework token acquisition failed."}
        payload = await response.json()
        if not isinstance(payload, dict) or not payload.get("access_token"):
            return {"ok": False, "category": "token_acquisition_failed", "message": "Bot Framework returned no usable token."}
    except (asyncio.TimeoutError, TimeoutError):
        return {"ok": False, "category": "timeout", "message": "Token acquisition timed out."}
    except Exception:
        return {"ok": False, "category": "network_error", "message": "Could not reach Microsoft Entra ID."}

    return {
        "ok": True,
        "category": "success",
        "message": "Teams configuration passed the Bot Framework preflight.",
        "checklist": {
            "capabilities": [{"name": "Bot Framework messaging", "status": "verified"}],
            "permissions": [
                {"name": "Microsoft Entra admin consent", "status": "unknown", "note": "Not introspectable from a Bot Framework token request."},
                {"name": "Teams channel messaging", "status": "unknown", "note": "Not introspectable from a Bot Framework token request."},
            ],
        },
    }
