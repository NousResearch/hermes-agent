"""Databricks AI Gateway model and alias discovery.

Databricks exposes serving endpoints and governed aliases through workspace
catalog APIs instead of a per-surface ``/models`` route.  This module keeps
that provider-specific catalog logic out of the generic model probe.
"""

from __future__ import annotations

import logging
import urllib.parse
from collections.abc import Callable
from typing import Any

from agent.anthropic_endpoints import _is_databricks_workspace_endpoint

logger = logging.getLogger(__name__)

_SURFACE_API_TYPE = {
    "anthropic": "anthropic/v1/messages",
    "openai": "openai/v1/responses",
    "mlflow": "mlflow/v1/chat/completions",
}
_API_TYPE_PREFERENCE = tuple(_SURFACE_API_TYPE.values())
_SERVING_ENDPOINTS_PATH = "/api/2.0/serving-endpoints"

_ALIAS_CATALOGS = (
    {
        "list_path": "/api/ai-gateway/v2/endpoints?page_size=100",
        "item_path": "/api/ai-gateway/v2/endpoints/{name}",
        "items_key": "endpoints",
        "name_prefix": "",
        "destinations": ("config", "destinations"),
    },
    {
        "list_path": "/api/2.1/unity-catalog/model-services?view=FULL&page_size=100",
        "item_path": "/api/2.1/unity-catalog/model-services/{name}",
        "items_key": "model_services",
        "name_prefix": "model-services/",
        "destinations": ("config", "routing", "destinations"),
    },
)
_UNRESOLVED = object()


def _gateway_surface(base_url: str) -> tuple[str, str] | None:
    """Return ``(workspace origin, surface)`` for a Databricks gateway URL."""
    try:
        parsed = urllib.parse.urlparse(base_url)
    except (TypeError, ValueError):
        return None
    if not _is_databricks_workspace_endpoint(base_url):
        return None
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2 or parts[0] != "ai-gateway":
        return None
    surface = parts[1].lower()
    if surface not in _SURFACE_API_TYPE:
        return None
    return f"{parsed.scheme}://{parsed.netloc}", surface


def _preferred_api_type(api_types: Any) -> str | None:
    listed = {item for item in (api_types or ()) if isinstance(item, str)}
    return next(
        (candidate for candidate in _API_TYPE_PREFERENCE if candidate in listed), None
    )


def _serving_endpoint_api_type(endpoint: dict[str, Any]) -> str | None:
    state = endpoint.get("state")
    capabilities = endpoint.get("capabilities")
    config = endpoint.get("config")
    if not all(isinstance(value, dict) for value in (state, capabilities, config)):
        return None
    if (
        state.get("ready") != "READY"
        or capabilities.get("function_calling") is not True
    ):
        return None

    api_types: set[str] = set()
    for entity in config.get("served_entities") or ():
        foundation = (
            entity.get("foundation_model") if isinstance(entity, dict) else None
        )
        if (
            not isinstance(foundation, dict)
            or foundation.get("ai_gateway_v2_supported") is not True
        ):
            continue
        listed = foundation.get("api_types")
        if isinstance(listed, list):
            api_types.update(item for item in listed if isinstance(item, str))
    return _preferred_api_type(api_types)


def _classify_serving_endpoints(payload: Any) -> dict[str, str | None]:
    """Map endpoints with a catalog verdict to their preferred API type."""
    endpoints = payload.get("endpoints") if isinstance(payload, dict) else None
    if not isinstance(endpoints, list):
        return {}
    classified: dict[str, str | None] = {}
    for endpoint in endpoints:
        if not isinstance(endpoint, dict) or not isinstance(
            endpoint.get("capabilities"), dict
        ):
            continue
        name = str(endpoint.get("name") or "").strip()
        if name:
            classified[name] = _serving_endpoint_api_type(endpoint)
    return classified


def _destination_model_name(destination: dict[str, Any]) -> str:
    """Resolve both legacy and public model-service destination shapes."""
    pay_per_token = destination.get("pay_per_token_config")
    raw = (
        pay_per_token.get("model")
        if isinstance(pay_per_token, dict)
        else destination.get("name")
    )
    return str(raw or "").strip().rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def _inherited_api_type(
    destination: dict[str, Any], endpoints: dict[str, str | None]
) -> Any:
    return endpoints.get(_destination_model_name(destination), _UNRESOLVED)


def _native_api_type(
    destination: dict[str, Any], _endpoints: dict[str, str | None]
) -> Any:
    external = destination.get("external_model_config")
    target = external.get("target") if isinstance(external, dict) else None
    native = target.get("native_api_types") if isinstance(target, dict) else None
    return _preferred_api_type(native) if isinstance(native, list) else _UNRESOLVED


_DESTINATION_API_TYPE = {
    "PAY_PER_TOKEN_FOUNDATION_MODEL": _inherited_api_type,
    "EXTERNAL_FOUNDATION_MODEL": _native_api_type,
}


def _page_alias_api_types(
    payload: Any,
    catalog: dict[str, Any],
    endpoints: dict[str, str | None],
) -> tuple[dict[str, str | None], list[str]]:
    """Return placed and unresolved aliases from one catalog page."""
    items = payload.get(catalog["items_key"]) if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return {}, []

    placed: dict[str, str | None] = {}
    unresolved: list[str] = []
    prefix = catalog["name_prefix"]
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if prefix and name.startswith(prefix):
            name = name[len(prefix) :]
        if not name or name.startswith("system.ai.") or name in endpoints:
            continue

        destinations: Any = item
        for key in catalog["destinations"]:
            destinations = (
                destinations.get(key) if isinstance(destinations, dict) else None
            )
        if not (
            isinstance(destinations, list)
            and len(destinations) == 1
            and isinstance(destinations[0], dict)
        ):
            continue
        destination = destinations[0]
        if (
            destination.get("traffic_percentage") not in (None, 100)
            or destination.get("is_deleted") is True
        ):
            continue

        supported = item.get("supported_api_types")
        if isinstance(supported, list):
            placed[name] = _preferred_api_type(supported)
            continue

        kind = str(
            destination.get("destination_type") or destination.get("type") or ""
        ).removeprefix("DESTINATION_TYPE_")
        derive = _DESTINATION_API_TYPE.get(kind)
        api_type = derive(destination, endpoints) if derive else _UNRESOLVED
        if api_type is _UNRESOLVED:
            unresolved.append(name)
        else:
            placed[name] = api_type
    return placed, unresolved


def _fetch_aliases(
    workspace_root: str,
    api_key: str,
    endpoints: dict[str, str | None],
    *,
    timeout: float,
    get_json: Callable[..., Any],
    user_agent: str,
    ssl_context: Any = None,
) -> dict[str, str | None]:
    open_kwargs = {"ssl_context": ssl_context} if ssl_context is not None else {}
    headers = {"User-Agent": user_agent, "Authorization": f"Bearer {api_key}"}

    def fetch(path: str) -> Any:
        try:
            return get_json(
                workspace_root + path,
                timeout=timeout,
                headers=headers,
                **open_kwargs,
            )
        except Exception as exc:
            logger.warning(
                "Databricks alias catalog %s unavailable; model list may be incomplete: %s",
                path,
                exc,
            )
            return None

    aliases: dict[str, str | None] = {}
    for catalog in _ALIAS_CATALOGS:
        unresolved: list[str] = []
        page_token = ""
        seen_tokens: set[str] = set()
        while True:
            path = catalog["list_path"]
            if page_token:
                path += "&page_token=" + urllib.parse.quote(page_token, safe="")
            payload = fetch(path)
            if payload is None:
                break
            placed, pending = _page_alias_api_types(payload, catalog, endpoints)
            aliases.update(placed)
            unresolved.extend(pending)
            page_token = (
                str(payload.get("next_page_token") or "")
                if isinstance(payload, dict)
                else ""
            )
            if not page_token:
                break
            if page_token in seen_tokens:
                logger.warning(
                    "Databricks alias catalog %s repeated page token; model list may be incomplete",
                    catalog["list_path"],
                )
                break
            seen_tokens.add(page_token)

        for name in unresolved:
            quoted_name = urllib.parse.quote(name, safe="")
            item = fetch(catalog["item_path"].format(name=quoted_name))
            supported = (
                item.get("supported_api_types") if isinstance(item, dict) else None
            )
            aliases[name] = (
                _preferred_api_type(supported) if isinstance(supported, list) else None
            )

    unplaced = sorted(name for name, api_type in aliases.items() if api_type is None)
    if unplaced:
        logger.info(
            "Databricks alias catalogs: %d alias(es) not offered on a supported model API surface: %s",
            len(unplaced),
            ", ".join(unplaced),
        )
    return aliases


def probe_databricks_gateway(
    base_url: str,
    api_key: str,
    *,
    timeout: float,
    get_json: Callable[..., Any],
    user_agent: str,
    ssl_context: Any = None,
) -> tuple[list[str], str] | None:
    """Discover models for one Databricks AI Gateway surface.

    Returns ``None`` when the URL is not a supported Databricks gateway or the
    serving-endpoints catalog cannot be reached.  A successful catalog may be
    empty; external aliases can still populate the requested surface.
    """
    gateway = _gateway_surface(base_url)
    if gateway is None:
        return None
    workspace_root, surface = gateway
    catalog_url = workspace_root + _SERVING_ENDPOINTS_PATH
    open_kwargs = {"ssl_context": ssl_context} if ssl_context is not None else {}
    headers = {"User-Agent": user_agent, "Authorization": f"Bearer {api_key}"}
    try:
        payload = get_json(catalog_url, timeout=timeout, headers=headers, **open_kwargs)
    except Exception as exc:
        logger.debug("Databricks serving-endpoints catalog unavailable: %s", exc)
        return None

    endpoints = _classify_serving_endpoints(payload)
    aliases = _fetch_aliases(
        workspace_root,
        api_key,
        endpoints,
        timeout=timeout,
        get_json=get_json,
        user_agent=user_agent,
        ssl_context=ssl_context,
    )
    wanted = _SURFACE_API_TYPE[surface]
    models = sorted(
        name
        for name, api_type in {**aliases, **endpoints}.items()
        if api_type == wanted
    )
    return models, catalog_url
