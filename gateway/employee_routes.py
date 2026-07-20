"""Config/data-driven employee background routing definitions."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from gateway.config import GatewayConfig, Platform
from gateway.employee_route_schema import normalize_employee_route
from gateway.employee_route_store import list_employee_routes

logger = logging.getLogger(__name__)


def _load_static_employee_routes(config: GatewayConfig | None, *, platform: Platform) -> list[dict[str, Any]]:
    source_routes: Any = None
    try:
        pconfig = (config.platforms or {}).get(platform) if config else None
        extra = getattr(pconfig, "extra", None) if pconfig else None
        if isinstance(extra, dict) and "employee_routes" in extra:
            source_routes = extra.get("employee_routes")
    except Exception:
        source_routes = None

    if source_routes is None or not isinstance(source_routes, list):
        return []

    normalized: list[dict[str, Any]] = []
    for route in source_routes:
        if not isinstance(route, dict):
            continue
        if route.get("enabled", True) is False:
            continue
        item = normalize_employee_route(route)
        if item is not None:
            normalized.append(item)
    return normalized


def get_employee_routes(config: GatewayConfig | None, *, platform: Platform) -> List[Dict[str, Any]]:
    """Return normalized employee-route definitions for a platform.

    Employee routes are intentionally config-only. The framework does not ship
    built-in worker personas so deployments stay policy-driven and upstream-safe.
    """
    merged_routes: list[dict[str, Any]] = []
    route_indexes: dict[str, int] = {}

    for route in _load_static_employee_routes(config, platform=platform):
        route_indexes[route["worker_name"]] = len(merged_routes)
        merged_routes.append(route)

    for route in list_employee_routes(platform):
        if route.get("enabled", True) is False:
            continue
        normalized = normalize_employee_route(route)
        if normalized is None:
            continue
        existing_index = route_indexes.get(normalized["worker_name"])
        if existing_index is None:
            route_indexes[normalized["worker_name"]] = len(merged_routes)
            merged_routes.append(normalized)
            continue
        merged_routes[existing_index] = normalized

    return merged_routes
