"""Config/data-driven employee background routing definitions."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from gateway.config import GatewayConfig, Platform

logger = logging.getLogger(__name__)

_DEFAULT_ROUTES_PATH = Path(__file__).resolve().parent / "data" / "employee_routes.json"


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    normalized: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_match_modes(value: Any) -> Tuple[str, ...]:
    raw_modes = _coerce_str_list(value)
    if not raw_modes:
        return ("explicit",)
    normalized: List[str] = []
    for item in raw_modes:
        mode = str(item or "").strip().lower()
        if mode in {"explicit", "heuristic"} and mode not in normalized:
            normalized.append(mode)
    return tuple(normalized) or ("explicit",)


@lru_cache(maxsize=1)
def _load_default_routes() -> List[Dict[str, Any]]:
    try:
        raw = json.loads(_DEFAULT_ROUTES_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Employee routes data file missing: %s", _DEFAULT_ROUTES_PATH)
        return []
    except Exception as exc:
        logger.warning("Failed to load employee routes from %s: %s", _DEFAULT_ROUTES_PATH, exc)
        return []
    if not isinstance(raw, list):
        logger.warning("Employee routes file must contain a list: %s", _DEFAULT_ROUTES_PATH)
        return []
    return [item for item in raw if isinstance(item, dict)]


def _normalize_route(route: Dict[str, Any]) -> Dict[str, Any] | None:
    worker_name = str(route.get("worker_name") or route.get("name") or "").strip()
    if not worker_name:
        return None
    routing_hints = route.get("routing_hints")
    if not isinstance(routing_hints, dict):
        routing_hints = {}
    normalized = {
        "worker_name": worker_name,
        "aliases": _coerce_str_list(route.get("aliases")),
        "preloaded_skills": _coerce_str_list(route.get("preloaded_skills") or route.get("skills")),
        "match_modes": _normalize_match_modes(route.get("match_modes") or route.get("match_mode")),
        "action_terms": tuple(
            _coerce_str_list(routing_hints.get("action_terms") or route.get("action_terms"))
        ),
        "subject_terms": tuple(
            _coerce_str_list(routing_hints.get("subject_terms") or route.get("subject_terms"))
        ),
        "pain_terms": tuple(
            _coerce_str_list(routing_hints.get("pain_terms") or route.get("pain_terms"))
        ),
    }
    return normalized


def get_employee_routes(config: GatewayConfig | None, *, platform: Platform) -> List[Dict[str, Any]]:
    """Return normalized employee-route definitions for a platform."""
    source_routes: Any = None
    try:
        pconfig = (config.platforms or {}).get(platform) if config else None
        extra = getattr(pconfig, "extra", None) if pconfig else None
        if isinstance(extra, dict) and "employee_routes" in extra:
            source_routes = extra.get("employee_routes")
    except Exception:
        source_routes = None

    if source_routes is None:
        source_routes = _load_default_routes()
    if not isinstance(source_routes, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for route in source_routes:
        if not isinstance(route, dict):
            continue
        if route.get("enabled", True) is False:
            continue
        item = _normalize_route(route)
        if item is not None:
            normalized.append(item)
    return normalized
