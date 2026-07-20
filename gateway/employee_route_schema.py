"""Shared normalization helpers for employee background routes."""

from __future__ import annotations

from typing import Any


def coerce_employee_route_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, tuple | set):
        value = list(value)
    elif not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


def normalize_employee_route_match_modes(value: Any) -> tuple[str, ...]:
    raw_modes = coerce_employee_route_str_list(value)
    if not raw_modes:
        return ("explicit",)
    normalized: list[str] = []
    for item in raw_modes:
        mode = str(item or "").strip().lower()
        if mode in {"explicit", "heuristic"} and mode not in normalized:
            normalized.append(mode)
    return tuple(normalized) or ("explicit",)


def normalize_employee_route(route: dict[str, Any]) -> dict[str, Any] | None:
    worker_name = str(route.get("worker_name") or route.get("name") or "").strip()
    if not worker_name:
        return None
    routing_hints = route.get("routing_hints")
    if not isinstance(routing_hints, dict):
        routing_hints = {}
    return {
        "worker_name": worker_name,
        "aliases": coerce_employee_route_str_list(route.get("aliases")),
        "preloaded_skills": coerce_employee_route_str_list(
            route.get("preloaded_skills") or route.get("skills")
        ),
        "match_modes": normalize_employee_route_match_modes(
            route.get("match_modes") or route.get("match_mode")
        ),
        "action_terms": tuple(
            coerce_employee_route_str_list(
                routing_hints.get("action_terms") or route.get("action_terms")
            )
        ),
        "subject_terms": tuple(
            coerce_employee_route_str_list(
                routing_hints.get("subject_terms") or route.get("subject_terms")
            )
        ),
        "pain_terms": tuple(
            coerce_employee_route_str_list(
                routing_hints.get("pain_terms") or route.get("pain_terms")
            )
        ),
    }
