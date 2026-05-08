"""Configuration parsing for the MemPalace Hermes plugin."""

from __future__ import annotations

import os
from dataclasses import dataclass
from string import Formatter
from typing import Any

DEFAULT_WING = "conversations"
DEFAULT_N_RESULTS = 5
DEFAULT_TOOL_MAX_RESULTS = 20
DEFAULT_ENABLE_KG = True
DEFAULT_COLLECTION_TEMPLATE = "hermes-{platform}-{user_id}"
DEFAULT_ROOM_STRATEGY = "platform_session"
DEFAULT_FIXED_ROOM = "memory"
VALID_ROOM_STRATEGIES = {"fixed", "session", "platform_session", "user_platform"}
VALID_TEMPLATE_FIELDS = {"user_id", "platform", "session_id", "agent_id"}
MAX_COLLECTION_NAME_LENGTH = 63
MAX_ROOM_NAME_LENGTH = 63


@dataclass(slots=True)
class MemPalaceConfig:
    palace_path: str = ""
    wing: str = DEFAULT_WING
    n_results: int = DEFAULT_N_RESULTS
    tool_max_results: int = DEFAULT_TOOL_MAX_RESULTS
    enable_kg: bool = DEFAULT_ENABLE_KG
    collection_name: str = ""
    collection_template: str = DEFAULT_COLLECTION_TEMPLATE
    room_strategy: str = DEFAULT_ROOM_STRATEGY
    fixed_room: str = DEFAULT_FIXED_ROOM


def _coerce_positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _plugin_section(full_config: object) -> dict[str, Any]:
    if not isinstance(full_config, dict):
        return {}
    raw: dict[str, Any] = full_config
    nested = raw.get("mempalace")
    if isinstance(nested, dict):
        return nested
    return raw


def _sanitize_label(value: object, default: str, *, max_length: int) -> str:
    text = str(value or "").strip().lower()
    cleaned = []
    last_dash = False
    for ch in text:
        if ch.isascii() and ch.isalnum():
            cleaned.append(ch)
            last_dash = False
        elif ch in {"-", "_", " ", "/", ":"}:
            if not last_dash:
                cleaned.append("-")
                last_dash = True
    result = "".join(cleaned).strip("-")
    if not result:
        result = default
    return result[:max_length].strip("-") or default


def _normalize_template(value: object) -> str:
    template = (
        str(value or DEFAULT_COLLECTION_TEMPLATE).strip() or DEFAULT_COLLECTION_TEMPLATE
    )
    formatter = Formatter()
    fields = {
        field_name for _, field_name, _, _ in formatter.parse(template) if field_name
    }
    if not fields:
        return DEFAULT_COLLECTION_TEMPLATE
    if any(field not in VALID_TEMPLATE_FIELDS for field in fields):
        return DEFAULT_COLLECTION_TEMPLATE
    return template


def load_mempalace_config(
    full_config: object, hermes_home: str | None = None
) -> MemPalaceConfig:
    raw = _plugin_section(full_config)

    palace_path = str(raw.get("palace_path") or "").strip()
    if palace_path:
        palace_path = os.path.expanduser(palace_path)
    elif hermes_home:
        palace_path = os.path.join(hermes_home, "mempalace")
    else:
        palace_path = os.path.expanduser("~/.hermes/mempalace")
    palace_path = os.path.abspath(palace_path)

    wing = str(raw.get("wing") or DEFAULT_WING).strip() or DEFAULT_WING
    room_strategy = (
        str(raw.get("room_strategy") or DEFAULT_ROOM_STRATEGY).strip()
        or DEFAULT_ROOM_STRATEGY
    )
    if room_strategy not in VALID_ROOM_STRATEGIES:
        room_strategy = DEFAULT_ROOM_STRATEGY

    n_results = _coerce_positive_int(raw.get("n_results"), DEFAULT_N_RESULTS)
    tool_max_results = _coerce_positive_int(
        raw.get("tool_max_results"), DEFAULT_TOOL_MAX_RESULTS
    )
    if tool_max_results < n_results:
        tool_max_results = n_results

    collection_name = _sanitize_label(
        raw.get("collection_name"), "", max_length=MAX_COLLECTION_NAME_LENGTH
    )
    if collection_name == "":
        collection_name = ""
    collection_template = _normalize_template(raw.get("collection_template"))
    fixed_room = _sanitize_label(
        raw.get("fixed_room"), DEFAULT_FIXED_ROOM, max_length=MAX_ROOM_NAME_LENGTH
    )

    return MemPalaceConfig(
        palace_path=palace_path,
        wing=wing,
        n_results=n_results,
        tool_max_results=tool_max_results,
        enable_kg=_coerce_bool(raw.get("enable_kg"), DEFAULT_ENABLE_KG),
        collection_name=collection_name,
        collection_template=collection_template,
        room_strategy=room_strategy,
        fixed_room=fixed_room,
    )
