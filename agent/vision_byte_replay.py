"""Session-owned fingerprints for tool images evicted by 413 recovery.

Managed turns reconstruct AIAgent. Only route and image fingerprints are persisted;
canonical image history and endpoint credentials never enter this replay sidecar.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

MODEL_CONFIG_KEY = "_vision_payload_evictions_v1"


def route_fingerprint(route: tuple[str, str, str, str]) -> str:
    return hashlib.sha256(json.dumps(route).encode("utf-8")).hexdigest()


def _decode(raw: Any) -> dict[str, set[tuple[str, bytes]]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Invalid vision eviction sidecar")
    state = {}
    for route, entries in raw.items():
        if not isinstance(route, str) or not re.fullmatch(r"[a-f0-9]{64}", route) or not isinstance(entries, list):
            raise ValueError("Invalid vision eviction sidecar")
        rejected = set()
        for entry in entries:
            if (not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str)
                    or not isinstance(entry[1], str) or not re.fullmatch(r"[a-f0-9]{64}", entry[1])):
                raise ValueError("Invalid vision eviction sidecar")
            rejected.add((entry[0], bytes.fromhex(entry[1])))
        state[route] = rejected
    return state


def byte_image_records(agent: Any) -> dict[str, set[tuple[str, bytes]]]:
    """Cache per logical session; a new worker restores exactly that session's record."""
    sid = getattr(agent, "session_id", None)
    cached = getattr(agent, "_byte_stripped_tool_images", None)
    if isinstance(cached, dict) and getattr(agent, "_byte_stripped_session_id", None) == sid:
        return cached
    getter = getattr(getattr(agent, "_session_db", None), "get_session_model_config_value", None)
    raw = getter(sid, MODEL_CONFIG_KEY, {}) if sid and callable(getter) else {}
    state = _decode(raw)
    agent._byte_stripped_tool_images = state
    agent._byte_stripped_session_id = sid
    return state


def persist_byte_image_records(agent: Any, retained_ids: set[str] | None) -> None:
    state = byte_image_records(agent)
    if retained_ids is not None:
        for route in list(state):
            state[route] = {entry for entry in state[route] if entry[0] in retained_ids}
            if not state[route]:
                del state[route]
    if getattr(agent, "_persist_disabled", False):
        return
    patcher = getattr(getattr(agent, "_session_db", None), "patch_session_model_config", None)
    sid = getattr(agent, "session_id", None)
    if sid and callable(patcher):
        encoded = {route: [[call_id, digest.hex()] for call_id, digest in sorted(entries)]
                   for route, entries in state.items()}
        patcher(sid, {MODEL_CONFIG_KEY: encoded})
