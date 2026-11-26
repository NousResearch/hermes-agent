"""The exact-retry identity of a local create, private pure rules of the reference bindings repository.

``same_create_retry`` compares the create request as sent and the client environment normalised by the rules frozen with the session. A stored map frozen with rules filters by the frozen secret substrings, so no live secret policy is consulted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Optional

__all__ = ["same_create_retry"]

_MAX_NAMES = 256
_MAX_BYTES = 256 * 1024


def _client_request(request_json: str, derived: Optional[list[str]] = None) -> Any:
    """The frozen create request as the client sent it."""
    request = json.loads(request_json)
    request.pop("desktop_local", None)
    if derived is None:
        request.pop("route_derived", None)
    if "model_requested" in request:
        request.pop("model", None)
        for key in ("base_url",) if derived is None else derived:
            request.pop(key, None)
    return request


def _same_create_request(saved_json: str, retry_json: str) -> bool:
    """Whether the retried create request is the stored one as the client sent it."""
    derived = json.loads(saved_json).get("route_derived")
    if _client_request(saved_json, derived) == _client_request(retry_json, derived):
        return True
    retry = json.loads(retry_json)
    if retry.get("model_requested") == retry.get("model"):
        retry.pop("model_requested", None)
        retry.pop("route_derived", None)
    return bool(json.loads(saved_json) == retry)


def _kept(name: str, rules: Mapping[str, Any]) -> bool:
    travels = not any(s in name.upper() for s in rules["secret"]) and not name.startswith("HERMES_")
    return name in rules["behaviour"] or (travels and (name in rules["names"] or name.startswith(tuple(rules["prefixes"]))))


def _normalise_client_env(value: Any, rules: Optional[Mapping[str, Any]]) -> dict[str, str]:
    if (not isinstance(value, dict) or len(value) > _MAX_NAMES
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items())
            or sum(len(k) + len(v) for k, v in value.items()) > _MAX_BYTES):
        raise ValueError("client_env must be a map of variable names to string values")
    return {name: value[name] for name in sorted(value) if rules is None or _kept(name, rules)}


def _retry_matches(committed: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    """The retried map, normalised by the rules frozen with the session, equals the stored one."""
    stored = committed.get("client_env_json")
    if stored is None or "client_env" not in request:
        return stored is None and "client_env" not in request
    rules = committed.get("client_env_rules_json")
    try:
        sent = _normalise_client_env(request["client_env"], None if rules is None else json.loads(rules))
    except ValueError:
        return False
    return bool(sent == json.loads(stored))


def same_create_retry(saved_policy: Mapping[str, Any], retry_json: str, client_request: Mapping[str, Any]) -> bool:
    """The same request as sent, and the same client environment under the frozen rules."""
    return _same_create_request(saved_policy["request_json"], retry_json) and _retry_matches(saved_policy, client_request)
