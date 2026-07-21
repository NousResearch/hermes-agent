"""Fail-closed resolution of logical model roles through llm-pool-proxy."""

from __future__ import annotations

import json
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from agent.secret_scope import get_secret

MODEL_ROLE_PREFIX = "role:"
MODEL_ROLE_PROXY_BASE_URL = "http://127.0.0.1:18989"
MODEL_ROLE_RESOLUTION_TIMEOUT_SECONDS = 5


class LogicalModelRoleResolutionError(RuntimeError):
    """The proxy could not resolve a requested logical model role."""


def _error_message(raw: bytes, fallback: str) -> str:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return fallback
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    return fallback


def resolve_logical_role(
    role: str,
    *,
    proxy_base_url: str = MODEL_ROLE_PROXY_BASE_URL,
    api_key: str | None = None,
) -> dict[str, str]:
    """Resolve *role* once via the authenticated proxy role-router endpoint.

    Returns only the atomic model/provider/api_mode tuple.  Any proxy or
    transport failure raises instead of selecting a physical-model fallback.
    """
    role = role.strip()
    if not role:
        raise LogicalModelRoleResolutionError("Logical model role must not be empty")
    api_key = api_key if api_key is not None else (get_secret("LLM_POOL_PROXY_API_KEY", "") or "")
    if not api_key:
        raise LogicalModelRoleResolutionError(
            "Cannot resolve logical model role: LLM_POOL_PROXY_API_KEY is not configured"
        )
    url = proxy_base_url.rstrip("/") + "/api/model-route/resolve"
    request = urllib_request.Request(
        url,
        data=json.dumps({"role": role}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=MODEL_ROLE_RESOLUTION_TIMEOUT_SECONDS) as response:
            payload: Any = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        raise LogicalModelRoleResolutionError(
            f"Logical model role '{role}' resolution failed ({exc.code}): "
            f"{_error_message(exc.read(), exc.reason)}"
        ) from exc
    except (urllib_error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise LogicalModelRoleResolutionError(
            f"Logical model role '{role}' resolution failed: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise LogicalModelRoleResolutionError(
            f"Logical model role '{role}' resolution returned an invalid response"
        )
    tuple_fields = {field: payload.get(field) for field in ("model", "provider", "api_mode")}
    if not all(isinstance(value, str) and value.strip() for value in tuple_fields.values()):
        raise LogicalModelRoleResolutionError(
            f"Logical model role '{role}' resolution returned an incomplete runtime tuple"
        )
    return {field: value.strip() for field, value in tuple_fields.items()}


def resolve_logical_model_role(model: str) -> dict[str, str] | None:
    """Resolve ``role:<name>`` models, leaving physical model strings untouched."""
    if not model.startswith(MODEL_ROLE_PREFIX):
        return None
    return resolve_logical_role(model[len(MODEL_ROLE_PREFIX):])
