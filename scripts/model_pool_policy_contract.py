#!/usr/bin/env python3
"""Shared model-pool policy contract helpers.

Centralizes policy schema loading + validation so gate/router/snapshot runtimes
consume one deterministic policy contract implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


def json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(p) for p in seq)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_schema(path: Path) -> Tuple[bool, Optional[str], Dict[str, Any], Optional[Dict[str, Any]]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "gate_unavailable", {"error": "jsonschema_validator_unavailable"}, None
    if not path.exists():
        return False, "gate_unavailable", {"error": "schema_missing", "schema_path": str(path)}, None
    try:
        payload = load_json_file(path)
    except Exception as exc:
        return False, "gate_unavailable", {"error": "schema_unreadable", "schema_path": str(path), "detail": str(exc)}, None
    if not isinstance(payload, dict):
        return False, "gate_unavailable", {"error": "schema_not_object", "schema_path": str(path)}, None
    return True, None, {"schema_path": str(path)}, payload


def load_pool_policy(policy_path: Path, policy_schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any], Optional[Dict[str, Any]]]:
    try:
        policy_doc = load_json_file(policy_path)
    except Exception as exc:
        return (
            False,
            "pool_policy_invalid",
            {"error": "pool_policy_unreadable", "path": str(policy_path), "detail": str(exc)},
            None,
        )

    if not isinstance(policy_doc, dict):
        return False, "pool_policy_invalid", {"error": "pool_policy_not_object", "path": str(policy_path)}, None

    ok, reason, details, schema_doc = _load_json_schema(policy_schema_path)
    if not ok:
        return False, reason, details, None

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(policy_doc),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if errors:
        err = errors[0]
        return (
            False,
            "pool_policy_invalid",
            {
                "error": "pool_policy_schema_validation_failed",
                "path": str(policy_path),
                "data_path": json_ptr(err.absolute_path),
                "schema_path": json_ptr(err.absolute_schema_path),
                "message": str(err.message),
            },
            None,
        )

    return True, None, {"path": str(policy_path), "schema_path": str(policy_schema_path), "policy_id": policy_doc.get("policy_id")}, policy_doc


def policy_route_entry(pool_policy: Mapping[str, Any], route_class: str) -> Optional[Mapping[str, Any]]:
    route_classes = pool_policy.get("route_classes") if isinstance(pool_policy.get("route_classes"), Mapping) else {}
    entry = route_classes.get(route_class)
    return entry if isinstance(entry, Mapping) else None


def policy_allowed_models(pool_policy: Mapping[str, Any], route_class: str) -> set[str]:
    entry = policy_route_entry(pool_policy, route_class)
    if not entry:
        return set()
    return {
        str(model_key)
        for model_key in (entry.get("allowed_models") if isinstance(entry.get("allowed_models"), list) else [])
        if isinstance(model_key, str) and model_key.strip()
    }
