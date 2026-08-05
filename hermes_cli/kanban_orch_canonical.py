"""
ORCH V4 canonical data contract — normative implementation.

Source of truth: 拆卡機制四大缺陷-詳細實作計畫.md §3.1 + §I.5.
This module is the authoritative canonical JSON/digest/type gate.
Do NOT modify formulas without contract update.
"""

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Any

MAX_ARTIFACT_BYTES = 1_048_576
MAX_DEPTH = 32
MAX_CONTAINER_ITEMS = 10_000
MAX_STRING_BYTES = 65_536
MIN_INT = -(2**63)
MAX_INT = 2**63 - 1


class CanonicalError(ValueError):
    """Canonical validation error. Only ``code`` is public; no cause/context leaks."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code
        self.__cause__ = None
        self.__context__ = None
        self.__suppress_context__ = True


def require_exact_type(value: Any, expected: type | tuple[type, ...], *, code: str) -> Any:
    """Reject bool-as-int and any non-exact Python type before digest/formula use."""
    if expected is int:
        if type(value) is not int:  # bool is subclass of int — reject
            raise CanonicalError(code)
        return value
    if expected is bool:
        if type(value) is not bool:
            raise CanonicalError(code)
        return value
    if isinstance(expected, tuple):
        if type(value) not in expected:
            raise CanonicalError(code)
        if int in expected and type(value) is bool:
            raise CanonicalError(code)
        return value
    if type(value) is not expected:
        raise CanonicalError(code)
    return value


def assert_digest_matches(value: Any, claimed: Any, *, code: str = "digest_mismatch") -> str:
    """Server-side recompute: caller digest is assertion-only, never authority."""
    claimed_hex = require_sha256_hex(claimed, code="invalid_digest")
    actual = digest(value)
    if actual != claimed_hex:
        raise CanonicalError(code)
    return actual


def assert_raw_json_digest_matches(raw_json: str | bytes, claimed: Any, *, code: str = "digest_mismatch") -> str:
    """Recompute digest from raw JSON text/bytes after strict parse."""
    if type(raw_json) is str:
        raw = raw_json.encode("utf-8")
    elif type(raw_json) is bytes:
        raw = raw_json
    else:
        raise CanonicalError("invalid_json")
    parsed = strict_json_loads(raw)
    return assert_digest_matches(parsed, claimed, code=code)


def require_sha256_hex(value: Any, *, code: str = "invalid_digest") -> str:
    s = require_exact_type(value, str, code=code)
    if len(s) != 64 or any(ch not in "0123456789abcdef" for ch in s):
        raise CanonicalError(code)
    return s


def _normalize_string(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in normalized):
        raise CanonicalError("invalid_unicode")
    try:
        encoded = normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        err = CanonicalError("invalid_unicode")
        raise err from None
    if len(encoded) > MAX_STRING_BYTES:
        raise CanonicalError("string_too_large")
    return normalized


def normalize_tenant_scope(value: str | None) -> str:
    if value is None:
        return ""
    if type(value) is not str:
        raise CanonicalError("invalid_tenant_scope")
    return _normalize_string(value).strip()


def _reject_float(_: str) -> None:
    raise CanonicalError("float_not_allowed")


def _reject_constant(_: str) -> None:
    raise CanonicalError("non_finite_not_allowed")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_key, value in pairs:
        if type(raw_key) is not str:
            raise CanonicalError("non_string_key")
        key = _normalize_string(raw_key)
        if key in out:
            raise CanonicalError("duplicate_or_nfc_colliding_key")
        out[key] = value
    return out


def strict_json_loads(raw: bytes) -> Any:
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise CanonicalError("artifact_too_large")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise CanonicalError("invalid_utf8") from None
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError:
        raise CanonicalError("invalid_json") from None
    except CanonicalError:
        raise
    except Exception:
        raise CanonicalError("invalid_json") from None
    return _validate_value(value, depth=0)


def _validate_value(value: Any, *, depth: int) -> Any:
    if depth > MAX_DEPTH:
        raise CanonicalError("max_depth_exceeded")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if value < MIN_INT or value > MAX_INT:
            raise CanonicalError("int_out_of_range")
        return value
    if type(value) is str:
        return _normalize_string(value)
    if type(value) is list:
        if len(value) > MAX_CONTAINER_ITEMS:
            raise CanonicalError("container_too_large")
        return [_validate_value(item, depth=depth + 1) for item in value]
    if type(value) is dict:
        if len(value) > MAX_CONTAINER_ITEMS:
            raise CanonicalError("container_too_large")
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            if type(raw_key) is not str:
                raise CanonicalError("non_string_key")
            key = _normalize_string(raw_key)
            if key in normalized:
                raise CanonicalError("duplicate_or_nfc_colliding_key")
            normalized[key] = _validate_value(item, depth=depth + 1)
        return normalized
    raise CanonicalError("unsupported_type")


def canonical_json_bytes(value: Any) -> bytes:
    normalized = _validate_value(value, depth=0)
    raw = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise CanonicalError("artifact_too_large")
    return raw


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


ARTIFACT_KEYS: dict[str, tuple[frozenset[str], frozenset[str], frozenset[str]]] = {
    # (required, nullable, optional); unknown keys are always rejected.
    "orch_route": (
        frozenset({"schema_version", "kind", "origin_kind", "platform", "adapter_instance_id",
                   "account_id", "conversation_id", "route_revision", "required_ack_family",
                   "required_ack_strength"}),
        frozenset({"thread_id", "reply_to_id", "session_id", "notifier_profile"}), frozenset(),
    ),
    "orch_origin": (
        frozenset({"schema_version", "kind", "board_instance_id", "tenant_scope", "selector_key",
                   "route_digest"}), frozenset(), frozenset({"diagnostic_client_key"}),
    ),
    "orch_requirement_payload": (
        frozenset({"schema_version", "kind", "required", "text", "done_when"}),
        frozenset(), frozenset(),
    ),
    "orch_request": (
        frozenset({"schema_version", "kind", "selector_key", "request_key", "origin_id", "lineage_id",
                   "generation", "title", "synthesis_strategy", "completion_policy", "requirements"}),
        frozenset(), frozenset(),
    ),
    "orch_plan": (
        frozenset({"schema_version", "kind", "request_binding", "nodes", "edges", "coverage"}),
        frozenset(), frozenset(),
    ),
    "orch_result": (
        frozenset({"schema_version", "kind", "request_digest", "plan_digest", "accepted_lane_set",
                   "synthesis"}), frozenset(), frozenset(),
    ),
}


def validate_artifact_schema(kind: str, value: Any) -> dict[str, Any]:
    if type(value) is not dict or kind not in ARTIFACT_KEYS:
        raise CanonicalError("unsupported_artifact_kind")
    required, nullable, optional = ARTIFACT_KEYS[kind]
    allowed = required | nullable | optional
    keys = frozenset(value)
    if keys - allowed:
        raise CanonicalError("unknown_artifact_key")
    if required - keys:
        raise CanonicalError("missing_artifact_key")
    if any(value[key] is None for key in required):
        raise CanonicalError("null_required_key")
    if value.get("schema_version") != 4 or value.get("kind") != kind:
        raise CanonicalError("unsupported_artifact_schema")
    if "tenant_scope" in value:
        normalized = normalize_tenant_scope(value["tenant_scope"])
        if value["tenant_scope"] != normalized:
            raise CanonicalError("noncanonical_tenant_scope")
    return value


# ============================================================
# Artifact formula functions (§3.3)
# ============================================================

def route_digest(a: dict[str, Any]) -> str:
    return digest({
        "schema_version": 4,
        "kind": "orch_route",
        "origin_kind": a["origin_kind"],
        "platform": a["platform"],
        "adapter_instance_id": a["adapter_instance_id"],
        "account_id": a["account_id"],
        "conversation_id": a["conversation_id"],
        "thread_id": a["thread_id"],
        "reply_to_id": a["reply_to_id"],
        "session_id": a["session_id"],
        "notifier_profile": a["notifier_profile"],
        "required_ack_family": a["required_ack_family"],
        "required_ack_strength": a["required_ack_strength"],
        "route_revision": a["route_revision"],
    })


def origin_id(a: dict[str, Any]) -> str:
    return digest({
        "schema_version": 4,
        "kind": "orch_origin",
        "board_instance_id": a["board_instance_id"],
        "tenant_scope": normalize_tenant_scope(a.get("tenant_scope")),
        "selector_key": a["selector_key"],
        "route_digest": a["route_digest"],
    })


def replay_selector_key(a: dict[str, Any]) -> str:
    return digest({
        "schema_version": 4,
        "kind": "orch_replay_selector",
        "board_instance_id": a["board_instance_id"],
        "tenant_scope": normalize_tenant_scope(a.get("tenant_scope")),
        "adapter_instance_id": a["adapter_instance_id"],
        "conversation_id": a["conversation_id"],
        "selector_kind": a["selector_kind"],
        "selector_value": a["selector_value"],
    })


def request_key(a: dict[str, Any]) -> str:
    return digest({
        "schema_version": 4,
        "kind": "orch_request_generation",
        "board_instance_id": a["board_instance_id"],
        "tenant_scope": normalize_tenant_scope(a.get("tenant_scope")),
        "selector_key": a["selector_key"],
        "lineage_id": a["lineage_id"],
        "generation": a["generation"],
    })


def validate_supersession(predecessor: dict[str, Any], successor: dict[str, Any]) -> str:
    """§I.6.3: Validate supersession — predecessor must be terminal, generation+1.

    Returns the successor request_key if valid, raises CanonicalError otherwise.
    """
    if predecessor.get("lifecycle_state") not in ("failed", "cancelled"):
        raise CanonicalError("supersession_predecessor_not_terminal")
    pred_gen = require_exact_type(predecessor.get("generation"), int, code="invalid_int")
    succ_gen = require_exact_type(successor.get("generation"), int, code="invalid_int")
    if succ_gen != pred_gen + 1:
        raise CanonicalError("supersession_generation_not_incremented")
    if predecessor.get("lineage_id") != successor.get("lineage_id"):
        raise CanonicalError("supersession_lineage_mismatch")
    return request_key(successor)


def lane_lineage_key(a: dict[str, Any]) -> str:
    return digest({
        "schema_version": 4,
        "kind": "orch_lane_lineage",
        "board_instance_id": a["board_instance_id"],
        "tenant_scope": normalize_tenant_scope(a.get("tenant_scope")),
        "lineage_id": a["lineage_id"],
        "role": a["role"],
        "lane_label": a["lane_label"],
        "goal": a["normalized_goal"],
        "done_when": a["normalized_done_when"],
        "required": a["required"],
        "requirement_ids": sorted(a["requirement_ids"]),
    })


def requirement_digest(requirement: dict[str, Any]) -> str:
    return digest({
        "schema_version": 4,
        "kind": "orch_requirement_payload",
        "required": requirement["required"],
        "text": requirement["text"],
        "done_when": requirement["done_when"],
    })


def request_digest(request: dict[str, Any]) -> str:
    return digest({
        "schema_version": 4,
        "kind": "orch_request",
        "selector_key": request["selector_key"],
        "request_key": request["request_key"],
        "origin_id": request["origin_id"],
        "lineage_id": request["lineage_id"],
        "generation": request["generation"],
        "title": request["title"],
        "synthesis_strategy": request["synthesis_strategy"],
        "completion_policy": request["completion_policy"],
        "requirements": sorted(request["requirements"], key=lambda row: row["ordinal"]),
    })


def requirement_id(a: dict[str, Any]) -> str:
    return digest({
        "schema_version": 4,
        "kind": "orch_requirement",
        "request_key": a["request_key"],
        "ordinal": a["ordinal"],
        "requirement_digest": a["requirement_digest"],
    })


def node_key(a: dict[str, Any]) -> str:
    if a["role"] == "parent":
        return "__parent__"
    return digest({
        "schema_version": 4,
        "kind": "orch_node",
        "orch_id": a["orch_id"],
        "generation": a["generation"],
        "plan_version": a["plan_version"],
        "lane_lineage_key": a["lane_lineage_key"],
    })


def edge_key(a: dict[str, Any]) -> str:
    return digest({
        "schema_version": 4,
        "kind": "orch_edge",
        "orch_id": a["orch_id"],
        "plan_version": a["plan_version"],
        "parent_node_key": a["parent_node_key"],
        "child_node_key": a["child_node_key"],
        "edge_kind": a["edge_kind"],
    })


def event_key(a: dict[str, Any]) -> str:
    return digest({
        "schema_version": 4,
        "kind": "orch_event",
        "board_instance_id": a["board_instance_id"],
        "tenant_scope": normalize_tenant_scope(a.get("tenant_scope")),
        "orch_id": a["orch_id"],
        "lifecycle_revision": a["lifecycle_revision"],
        "cancel_epoch": a["cancel_epoch"],
        "event_kind": a["event_kind"],
        "target_key": a["target_key"],
        "payload_digest": a["payload_digest"],
    })
