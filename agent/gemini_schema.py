"""Tool-schema preparation for Gemini's native API.

Two wire shapes: ``parametersJsonSchema`` (plain JSON Schema, v1beta only) gets a light
normalizer (``prepare_gemini_tool_parameters``); the legacy ``parameters`` field accepts
only the OpenAPI ``Schema`` subset and keeps the lossy translator
(``sanitize_gemini_tool_parameters``) for API versions without the JSON Schema field.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional

from tools.schema_sanitizer import _normalize_type_array

logger = logging.getLogger(__name__)

# Gemini's ``FunctionDeclaration.parameters`` accepts only a subset of OpenAPI 3.0 /
# JSON Schema (the ``Schema`` object); everything else is stripped.
_GEMINI_SCHEMA_ALLOWED_KEYS = {
    "type", "format", "title", "description", "nullable", "enum", "maxItems", "minItems", "properties", "required",
    "minProperties", "maxProperties", "minLength", "maxLength", "pattern", "example", "anyOf", "propertyOrdering",
    "default", "items", "minimum", "maximum",
}


_GEMINI_STRUCTURAL_KEYS = {
    "array": {"items", "minItems", "maxItems"},
    "object": {"properties", "required", "minProperties", "maxProperties", "propertyOrdering"},
}


def _stringify_enum_value(item: Any) -> Any:
    """Gemini-safe string for a scalar enum entry, or None to drop it."""
    if isinstance(item, bool):
        return "true" if item else "false"
    if isinstance(item, (int, float)) and math.isfinite(item):
        return str(item)
    return item if isinstance(item, str) else None


def _normalize_gemini_type_array(type_array: list, cleaned: Dict[str, Any]) -> None:
    """Keep union alternatives and their branch-local structural constraints."""
    derived: Dict[str, Any] = {}
    _normalize_type_array(type_array, derived)
    if "anyOf" in derived:
        constraints = {"anyOf": cleaned["anyOf"]} if "anyOf" in cleaned else {}
        # Gemini requires items/properties on the typed branch itself, not
        # its typeless parent. Keep required paired with those properties.
        structural = {key: cleaned.pop(key) for keys in _GEMINI_STRUCTURAL_KEYS.values()
                      for key in keys if key in cleaned}
        cleaned["anyOf"] = [
            sanitize_gemini_schema({**branch, **constraints, **{
                key: value for key, value in structural.items()
                if key in _GEMINI_STRUCTURAL_KEYS.get(branch["type"], ())
            }}) for branch in derived["anyOf"]
        ]
    else:
        cleaned["type"] = derived["type"]
    if derived.get("nullable"):
        # Derived from "null" in the array. Set AFTER the loop so it beats an input
        # ``nullable: false`` regardless of which key the producer emitted first.
        cleaned["nullable"] = True


def sanitize_gemini_schema(schema: Any) -> Dict[str, Any]:
    """Gemini-compatible copy of a tool parameter schema: keeps only the documented subset
    (drops e.g. ``$schema`` / ``additionalProperties``) and recursively sanitizes nested
    ``properties`` / ``items`` / ``anyOf``."""
    if not isinstance(schema, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _GEMINI_SCHEMA_ALLOWED_KEYS:
            continue
        if key == "properties":
            if isinstance(value, dict):
                cleaned[key] = {name: sanitize_gemini_schema(sub) for name, sub in value.items() if isinstance(name, str)}
        elif key == "items":
            cleaned[key] = sanitize_gemini_schema(value)
        elif key == "anyOf":
            if isinstance(value, list):
                cleaned[key] = [sanitize_gemini_schema(item) for item in value if isinstance(item, dict)]
        else:
            cleaned[key] = value

    type_array = cleaned.get("type")
    if isinstance(type_array, list):
        cleaned.pop("type")
        _normalize_gemini_type_array(type_array, cleaned)

    # Gemini requires every ``enum`` entry to be a string even for
    # integer/number/boolean types; the declared type stays intact and Gemini
    # still emits typed tool arguments at runtime. dict.fromkeys = ordered dedupe.
    enum_val = cleaned.get("enum")
    if isinstance(enum_val, list) and (
        isinstance(type_array, list) or cleaned.get("type") in {"integer", "number", "boolean"}
    ):
        if stringified := list(dict.fromkeys(v for v in map(_stringify_enum_value, enum_val) if v is not None)):
            cleaned["enum"] = stringified
        else:
            cleaned.pop("enum", None)

    # Gemini validates ``required`` strictly against the same node's ``properties`` (HTTP 400
    # "property is not defined") and one bad tool schema fails the ENTIRE request. MCP servers
    # routinely emit ``required`` without ``properties``, so keep only names that exist here;
    # the tool handler still validates required fields at execution time.
    required_val = cleaned.get("required")
    if isinstance(required_val, list):
        props_val = cleaned.get("properties")
        prop_names = set(props_val) if isinstance(props_val, dict) else set()
        valid_required = [name for name in required_val if isinstance(name, str) and name in prop_names]
        if not valid_required:
            cleaned.pop("required", None)
        elif len(valid_required) != len(required_val):
            cleaned["required"] = valid_required
    return cleaned


def sanitize_gemini_tool_parameters(parameters: Any) -> Dict[str, Any]:
    """Normalize tool parameters to a valid Gemini object schema."""
    return sanitize_gemini_schema(parameters) or {"type": "object", "properties": {}}


# ── parametersJsonSchema (full JSON Schema) ─────────────────────────────────
#
# The legacy translator is lossy: anyOf unions without an outer type, bare arrays,
# $ref/$defs and additionalProperties had to be stripped or repaired, and one
# unrepresentable construct 400s the ENTIRE request. Through parametersJsonSchema the
# schema goes as-is; only same-document $refs are inlined (MCP pydantic / zod emit
# them and Google rejects reference indirection) and root ``$schema`` is dropped.

_EMPTY_OBJECT_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {}}
# Real tool schemas hold a handful of refs; the cap stops circular pydantic models
# from expanding forever.
_MAX_REF_EXPANSIONS = 256


def _resolve_local_ref(root: Dict[str, Any], ref: str) -> Optional[Dict[str, Any]]:
    """Resolve a same-document JSON pointer (``#/$defs/Foo``) against *root*."""
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    node: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


def _inline_refs(node: Any, root: Dict[str, Any], budget: List[int], stack: tuple = ()) -> Any:
    """Recursively inline same-document ``$ref`` nodes; ``ValueError`` on an unresolvable
    or circular reference or an exhausted budget (the caller then keeps the original)."""
    if isinstance(node, list):
        return [_inline_refs(item, root, budget, stack) for item in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if not isinstance(ref, str):
        return {key: _inline_refs(value, root, budget, stack) for key, value in node.items()}
    if ref in stack:
        raise ValueError(f"circular $ref {ref!r}")
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("$ref expansion budget exhausted")
    target = _resolve_local_ref(root, ref)
    if target is None:
        raise ValueError(f"unresolvable $ref {ref!r}")
    inlined = _inline_refs(target, root, budget, stack + (ref,))
    # JSON Schema: siblings of $ref (description, default, ...) apply alongside the
    # referenced schema and win over it.
    siblings = {k: v for k, v in node.items() if k != "$ref"}
    return {**inlined, **_inline_refs(siblings, root, budget, stack)} if siblings else inlined


def prepare_gemini_tool_parameters(parameters: Any) -> Dict[str, Any]:
    """Full JSON Schema for ``parametersJsonSchema``: deep-copied, root ``$schema`` dropped,
    same-document ``$ref`` inlined, object root guaranteed. A schema whose references
    cannot all be resolved is sent untouched so the provider names the real problem."""
    if not isinstance(parameters, dict) or not parameters:
        return dict(_EMPTY_OBJECT_SCHEMA)
    schema = copy.deepcopy(parameters)
    schema.pop("$schema", None)
    try:
        schema = _inline_refs(schema, schema, [_MAX_REF_EXPANSIONS])
    except ValueError as exc:
        logger.debug("Gemini tool schema kept as-is ($ref inlining skipped): %s", exc)
        return schema
    schema.pop("$defs", None)
    schema.pop("definitions", None)
    if not schema:
        return dict(_EMPTY_OBJECT_SCHEMA)
    if schema.get("type") == "object" and "properties" not in schema:
        schema["properties"] = {}
    return schema
