"""Helpers for translating OpenAI-style tool schemas to MiniMax's schema subset.

MiniMax's OpenAI-compatible endpoint (api.minimax.io/v1) rejects several
standard JSON Schema constructs that OpenAI and other providers accept:

1. ``boolean`` type parameters — MiniMax expects ``"type": "string"`` with
   an ``enum`` of ``["true", "false"]`` or similar string alternatives.
2. ``nullable`` keyword — non-standard in JSON Schema; MiniMax rejects it.
3. ``anyOf``/``oneOf`` containing ``{"type": "null"}`` branches — collapse them.
4. Missing ``type`` on property schemas — MiniMax requires explicit types.
5. Missing ``required`` array on object schemas — MiniMax requires it.
6. Top-level ``parameters`` must be ``type: object``.

Known rejection sources include browser tool schemas that use ``"type": "boolean"``
(e.g. ``browser_snapshot.full``), which MiniMax cannot parse.

Reference: https://platform.minimax.io/docs/api-reference/text-anthropic-api
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

# Keys whose values are maps of name → schema (not schemas themselves).
# When we recurse, we walk the values of these maps as schemas, but we do
# NOT apply the missing-type repair to the map itself.
_SCHEMA_MAP_KEYS = frozenset({"properties", "patternProperties", "$defs", "definitions"})

# Keys whose values are lists of schemas.
_SCHEMA_LIST_KEYS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})

# Keys whose values are a single nested schema.
_SCHEMA_NODE_KEYS = frozenset({"items", "contains", "not", "additionalProperties", "propertyNames"})

# Model slugs (lowercased) that identify MiniMax models.
# Covers the known family: minimax-m3, minimax-m2.7, minimax-m2.5, etc.
_MINIMAX_MODEL_PREFIXES = frozenset({"minimax-m3", "minimax-m2", "minimax3", "minimax-m1"})


def _repair_schema(node: Any, is_schema: bool = True) -> Any:
    """Recursively apply MiniMax repairs to a schema node.

    ``is_schema=True`` means this dict is a JSON Schema node and gets the
    missing-type, boolean→string, null-collapse, and required-array repairs
    applied.  ``is_schema=False`` means it's a container map (e.g. the value
    of ``properties``) and we only recurse into its values.
    """
    if isinstance(node, list):
        # Lists only show up under schema-list keys (anyOf/oneOf/allOf), so
        # every element is itself a schema.
        return [_repair_schema(item, is_schema=True) for item in node]
    if not isinstance(node, dict):
        return node

    # Walk the dict, deciding per-key whether recursion is into a schema
    # node, a container map, or a scalar.
    repaired: Dict[str, Any] = {}
    for key, value in node.items():
        if key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
            # Map of name → schema.  Don't treat the map itself as a schema
            # (it has no type / properties of its own), but each value is.
            repaired[key] = {
                sub_key: _repair_schema(sub_val, is_schema=True)
                for sub_key, sub_val in value.items()
            }
        elif key in _SCHEMA_LIST_KEYS and isinstance(value, list):
            repaired[key] = [_repair_schema(v, is_schema=True) for v in value]
        elif key in _SCHEMA_NODE_KEYS:
            # items / not / additionalProperties: single nested schema.
            # additionalProperties can also be a bool — leave those alone.
            if isinstance(value, dict):
                repaired[key] = _repair_schema(value, is_schema=True)
            else:
                repaired[key] = value
        else:
            # Scalars (description, title, format, enum values, etc.) pass through.
            repaired[key] = value

    if not is_schema:
        return repaired

    # Rule 1: Collapse anyOf/oneOf with null branches.
    # MiniMax rejects {"anyOf": [{"type": "null"}, {"type": "string"}]}.
    # Collapse such anyOf to the first non-null branch.
    for kw in ("anyOf", "oneOf"):
        if kw in repaired and isinstance(repaired[kw], list):
            non_null = [
                b for b in repaired[kw]
                if isinstance(b, dict) and b.get("type") != "null"
            ]
            if non_null and len(non_null) < len(repaired[kw]):
                if len(non_null) == 1:
                    # Promote the single non-null branch into the parent.
                    merged = {k: v for k, v in repaired.items() if k != kw}
                    merged.update(non_null[0])
                    repaired = merged
                    # Continue with the merged node so that further repairs
                    # (boolean→string, missing type, etc.) apply.
                else:
                    # Keep the reduced union (all non-null branches) and fall
                    # through to the common cleanup so outer ``nullable`` /
                    # ``default`` keywords are still stripped.
                    repaired[kw] = non_null
            elif not non_null:
                # Everything was null; collapse to a string schema.
                repaired.pop(kw, None)
                repaired["type"] = "string"
            # else: no null branches — union is already clean; fall through to
            # the common cleanup for the outer keywords.

    # Rule 2: Strip non-standard keywords that MiniMax rejects.
    repaired.pop("nullable", None)
    # `default` is not part of the OpenAI function-calling spec and many
    # providers reject it.  Fold the value into the description instead.
    default_val = repaired.pop("default", None)
    if default_val is not None and "description" in repaired:
        repaired["description"] = (
            f"{repaired['description']} (default: {json.dumps(default_val)})"
        )

    # Rule 3: Convert boolean type to integer (0/1).
    # MiniMax rejects ``"type": "boolean"`` in tool schemas.
    # Using integer (not string) so Python truthiness works: 0→False, 1→True.
    # The description is updated so the model knows to emit 0 or 1.
    type_val = repaired.get("type")
    if type_val == "boolean":
        repaired["type"] = "integer"
        repaired["enum"] = [0, 1]
        if "description" in repaired:
            repaired["description"] = (
                f"{repaired['description']} (0=false, 1=true)"
            )

    # Rule 4: Fix enum arrays that MiniMax rejects.
    # MiniMax rejects non-string values in enum arrays (boolean, integer, etc.).
    if "enum" in repaired and isinstance(repaired["enum"], list):
        # If type was already string or we just set it above, coerce enum values to strings.
        if repaired.get("type") == "string":
            cleaned: List[str] = []
            for v in repaired["enum"]:
                if v is None:
                    continue
                if isinstance(v, bool):
                    cleaned.append("true" if v else "false")
                else:
                    cleaned.append(str(v))
            cleaned = [v for v in cleaned if v != ""]
            if cleaned:
                repaired["enum"] = cleaned
            else:
                repaired.pop("enum")

    # Rule 5: $ref nodes and retained union nodes are exempt from type
    # inference.  A kept anyOf/oneOf has no single type — synthesizing one
    # (e.g. the ``string`` fallback) would wrongly constrain the union.
    if (
        "$ref" not in repaired
        and "anyOf" not in repaired
        and "oneOf" not in repaired
    ):
        repaired = _fill_missing_type(repaired)

    # Rule 6: Object schemas must carry a `required` array.
    if repaired.get("type") == "object":
        repaired = _ensure_required_array(repaired)

    return repaired


def _ensure_required_array(node: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee an object schema carries a ``required`` array (MiniMax rule).

    Standard JSON Schema lets you omit ``required`` when nothing is required;
    MiniMax rejects that.  Ensure the key is a list.  When ``properties`` is
    known, prune ``required`` entries that don't name a real property —
    defensive against dangling names.  Mutates and returns ``node``.
    """
    props = node.get("properties")
    req = node.get("required")
    if isinstance(req, list):
        if isinstance(props, dict):
            node["required"] = [r for r in req if r in props]
    else:
        node["required"] = []
    return node


def _fill_missing_type(node: Dict[str, Any]) -> Dict[str, Any]:
    """Infer a reasonable ``type`` if this schema node has none."""
    node_type = node.get("type")
    if isinstance(node_type, list):
        concrete = next(
            (t for t in node_type if isinstance(t, str) and t not in {"", "null"}),
            "string",
        )
        return {**node, "type": concrete}
    if "type" in node and node_type not in {None, ""}:
        return node

    # Heuristic: presence of properties → object, items → array, enum → string,
    # else fall back to ``string`` (safest scalar).
    if "properties" in node or "required" in node or "additionalProperties" in node:
        inferred = "object"
    elif "items" in node or "prefixItems" in node:
        inferred = "array"
    elif "enum" in node and isinstance(node["enum"], list) and node["enum"]:
        inferred = "string"
    else:
        inferred = "string"

    return {**node, "type": inferred}


def sanitize_minimax_tool_parameters(parameters: Any) -> Dict[str, Any]:
    """Normalize tool parameters to a MiniMax-compatible object schema.

    Returns a deep-copied schema with all MiniMax repair rules applied.
    Input is not mutated.
    """
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}, "required": []}

    repaired = _repair_schema(copy.deepcopy(parameters), is_schema=True)
    if not isinstance(repaired, dict):
        return {"type": "object", "properties": {}, "required": []}

    # Top-level must be an object schema
    if repaired.get("type") != "object":
        repaired["type"] = "object"
    if "properties" not in repaired:
        repaired["properties"] = {}
    _ensure_required_array(repaired)

    return repaired


def sanitize_minimax_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply ``sanitize_minimax_tool_parameters`` to every tool's parameters.

    Handles both Anthropic-style tool schemas (``input_schema``) and
    OpenAI-style tool schemas (``function.parameters``).

    Returns a new list with sanitized tools if any changes were made,
    or the original list if nothing needed repair.
    """
    if not tools:
        return tools

    sanitized: List[Dict[str, Any]] = []
    any_change = False

    for tool in tools:
        if not isinstance(tool, dict):
            sanitized.append(tool)
            continue

        tool_copy: Dict[str, Any] = copy.deepcopy(tool)

        # --- OpenAI-style: tool["function"]["parameters"] ---
        fn = tool_copy.get("function")
        if isinstance(fn, dict):
            params = fn.get("parameters")
            repaired = sanitize_minimax_tool_parameters(params)
            if repaired is not params:
                any_change = True
                fn["parameters"] = repaired

        # --- Anthropic-style: tool["input_schema"] ---
        input_schema = tool_copy.get("input_schema")
        if isinstance(input_schema, dict):
            # Anthropic input_schema is already the parameters dict itself
            repaired = sanitize_minimax_tool_parameters(input_schema)
            if repaired is not input_schema:
                any_change = True
                tool_copy["input_schema"] = repaired

        sanitized.append(tool_copy)

    return sanitized if any_change else tools


def is_minimax_model(model: str | None) -> bool:
    """True for any MiniMax model slug, regardless of aggregator prefix.

    Matches bare names (``minimax-m3``, ``MiniMax-M2.7``, ``MiniMax3``),
    vendor-prefixed slugs (``minimax/MiniMax-M3``),
    and aggregator-prefixed slugs (``nous/minimax/minimax-m3``,
    ``openrouter/minimax/...``).

    Detection by model name covers aggregators that route to MiniMax's
    inference where the base URL is the aggregator's, not api.minimax.io.

    Model name patterns matched:
    - ``minimax-m3``, ``minimax-m2.7``, ``minimax-m2.5`` (and any minimax-m*)
    - ``MiniMax-M3``, ``MiniMax-M2.7`` (case-insensitive)
    - ``minimax/MiniMax-M3``, ``openrouter/minimax/...``
    - ``MiniMax3``, ``minimax3`` (compact form)
    - ``MiniMax-M2.7-free``, ``minimax-m2.5-free`` (suffixed variants)
    """
    if not model:
        return False
    bare = model.strip().lower()

    # Last path segment (covers aggregator-prefixed slugs like
    # "openrouter/minimax/minimax-m3")
    tail = bare.rsplit("/", 1)[-1]

    # Check against known MiniMax model prefixes
    for prefix in _MINIMAX_MODEL_PREFIXES:
        if tail == prefix or tail.startswith(prefix + ".") or tail.startswith(prefix + "-"):
            return True

    # Also match the form "minimax3" (no hyphen) and "mini-max-*"
    if tail.startswith("minimax3") or tail.startswith("mini-max-"):
        return True

    # Vendor-prefixed forms — "minimax/" in the path
    if "/minimax/" in bare or bare.startswith("minimax/"):
        return True

    # Catch-all: the bare word "minimax" anywhere is a strong signal
    if "minimax" in bare:
        # But avoid false matches on non-MiniMax slugs that happen to
        # contain the substring — "minimax" is distinctive enough.
        return True

    return False
