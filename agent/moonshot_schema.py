"""Helpers for translating OpenAI-style tool schemas to Moonshot's schema subset.

Moonshot (Kimi) accepts a stricter subset of JSON Schema than standard OpenAI
tool calling.  Requests that violate it fail with HTTP 400:

    tools.function.parameters is not a valid moonshot flavored json schema,
    details: <...>

Known rejection modes documented at
https://forum.moonshot.ai/t/tool-calling-specification-violation-on-moonshot-api/102
and MoonshotAI/kimi-cli#1595:

1. Every property schema must carry a ``type``.  Standard JSON Schema allows
   type to be omitted (the value is then unconstrained); Moonshot refuses.
2. When ``anyOf`` is used, ``type`` must be on the ``anyOf`` children, not
   the parent.  Presence of both causes "type should be defined in anyOf
   items instead of the parent schema".

The ``#/definitions/...`` → ``#/$defs/...`` rewrite for draft-07 refs is
handled separately in ``tools/mcp_tool._normalize_mcp_input_schema`` so it
applies at MCP registration time for all providers.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

# Keys whose values are maps of name → schema (not schemas themselves).
# When we recurse, we walk the values of these maps as schemas, but we do
# NOT apply the missing-type repair to the map itself.
_SCHEMA_MAP_KEYS = frozenset({"properties", "patternProperties", "$defs", "definitions"})

# Keys whose values are lists of schemas.
_SCHEMA_LIST_KEYS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})

# Keys whose values are a single nested schema.
_SCHEMA_NODE_KEYS = frozenset({"items", "contains", "not", "additionalProperties", "propertyNames"})


def _repair_schema(node: Any, is_schema: bool = True) -> Any:
    """Recursively apply Moonshot repairs to a schema node.

    ``is_schema=True`` means this dict is a JSON Schema node and gets the
    missing-type + anyOf-parent repairs applied.  ``is_schema=False`` means
    it's a container map (e.g. the value of ``properties``) and we only
    recurse into its values.
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

    # Rule 2: when anyOf is present, type belongs only on the children.
    # Additionally, Moonshot rejects null-type branches inside anyOf
    # (enum value (<nil>) does not match any type in [string]).
    # Collapse the anyOf to the first non-null branch and infer its type.
    if "anyOf" in repaired and isinstance(repaired["anyOf"], list):
        repaired.pop("type", None)
        non_null = [b for b in repaired["anyOf"]
                    if isinstance(b, dict) and b.get("type") != "null"]
        if non_null and len(non_null) < len(repaired["anyOf"]):
            # Drop the anyOf wrapper — keep only the non-null branch.
            # If there's a single non-null branch, promote it and fall
            # through to Rules 1/3 so nullable/enum cleanup still applies
            # to the merged node.
            if len(non_null) == 1:
                merge = {k: v for k, v in repaired.items() if k != "anyOf"}
                merge.update(non_null[0])
                repaired = merge
            else:
                repaired["anyOf"] = non_null
                return repaired
        else:
            # Nothing to collapse — parent type stripped, children already
            # repaired by the recursive walk above.
            return repaired

    # Moonshot also rejects non-standard keywords like ``nullable`` on
    # parameter schemas — strip it.
    repaired.pop("nullable", None)

    # Rule 1: property schemas without type need one.  $ref nodes are exempt
    # — their type comes from the referenced definition.
    # Fill missing type BEFORE Rule 3 so enum cleanup can check the type.
    if "$ref" not in repaired:
        repaired = _fill_missing_type(repaired)

    # Rule 3: Moonshot rejects null/empty-string values inside enum arrays
    # when the parent type is a scalar (string, integer, etc.).  The error:
    #   "enum value (<nil>) does not match any type in [string]"
    # Strip null and empty-string from enum values, and if the enum becomes
    # empty, drop it entirely.
    if "enum" in repaired and isinstance(repaired["enum"], list):
        node_type = repaired.get("type")
        if node_type in {"string", "integer", "number", "boolean"}:
            cleaned = [v for v in repaired["enum"]
                       if v is not None and v != ""]
            if cleaned:
                repaired["enum"] = cleaned
            else:
                repaired.pop("enum")

    return repaired


def _fill_missing_type(node: Dict[str, Any]) -> Dict[str, Any]:
    """Infer a reasonable ``type`` if this schema node has none."""
    if "type" in node and node["type"] not in {None, ""}:
        return node

    # Heuristic: presence of ``properties`` → object, ``items`` → array, ``enum``
    # → type of first enum value, else fall back to ``string`` (safest scalar).
    if "properties" in node or "required" in node or "additionalProperties" in node:
        inferred = "object"
    elif "items" in node or "prefixItems" in node:
        inferred = "array"
    elif "enum" in node and isinstance(node["enum"], list) and node["enum"]:
        sample = node["enum"][0]
        if isinstance(sample, bool):
            inferred = "boolean"
        elif isinstance(sample, int):
            inferred = "integer"
        elif isinstance(sample, float):
            inferred = "number"
        else:
            inferred = "string"
    else:
        inferred = "string"

    return {**node, "type": inferred}


def sanitize_moonshot_tool_parameters(parameters: Any) -> Dict[str, Any]:
    """Normalize tool parameters to a Moonshot-compatible object schema.

    Returns a deep-copied schema with the two flavored-JSON-Schema repairs
    applied.  Input is not mutated.
    """
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}

    repaired = _repair_schema(copy.deepcopy(parameters), is_schema=True)
    if not isinstance(repaired, dict):
        return {"type": "object", "properties": {}}

    # Top-level must be an object schema
    if repaired.get("type") != "object":
        repaired["type"] = "object"
    if "properties" not in repaired:
        repaired["properties"] = {}

    return repaired


def sanitize_moonshot_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply ``sanitize_moonshot_tool_parameters`` to every tool's parameters."""
    if not tools:
        return tools

    sanitized: List[Dict[str, Any]] = []
    any_change = False
    for tool in tools:
        if not isinstance(tool, dict):
            sanitized.append(tool)
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            sanitized.append(tool)
            continue
        params = fn.get("parameters")
        repaired = sanitize_moonshot_tool_parameters(params)
        if repaired is not params:
            any_change = True
            new_fn = {**fn, "parameters": repaired}
            sanitized.append({**tool, "function": new_fn})
        else:
            sanitized.append(tool)

    return sanitized if any_change else tools


def is_moonshot_model(model: str | None) -> bool:
    """True for any Kimi / Moonshot model slug, regardless of aggregator prefix.

    Matches bare names (``kimi-k2.6``, ``moonshotai/Kimi-K2.6``) and aggregator-
    prefixed slugs (``nous/moonshotai/kimi-k2.6``, ``openrouter/moonshotai/...``).
    Detection by model name covers Nous / OpenRouter / other aggregators that
    route to Moonshot's inference, where the base URL is the aggregator's, not
    ``api.moonshot.ai``.
    """
    if not model:
        return False
    bare = model.strip().lower()
    # Last path segment (covers aggregator-prefixed slugs)
    tail = bare.rsplit("/", 1)[-1]
    if tail.startswith("kimi-") or tail == "kimi":
        return True
    # Vendor-prefixed forms commonly used on aggregators
    if "moonshot" in bare or "/kimi" in bare or bare.startswith("kimi"):
        return True
    return False
