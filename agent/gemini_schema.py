"""Helpers for preparing OpenAI-style tool schemas for Gemini's native API.

Historical note: Hermes used to translate every tool schema into the
restricted OpenAPI-flavored ``Schema`` subset that Gemini's legacy
``FunctionDeclaration.parameters`` field accepts.  That translation was
inherently lossy — ``anyOf`` unions whose outer node carries no ``type``,
bare arrays without ``items``, ``$ref``/``$defs`` indirection, and
``additionalProperties`` all had to be stripped or repaired, and a single
unrepresentable construct could 400 the ENTIRE request (every tool lost,
before any model output).

Google now ships ``parametersJsonSchema`` alongside ``parameters``, which
accepts plain JSON Schema on all current Gemini models.  Hermes sends full
schemas through that field instead of down-translating (clean-room port of
the approach in zed-industries/zed#63342).  What remains here is a light
normalizer, not a dialect translator:

- deep-copies (callers' schemas are never mutated),
- strips root-level ``$schema`` (pure metadata, costs tokens),
- inlines same-document ``$ref`` pointers into ``$defs``/``definitions``
  when all of them resolve — MCP's Python SDK (pydantic) and
  zod-to-json-schema both emit reference indirection for nested models,
  and some Google routes reject references outright.  A schema whose
  references cannot all be resolved is passed through untouched (with the
  reason logged) so the provider reports what is wrong instead of
  receiving something half-rewritten,
- guarantees an object root (``{"type": "object", "properties": {}}``)
  for empty/invalid input, which the API requires.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_EMPTY_OBJECT_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {}}

# How many total $ref inlinings we allow before assuming a pathological or
# circular schema.  Real tool schemas contain a handful of refs; circular
# pydantic models could otherwise expand forever.
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
    """Recursively inline same-document ``$ref`` nodes.

    Raises ``ValueError`` when a reference cannot be resolved, a cycle is
    detected, or the expansion budget is exhausted — the caller then keeps
    the original schema untouched.
    """
    if isinstance(node, list):
        return [_inline_refs(item, root, budget, stack) for item in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str):
        if ref in stack:
            raise ValueError(f"circular $ref {ref!r}")
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError("$ref expansion budget exhausted")
        target = _resolve_local_ref(root, ref)
        if target is None:
            raise ValueError(f"unresolvable $ref {ref!r}")
        inlined = _inline_refs(target, root, budget, stack + (ref,))
        # JSON Schema: siblings of $ref (description, default, ...) apply
        # alongside the referenced schema; keep them, referenced keys win
        # only where the sibling doesn't override.
        siblings = {k: v for k, v in node.items() if k != "$ref"}
        if siblings:
            merged = dict(inlined)
            merged.update(_inline_refs(siblings, root, budget, stack))
            return merged
        return inlined

    return {
        key: _inline_refs(value, root, budget, stack)
        for key, value in node.items()
    }


def prepare_gemini_tool_parameters(parameters: Any) -> Dict[str, Any]:
    """Normalize tool ``parameters`` for Gemini's ``parametersJsonSchema``.

    Returns full JSON Schema (NOT the legacy restricted subset).  See the
    module docstring for exactly what is normalized.
    """
    if not isinstance(parameters, dict) or not parameters:
        return dict(_EMPTY_OBJECT_SCHEMA)

    schema = copy.deepcopy(parameters)
    schema.pop("$schema", None)

    try:
        schema = _inline_refs(schema, schema, [_MAX_REF_EXPANSIONS])
    except ValueError as exc:
        logger.debug(
            "Gemini tool schema kept as-is ($ref inlining skipped): %s", exc
        )
        # Keep the original (minus root $schema): a provider-side error names
        # the real problem, a half-rewritten schema wouldn't.
        schema = copy.deepcopy(parameters)
        schema.pop("$schema", None)
        return schema

    # Reference definitions are dead weight once everything is inlined.
    if isinstance(schema, dict):
        schema.pop("$defs", None)
        schema.pop("definitions", None)

    if not isinstance(schema, dict) or not schema:
        return dict(_EMPTY_OBJECT_SCHEMA)
    if schema.get("type") == "object" and "properties" not in schema:
        schema["properties"] = {}
    return schema
