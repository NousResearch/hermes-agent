"""Moonshot repair must prune dangling ``required`` names at every depth.

``agent/moonshot_schema`` documents that Moonshot rejects ``required`` entries
naming a property the node does not declare. ``_ensure_required_array`` only
pruned when the node already carried a ``properties`` dict, so a node with
``required`` and NO ``properties`` — every entry dangling by definition — kept
them.

The top-level parameters object was rescued only incidentally, because
``sanitize_moonshot_tool_parameters`` injects ``properties: {}`` before the
repair runs. Nested nodes had no such rescue, and that is exactly the shape MCP
servers emit: array item schemas carrying ``required`` without ``properties``
(the same class is called out in ``agent/gemini_schema.py``). One such tool
400s the entire request before any model output.
"""

from __future__ import annotations

import pytest

from agent.moonshot_schema import sanitize_moonshot_tool_parameters as sanitize


def _walk(node):
    """Yield every schema node in the repaired output."""
    if isinstance(node, dict):
        yield node
        for key, val in node.items():
            if key in {"properties", "patternProperties", "$defs", "definitions"} and isinstance(val, dict):
                for sub in val.values():
                    yield from _walk(sub)
            elif key in {"anyOf", "oneOf", "allOf", "prefixItems"} and isinstance(val, list):
                for sub in val:
                    yield from _walk(sub)
            elif key in {"items", "contains", "not", "additionalProperties", "propertyNames"}:
                yield from _walk(val)
    elif isinstance(node, list):
        for sub in node:
            yield from _walk(sub)


def _dangling(out) -> list:
    """Every (node, name) pair where required names an undeclared property."""
    bad = []
    for node in _walk(out):
        req = node.get("required")
        if not isinstance(req, list):
            continue
        props = node.get("properties")
        known = set(props) if isinstance(props, dict) else set()
        bad += [(node.get("type"), r) for r in req if r not in known]
    return bad


class TestNoDanglingRequiredSurvives:
    def test_array_item_schema_with_required_and_no_properties(self):
        """The MCP shape: items carry `required` but declare no properties."""
        out = sanitize({
            "type": "object",
            "properties": {
                "files": {"type": "array", "items": {"type": "object", "required": ["path"]}},
            },
            "required": ["files"],
        })
        assert _dangling(out) == []
        assert out["properties"]["files"]["items"]["required"] == []

    def test_nested_object_with_inferred_type(self):
        """`required` alone infers type=object; its names are still dangling."""
        out = sanitize({
            "type": "object",
            "properties": {"opts": {"required": ["mode"]}},
            "required": [],
        })
        assert _dangling(out) == []
        assert out["properties"]["opts"]["required"] == []

    def test_deeply_nested_under_anyof(self):
        out = sanitize({
            "type": "object",
            "properties": {
                "target": {"anyOf": [
                    {"type": "object", "required": ["ghost"]},
                    {"type": "string"},
                ]},
            },
            "required": ["target"],
        })
        assert _dangling(out) == []

    def test_top_level_still_pruned(self):
        out = sanitize({"type": "object", "required": ["missing_prop"]})
        assert out["required"] == []


class TestExistingBehaviourPreserved:
    def test_valid_required_names_survive(self):
        out = sanitize({
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        })
        assert out["required"] == ["a", "b"]

    def test_partial_prune_keeps_the_real_name(self):
        out = sanitize({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a", "ghost"],
        })
        assert out["required"] == ["a"]

    def test_missing_required_still_becomes_empty_list(self):
        """Rule: Moonshot needs the key present even when nothing is required."""
        out = sanitize({"type": "object", "properties": {"a": {"type": "string"}}})
        assert out["required"] == []

    def test_nested_object_keeps_its_valid_required(self):
        out = sanitize({
            "type": "object",
            "properties": {
                "cfg": {"type": "object", "properties": {"k": {"type": "string"}}, "required": ["k"]},
            },
            "required": ["cfg"],
        })
        assert out["properties"]["cfg"]["required"] == ["k"]

    def test_every_object_node_carries_a_required_array(self):
        out = sanitize({
            "type": "object",
            "properties": {"o": {"type": "object", "properties": {"x": {"type": "string"}}}},
            "required": ["o"],
        })
        for node in _walk(out):
            if node.get("type") == "object":
                assert isinstance(node.get("required"), list)


class TestRealToolSchemas:
    def test_no_builtin_tool_produces_a_dangling_required(self):
        import model_tools  # noqa: F401  (triggers tool discovery)
        from tools.registry import registry

        checked = 0
        for entry in getattr(registry, "_tools", {}).values():
            sc = entry.get("schema") if isinstance(entry, dict) else getattr(entry, "schema", None)
            if not isinstance(sc, dict):
                continue
            fn = sc.get("function") if "function" in sc else sc
            params = fn.get("parameters") if isinstance(fn, dict) else None
            if not isinstance(params, dict):
                continue
            checked += 1
            assert _dangling(sanitize(params)) == []
        assert checked > 0
