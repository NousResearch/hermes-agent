"""Tests for agent.gemini_schema — full-JSON-Schema normalization for Gemini.

Hermes sends tool schemas to Gemini through ``parametersJsonSchema`` (plain
JSON Schema) instead of down-translating into the legacy ``parameters``
subset (clean-room port of zed-industries/zed#63342).  These tests pin the
normalizer's contract: lossless passthrough of full JSON Schema constructs,
$ref inlining for MCP/pydantic-shaped schemas, and untouched passthrough
when references cannot be resolved.
"""

import copy

from agent.gemini_schema import prepare_gemini_tool_parameters


class TestFullSchemaPassthrough:
    def test_anyof_union_survives(self):
        # The exact shape the legacy subset translator broke: a union whose
        # outer node carries no ``type`` (pydantic ``str | list[str]``).
        schema = {
            "type": "object",
            "properties": {
                "globs": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "one or many globs",
                }
            },
            "required": ["globs"],
        }
        out = prepare_gemini_tool_parameters(schema)
        assert out["properties"]["globs"]["anyOf"] == schema["properties"]["globs"]["anyOf"]
        assert out["required"] == ["globs"]

    def test_bare_array_and_additional_properties_survive(self):
        schema = {
            "type": "object",
            "properties": {"tags": {"type": "array"}},
            "additionalProperties": False,
        }
        out = prepare_gemini_tool_parameters(schema)
        assert out["properties"]["tags"] == {"type": "array"}
        assert out["additionalProperties"] is False

    def test_root_dollar_schema_stripped(self):
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
        }
        out = prepare_gemini_tool_parameters(schema)
        assert "$schema" not in out

    def test_input_never_mutated(self):
        schema = {
            "$schema": "x",
            "type": "object",
            "properties": {"p": {"$ref": "#/$defs/P"}},
            "$defs": {"P": {"type": "integer"}},
        }
        snapshot = copy.deepcopy(schema)
        prepare_gemini_tool_parameters(schema)
        assert schema == snapshot

    def test_empty_or_invalid_input_returns_object_schema(self):
        assert prepare_gemini_tool_parameters(None) == {"type": "object", "properties": {}}
        assert prepare_gemini_tool_parameters({}) == {"type": "object", "properties": {}}
        assert prepare_gemini_tool_parameters("nope") == {"type": "object", "properties": {}}

    def test_object_root_gains_properties(self):
        out = prepare_gemini_tool_parameters({"type": "object"})
        assert out == {"type": "object", "properties": {}}


class TestRefInlining:
    def test_pydantic_style_defs_inlined(self):
        schema = {
            "type": "object",
            "properties": {"payload": {"$ref": "#/$defs/Payload"}},
            "$defs": {
                "Payload": {"type": "object", "properties": {"x": {"type": "integer"}}}
            },
        }
        out = prepare_gemini_tool_parameters(schema)
        assert out["properties"]["payload"]["type"] == "object"
        assert out["properties"]["payload"]["properties"]["x"] == {"type": "integer"}
        assert "$defs" not in out

    def test_legacy_definitions_inlined(self):
        schema = {
            "type": "object",
            "properties": {"p": {"$ref": "#/definitions/P"}},
            "definitions": {"P": {"type": "string"}},
        }
        out = prepare_gemini_tool_parameters(schema)
        assert out["properties"]["p"]["type"] == "string"
        assert "definitions" not in out

    def test_ref_siblings_preserved(self):
        # JSON Schema allows annotations next to $ref; they must survive.
        schema = {
            "type": "object",
            "properties": {
                "p": {"$ref": "#/$defs/P", "description": "the payload"}
            },
            "$defs": {"P": {"type": "string"}},
        }
        out = prepare_gemini_tool_parameters(schema)
        assert out["properties"]["p"]["type"] == "string"
        assert out["properties"]["p"]["description"] == "the payload"

    def test_nested_refs_inlined(self):
        schema = {
            "type": "object",
            "properties": {"outer": {"$ref": "#/$defs/Outer"}},
            "$defs": {
                "Outer": {
                    "type": "object",
                    "properties": {"inner": {"$ref": "#/$defs/Inner"}},
                },
                "Inner": {"type": "boolean"},
            },
        }
        out = prepare_gemini_tool_parameters(schema)
        assert (
            out["properties"]["outer"]["properties"]["inner"]["type"] == "boolean"
        )

    def test_unresolvable_ref_passes_schema_through_untouched(self):
        # Half-rewriting hides the real problem; the provider's error names
        # the bad pointer. Only the root $schema metadata is dropped.
        schema = {
            "$schema": "x",
            "type": "object",
            "properties": {"p": {"$ref": "#/$defs/Missing"}},
        }
        out = prepare_gemini_tool_parameters(schema)
        assert out["properties"]["p"] == {"$ref": "#/$defs/Missing"}
        assert "$schema" not in out

    def test_circular_ref_passes_schema_through_untouched(self):
        schema = {
            "type": "object",
            "properties": {"p": {"$ref": "#/$defs/Loop"}},
            "$defs": {"Loop": {"properties": {"again": {"$ref": "#/$defs/Loop"}}}},
        }
        out = prepare_gemini_tool_parameters(schema)
        # Original shape retained, including its $defs (still referenced).
        assert out["properties"]["p"] == {"$ref": "#/$defs/Loop"}
        assert "$defs" in out


class TestAdapterWiring:
    def test_translate_tools_emits_parameters_json_schema(self):
        from agent.gemini_native_adapter import _translate_tools_to_gemini

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "union_tool",
                    "description": "d",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "globs": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}},
                                ]
                            }
                        },
                    },
                },
            }
        ]
        out = _translate_tools_to_gemini(tools)
        decl = out[0]["functionDeclarations"][0]
        assert "parametersJsonSchema" in decl
        assert "parameters" not in decl
        assert decl["parametersJsonSchema"]["properties"]["globs"]["anyOf"]
