"""Tests for agent.gemini_schema — OpenAI→Gemini tool parameter translation."""

import copy

from agent.gemini_schema import (
    prepare_gemini_tool_parameters,
    sanitize_gemini_schema,
    sanitize_gemini_tool_parameters,
)


class TestSanitizeGeminiSchema:
    def test_strips_unknown_top_level_keys(self):
        """$schema / additionalProperties etc. must not reach Gemini."""
        schema = {
            "type": "object",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "additionalProperties": False,
            "properties": {"foo": {"type": "string"}},
        }
        cleaned = sanitize_gemini_schema(schema)
        assert "$schema" not in cleaned
        assert "additionalProperties" not in cleaned
        assert cleaned["type"] == "object"
        assert cleaned["properties"] == {"foo": {"type": "string"}}


    def test_stringifies_integer_enum_to_satisfy_gemini(self):
        """Gemini rejects numeric enum metadata unless values are strings.

        Regression for the Discord tool's ``auto_archive_duration``:
        ``{type: integer, enum: [60, 1440, 4320, 10080]}`` caused
        Gemini HTTP 400 INVALID_ARGUMENT
        "Invalid value ... (TYPE_STRING), 60" on every request that
        shipped the full tool catalog to generativelanguage.googleapis.com.
        """
        schema = {
            "type": "integer",
            "enum": [60, 1440, 4320, 10080],
            "description": "Minutes (60, 1440, 4320, 10080).",
        }
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "integer"
        assert cleaned["enum"] == ["60", "1440", "4320", "10080"]
        # Description remains useful model guidance.
        assert cleaned["description"].startswith("Minutes")





    def test_stringifies_nested_integer_enum_inside_properties(self):
        """The fix must apply recursively — the Discord case is nested."""
        schema = {
            "type": "object",
            "properties": {
                "auto_archive_duration": {
                    "type": "integer",
                    "enum": [60, 1440, 4320, 10080],
                    "description": "Thread archive duration in minutes.",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "archived"],
                },
            },
        }
        cleaned = sanitize_gemini_schema(schema)
        props = cleaned["properties"]
        # Integer enum is retained as Gemini-compatible string metadata...
        assert props["auto_archive_duration"]["type"] == "integer"
        assert props["auto_archive_duration"]["enum"] == ["60", "1440", "4320", "10080"]
        # ...but the sibling string enum is preserved.
        assert props["status"]["enum"] == ["active", "archived"]



    def test_non_dict_input_returns_empty(self):
        assert sanitize_gemini_schema(None) == {}
        assert sanitize_gemini_schema("not a schema") == {}
        assert sanitize_gemini_schema([1, 2, 3]) == {}


class TestRequiredPropertyPruning:
    """Gemini rejects ``required`` names missing from the node's ``properties``.

    Regression for the Kilo-Org/kilocode#11955 bug class: MCP servers (e.g.
    the GitHub remote MCP) emit array item schemas whose ``required`` lists
    reference properties that don't exist in the same node — Google fails the
    entire GenerateContentRequest with HTTP 400 "property is not defined".
    """



    def test_prunes_inside_array_items(self):
        """The exact shape from the GitHub MCP report — nested in items."""
        schema = {
            "type": "object",
            "properties": {
                "issue_fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["field_id", "value"],
                    },
                },
            },
            "required": ["issue_fields"],
        }
        cleaned = sanitize_gemini_schema(schema)
        items = cleaned["properties"]["issue_fields"]["items"]
        assert "required" not in items
        # Top-level required is valid and survives.
        assert cleaned["required"] == ["issue_fields"]


    def test_valid_required_untouched(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["required"] == ["a", "b"]


    def test_prunes_inside_anyof_branches(self):
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x", "ghost"],
                },
                {"type": "object", "required": ["orphan"]},
            ]
        }
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["anyOf"][0]["required"] == ["x"]
        assert "required" not in cleaned["anyOf"][1]


class TestSanitizeGeminiToolParameters:
    def test_empty_parameters_return_valid_object_schema(self):
        """Gemini requires ``parameters`` to be a valid object schema."""
        cleaned = sanitize_gemini_tool_parameters({})
        assert cleaned == {"type": "object", "properties": {}}

    def test_discord_create_thread_parameters_no_longer_trip_gemini(self):
        """End-to-end regression: the exact shape that was rejected in prod."""
        params = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create_thread"]},
                "auto_archive_duration": {
                    "type": "integer",
                    "enum": [60, 1440, 4320, 10080],
                    "description": "Thread archive duration in minutes "
                    "(create_thread, default 1440).",
                },
            },
            "required": ["action"],
        }
        cleaned = sanitize_gemini_tool_parameters(params)
        aad = cleaned["properties"]["auto_archive_duration"]
        # The field that triggered the Gemini 400 is now string metadata.
        assert aad["enum"] == ["60", "1440", "4320", "10080"]
        # Type + description survive so the model still knows what to send.
        assert aad["type"] == "integer"
        assert "1440" in aad["description"]
        # And the string-enum sibling is untouched.
        assert cleaned["properties"]["action"]["enum"] == ["create_thread"]


# ── parametersJsonSchema (full JSON Schema) path ─────────────────────────────

class TestPrepareGeminiToolParameters:
    def test_full_json_schema_survives_and_input_is_not_mutated(self):
        params = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "globs": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                "bare_array": {"type": "array"},
                "opts": {"type": "object", "additionalProperties": {"type": "string"}},
            },
            "required": ["globs"],
        }
        original = copy.deepcopy(params)
        out = prepare_gemini_tool_parameters(params)
        assert params == original
        assert "$schema" not in out
        assert out["properties"]["globs"]["anyOf"][1] == {"type": "array", "items": {"type": "string"}}
        assert out["properties"]["bare_array"] == {"type": "array"}
        assert out["properties"]["opts"]["additionalProperties"] == {"type": "string"}
        assert prepare_gemini_tool_parameters({}) == {"type": "object", "properties": {}}

    def test_local_refs_inlined_with_sibling_override_and_defs_dropped(self):
        params = {
            "type": "object",
            "properties": {"user": {"$ref": "#/$defs/User", "description": "who"}},
            "$defs": {"User": {"type": "object", "description": "generic", "properties": {
                "tags": {"type": "array", "items": {"$ref": "#/$defs/Tag"}}}},
                      "Tag": {"type": "string"}},
        }
        out = prepare_gemini_tool_parameters(params)
        assert "$defs" not in out
        user = out["properties"]["user"]
        assert user["description"] == "who"
        assert user["properties"]["tags"]["items"] == {"type": "string"}

    def test_unresolvable_or_circular_ref_passes_schema_through_untouched(self):
        for defs in ({"Loop": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Loop"}}}}, {}):
            params = {"type": "object", "properties": {"p": {"$ref": "#/$defs/Loop"}}, "$defs": defs}
            out = prepare_gemini_tool_parameters(params)
            assert out["properties"]["p"] == {"$ref": "#/$defs/Loop"}
            assert out["$defs"] == defs


class TestAdapterWireShape:
    _TOOLS = [{"type": "function", "function": {"name": "t", "parameters": {
        "type": "object", "properties": {"g": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}}}}}]

    def test_v1beta_sends_parameters_json_schema_other_versions_send_legacy_subset(self):
        from agent.gemini_native_adapter import GeminiNativeClient

        def decl(base_url):
            client = GeminiNativeClient(api_key="k", base_url=base_url)
            captured = {}
            client._http = type("H", (), {"post": lambda self, url, json, headers, timeout: captured.update(json) or
                                  type("R", (), {"status_code": 200, "json": lambda self: {"candidates": []}})()})()
            client._create_chat_completion(model="gemini-2.5-flash", messages=[{"role": "user", "content": "hi"}], tools=self._TOOLS)
            return captured["tools"][0]["functionDeclarations"][0]

        beta = decl("https://generativelanguage.googleapis.com/v1beta")
        assert "parameters" not in beta
        assert beta["parametersJsonSchema"]["properties"]["g"]["anyOf"][1]["items"] == {"type": "string"}
        v1 = decl("https://generativelanguage.googleapis.com/v1")
        assert "parametersJsonSchema" not in v1
        assert v1["parameters"]["properties"]["g"]["anyOf"]  # legacy translator still applied
