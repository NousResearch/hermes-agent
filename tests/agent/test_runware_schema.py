"""Tests for the Runware tool-schema sanitizer.

Runware's OpenAI-compatible endpoint rejects zero-argument tools that use
the otherwise-universal ``{"type": "object", "properties": {}}`` shape:

    HTTP 400: Invalid value for 'tools[N].schema.properties'. Function
    schema properties must be a non-empty object.

These tests cover the placeholder-property patch applied by
``agent/runware_schema.py``.
"""

from agent.runware_schema import is_runware_provider, sanitize_runware_tools


def _tool(name: str, parameters: dict | None) -> dict:
    fn: dict = {"name": name}
    if parameters is not None:
        fn["parameters"] = parameters
    return {"type": "function", "function": fn}


class TestSanitizeRunwareTools:
    def test_empty_properties_gets_placeholder(self):
        tools = [_tool("get_time", {"type": "object", "properties": {}})]
        sanitized = sanitize_runware_tools(tools)

        props = sanitized[0]["function"]["parameters"]["properties"]
        assert props
        assert sanitized[0]["function"]["parameters"]["type"] == "object"

    def test_missing_properties_key_gets_placeholder(self):
        tools = [_tool("get_time", {"type": "object"})]
        sanitized = sanitize_runware_tools(tools)

        assert sanitized[0]["function"]["parameters"]["properties"]

    def test_tool_with_real_parameters_untouched(self):
        params = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        tools = [_tool("read_file", params)]
        sanitized = sanitize_runware_tools(tools)

        assert sanitized[0]["function"]["parameters"] == params

    def test_mixed_list_only_patches_empty_ones(self):
        real_params = {"type": "object", "properties": {"path": {"type": "string"}}}
        tools = [
            _tool("read_file", real_params),
            _tool("list_dir", {"type": "object", "properties": {}}),
        ]
        sanitized = sanitize_runware_tools(tools)

        assert sanitized[0]["function"]["parameters"] == real_params
        assert sanitized[1]["function"]["parameters"]["properties"]

    def test_does_not_mutate_input(self):
        original = {"type": "object", "properties": {}}
        tools = [_tool("get_time", original)]
        sanitize_runware_tools(tools)

        assert original == {"type": "object", "properties": {}}

    def test_empty_tools_list_passthrough(self):
        assert sanitize_runware_tools([]) == []

    def test_no_change_returns_same_list_when_nothing_patched(self):
        params = {"type": "object", "properties": {"path": {"type": "string"}}}
        tools = [_tool("read_file", params)]
        assert sanitize_runware_tools(tools) is tools

    def test_idempotent_after_moonshot_sanitizer(self):
        """Regression: Runware's own Kimi models (e.g. moonshotai-kimi-k2-6)
        also match is_moonshot_model() by name, since the slug contains
        "moonshot"/"kimi". Moonshot's sanitizer runs first and re-adds an
        EMPTY properties dict for parameter-less tools (it only guarantees
        the key exists, not that it's non-empty) — Runware's sanitizer must
        still see and fix that empty dict when applied afterward.
        """
        from agent.moonshot_schema import sanitize_moonshot_tools

        tools = [_tool("get_time", {"type": "object"})]  # no properties key at all
        after_moonshot = sanitize_moonshot_tools(tools)
        assert after_moonshot[0]["function"]["parameters"]["properties"] == {}

        after_runware = sanitize_runware_tools(after_moonshot)
        assert after_runware[0]["function"]["parameters"]["properties"]


class TestIsRunwareProvider:
    def test_matches_runware_profile(self):
        class FakeProfile:
            name = "runware"

        assert is_runware_provider(FakeProfile()) is True

    def test_rejects_other_profiles(self):
        class FakeProfile:
            name = "gmi"

        assert is_runware_provider(FakeProfile()) is False

    def test_rejects_none(self):
        assert is_runware_provider(None) is False
