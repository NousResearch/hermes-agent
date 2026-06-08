"""Unit tests for agent.structured_output — the structured-output mapper.

Covers the canonical normalizers (Chat Completions + Responses shapes), the
constraint-aware support check, and the per-api_mode wire mapping (apply).
"""

import pytest

from agent.structured_output import (
    SUPPORTED_API_MODES,
    apply,
    normalize_response_format,
    normalize_responses_text_format,
    unsupported_reason,
)

_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
    "additionalProperties": False,
}


class TestNormalizeResponseFormat:
    def test_none_returns_no_format(self):
        assert normalize_response_format(None) == (None, None)

    def test_text_type_is_plain(self):
        assert normalize_response_format({"type": "text"}) == (None, None)

    def test_json_object(self):
        norm, err = normalize_response_format({"type": "json_object"})
        assert err is None
        assert norm == {"type": "json_object"}

    def test_json_schema_full(self):
        norm, err = normalize_response_format({
            "type": "json_schema",
            "json_schema": {"name": "Loc", "schema": _SCHEMA, "strict": True},
        })
        assert err is None
        assert norm == {
            "type": "json_schema",
            "json_schema": {"name": "Loc", "schema": _SCHEMA, "strict": True},
        }

    def test_json_schema_defaults_name(self):
        norm, err = normalize_response_format({
            "type": "json_schema",
            "json_schema": {"schema": _SCHEMA},
        })
        assert err is None
        assert norm["json_schema"]["name"] == "response"

    def test_json_schema_missing_schema_errors(self):
        norm, err = normalize_response_format({"type": "json_schema", "json_schema": {"name": "x"}})
        assert norm is None
        assert "schema" in err

    def test_non_dict_errors(self):
        norm, err = normalize_response_format("json")
        assert norm is None
        assert err

    def test_unknown_type_errors(self):
        norm, err = normalize_response_format({"type": "xml"})
        assert norm is None
        assert "xml" in err


class TestNormalizeResponsesTextFormat:
    def test_none(self):
        assert normalize_responses_text_format(None) == (None, None)

    def test_no_format_key(self):
        assert normalize_responses_text_format({}) == (None, None)

    def test_text_format(self):
        assert normalize_responses_text_format({"format": {"type": "text"}}) == (None, None)

    def test_json_object(self):
        norm, err = normalize_responses_text_format({"format": {"type": "json_object"}})
        assert err is None
        assert norm == {"type": "json_object"}

    def test_json_schema_converted(self):
        norm, err = normalize_responses_text_format({
            "format": {"type": "json_schema", "name": "Loc", "schema": _SCHEMA},
        })
        assert err is None
        assert norm == {
            "type": "json_schema",
            "json_schema": {"name": "Loc", "schema": _SCHEMA},
        }

    def test_json_schema_missing_schema_errors(self):
        norm, err = normalize_responses_text_format({"format": {"type": "json_schema", "name": "x"}})
        assert norm is None
        assert err


_JSON_SCHEMA = {"type": "json_schema", "json_schema": {"name": "Loc", "schema": _SCHEMA}}
_JSON_OBJECT = {"type": "json_object"}


class TestUnsupportedReason:
    def test_no_constraint_is_supported(self):
        assert unsupported_reason(None, "bedrock_converse") is None

    def test_unresolved_api_mode_defers(self):
        assert unsupported_reason(_JSON_SCHEMA, None) is None

    def test_chat_completions_allows_schema_and_object(self):
        assert unsupported_reason(_JSON_SCHEMA, "chat_completions") is None
        assert unsupported_reason(_JSON_OBJECT, "chat_completions") is None

    def test_anthropic_allows_json_schema(self):
        assert unsupported_reason(_JSON_SCHEMA, "anthropic_messages") is None

    def test_anthropic_rejects_json_object(self):
        reason = unsupported_reason(_JSON_OBJECT, "anthropic_messages")
        assert reason and "json_object" in reason and "anthropic_messages" in reason

    @pytest.mark.parametrize("mode", ["bedrock_converse", "codex_responses", "codex_app_server"])
    def test_unmapped_modes_rejected(self, mode):
        reason = unsupported_reason(_JSON_SCHEMA, mode)
        assert reason and mode in reason


class TestApply:
    def test_chat_completions_attaches_response_format(self):
        kwargs = {}
        apply(kwargs, _JSON_SCHEMA, "chat_completions")
        assert kwargs["response_format"] == _JSON_SCHEMA

    def test_chat_completions_does_not_override_existing(self):
        existing = {"type": "json_object"}
        kwargs = {"response_format": existing}
        apply(kwargs, _JSON_SCHEMA, "chat_completions")
        assert kwargs["response_format"] is existing

    def test_none_constraint_is_noop(self):
        kwargs = {}
        apply(kwargs, None, "chat_completions")
        assert "response_format" not in kwargs

    def test_anthropic_attaches_output_config_format(self):
        kwargs = {}
        apply(kwargs, _JSON_SCHEMA, "anthropic_messages")
        # name / strict have no Anthropic equivalent and are dropped.
        assert kwargs["output_config"] == {"format": {"type": "json_schema", "schema": _SCHEMA}}

    def test_anthropic_merges_with_existing_effort(self):
        kwargs = {"output_config": {"effort": "high"}}
        apply(kwargs, _JSON_SCHEMA, "anthropic_messages")
        assert kwargs["output_config"]["effort"] == "high"
        assert kwargs["output_config"]["format"] == {"type": "json_schema", "schema": _SCHEMA}

    def test_anthropic_json_object_attaches_nothing(self):
        # json_object has no Anthropic output_config expression; apply is a no-op
        # (the guard rejects it up-front, this is the defense-in-depth path).
        kwargs = {}
        apply(kwargs, _JSON_OBJECT, "anthropic_messages")
        assert "output_config" not in kwargs


def test_supported_api_modes_membership():
    assert "chat_completions" in SUPPORTED_API_MODES
    assert "anthropic_messages" in SUPPORTED_API_MODES
    assert "bedrock_converse" not in SUPPORTED_API_MODES


class TestBuildAnthropicKwargsWiring:
    """The runtime anthropic_messages path: build_anthropic_kwargs threads the
    constraint into output_config.format without clobbering effort."""

    _MSGS = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    def test_attaches_output_config_format(self):
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-6", messages=self._MSGS, tools=None,
            max_tokens=1024, reasoning_config=None, response_format=_JSON_SCHEMA,
        )
        assert kwargs["output_config"] == {"format": {"type": "json_schema", "schema": _SCHEMA}}

    def test_merges_with_adaptive_effort(self):
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-6", messages=self._MSGS, tools=None,
            max_tokens=1024, reasoning_config={"enabled": True, "effort": "high"},
            response_format=_JSON_SCHEMA,
        )
        assert kwargs["output_config"]["effort"] == "high"
        assert kwargs["output_config"]["format"] == {"type": "json_schema", "schema": _SCHEMA}

    def test_no_constraint_leaves_output_config_untouched(self):
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-6", messages=self._MSGS, tools=None,
            max_tokens=1024, reasoning_config={"enabled": True, "effort": "high"},
        )
        assert "format" not in kwargs.get("output_config", {})
