"""Scalar enum domains must survive the registered-tool Moonshot request path."""

import copy
import json
from itertools import combinations
from typing import Any, Literal

import pytest
from jsonschema import Draft202012Validator
from pydantic import create_model

from agent.moonshot_schema import sanitize_moonshot_tool_parameters
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile
from tools.registry import ToolRegistry
from tools.schema_sanitizer import sanitize_tool_schemas


_PROBES = [None, "", False, True, -1, 0, 1, 1.0, 1.5, 2, 2.5, "auto", "manual", [], {}]
_SCALAR_MIXTURES = [values for size in range(2, 5) for values in combinations((False, 0, 2.5, "auto"), size)]


def _request_schema(schema, provider):
    registry = ToolRegistry()
    registry.register(
        name="set_value", toolset="enum_test",
        schema={"description": "Choose a setting.", "parameters": schema},
        handler=lambda args, **kwargs: json.dumps(args),
    )
    tools = sanitize_tool_schemas(registry.get_definitions({"set_value"}, quiet=True))
    before = copy.deepcopy(tools)
    kwargs = {}
    if provider:
        kwargs["provider_profile"] = get_provider_profile(provider)
        assert kwargs["provider_profile"] is not None
    request = ChatCompletionsTransport().build_kwargs(
        "moonshotai/kimi-k2.6", [{"role": "user", "content": "Choose a setting."}], tools, **kwargs
    )
    assert tools == before
    return request["tools"][0]["function"]["parameters"]


def _placements(field):
    def obj(value, **extra):
        return {"type": "object", "properties": {"value": value}, "required": ["value"], **extra}

    return [
        (obj(field), lambda value: {"value": value}),
        (obj({"type": "array", "items": field}), lambda value: {"value": [value]}),
        (obj({"$ref": "#/$defs/Setting"}, **{"$defs": {"Setting": field}}), lambda value: {"value": value}),
        (obj({"anyOf": [field, {"type": "null"}], "nullable": True}), lambda value: {"value": value}),
        (obj({"anyOf": [{"anyOf": [field, {"type": "null"}]}, {"type": "null"}]}),
         lambda value: {"value": value}),
    ]


def _assert_union_shape(node):
    assert "nullable" not in node
    if "anyOf" in node:
        assert "type" not in node
        assert "enum" not in node
        for branch in node["anyOf"]:
            _assert_union_shape(branch)
    for key in ("properties", "$defs"):
        for child in node.get(key, {}).values():
            _assert_union_shape(child)
    if isinstance(node.get("items"), dict):
        _assert_union_shape(node["items"])


@pytest.mark.parametrize("provider", [None, "openrouter"], ids=["legacy-route", "profile-route"])
@pytest.mark.parametrize("values", [
    *_SCALAR_MIXTURES, (1, 2.5), (True, 1),
    (1, 2), (1.0, 2.5), (False, True), ("auto", "manual"),
    (None, "", 0, 2.5), (None, "", False, "auto"), ("", 1), (None, True),
    (9007199254740993, 0.5),
])
def test_registered_scalar_enums_keep_members_and_reject_nonmembers(values, provider):
    for ordered in (values, tuple(reversed(values))):
        literal = Literal.__getitem__(ordered)
        model = create_model("Setting", value=(literal, ...))
        produced = model.model_json_schema()
        for value in ordered:
            assert model(value=value).model_dump()["value"] == value
        for source, instance in _placements(produced["properties"]["value"]):
            Draft202012Validator.check_schema(source)
            before = copy.deepcopy(source)
            normalized = sanitize_moonshot_tool_parameters(source)
            wire = _request_schema(source, provider)
            assert source == before
            assert sanitize_moonshot_tool_parameters(normalized) == normalized
            _assert_union_shape(normalized)
            _assert_union_shape(wire)
            for candidate in [*ordered, *_PROBES]:
                allowed = candidate is not None and candidate != ""
                expected = allowed and Draft202012Validator(source).is_valid(instance(candidate))
                assert Draft202012Validator(normalized).is_valid(instance(candidate)) == expected, (ordered, source, candidate)
                assert Draft202012Validator(wire).is_valid(instance(candidate)) == expected, (ordered, source, candidate)


@pytest.mark.parametrize("field, definitions, expected_type", [
    pytest.param({"type": "integer", "enum": [1, 2.5]}, {}, "integer", id="explicit-type"),
    pytest.param({"type": ["string", "null"], "enum": ["auto", None, ""]}, {}, "string", id="type-array"),
    pytest.param({"type": ["number", "string"], "enum": [1, "auto"]}, {}, "number", id="existing-multi-type-policy"),
    pytest.param({"$ref": "#/$defs/Number", "enum": [1, 2.5]}, {"Number": {"type": "number"}}, None, id="ref"),
    pytest.param({"properties": {}, "enum": [{}, {"x": 1}]}, {}, "object", id="object-hint"),
    pytest.param({"items": {"type": "integer"}, "enum": [[1], [2]]}, {}, "array", id="array-hint"),
    pytest.param({"required": [], "enum": [1, 2.5]}, {}, "object", id="required-hint"),
    pytest.param({"enum": [1, {"type": "number", "enum": [1, 2.5]}]}, {}, "integer", id="structured-fallback"),
    pytest.param({"enum": [{"type": "number"}, 1]}, {}, "string", id="structured-first-fallback"),
    pytest.param({"enum": [None, ""]}, {}, "string", id="empty-members"),
    pytest.param({"enum": []}, {}, "string", id="empty-enum"),
    pytest.param({"enum": ["auto", 1, 2.5], "minimum": 2, "title": "Setting", "description": "Choose.",
                  "default": "auto"}, {}, None, id="sibling-constraints"),
    pytest.param({"enum": ["auto", 1, 2.5], "const": "auto",
                  "examples": [{"type": "number", "enum": [1, 2.5]}]}, {}, None, id="literal-metadata"),
    pytest.param({"enum": ["auto", 1], "oneOf": [{"type": "string"}, {"type": "integer"}]}, {}, None,
                 id="existing-composition"),
    pytest.param({"anyOf": [{"enum": ["auto", 1]}, {"enum": [False, 2.5]}]}, {}, None, id="non-null-union"),
    pytest.param({"enum": ["auto", 1, "", None], "anyOf": [
        {"anyOf": [{"type": "string"}, {"type": "integer"}]}, {"type": "null"},
    ]}, {}, "string", id="parent-enum-composition-policy"),
])
def test_enum_inference_keeps_explicit_constraints_and_compatibility_rules(field, definitions, expected_type):
    source: dict[str, Any] = {"type": "object", "properties": {"value": field}, "required": ["value"]}
    if definitions:
        source["$defs"] = definitions
    before = copy.deepcopy(source)
    out = sanitize_moonshot_tool_parameters(source)
    assert source == before
    # Parent enum+anyOf promotion already changes again on the second pass; preserve its first-pass policy.
    if not ("enum" in field and "anyOf" in field):
        assert sanitize_moonshot_tool_parameters(out) == out
    value_schema = out["properties"]["value"]
    if expected_type is None:
        assert "type" not in value_schema
    else:
        assert value_schema["type"] == expected_type
    for key in ("title", "description", "default", "examples", "const", "minimum", "oneOf"):
        if key in field:
            assert value_schema[key] == field[key]
    if field.get("enum") == [] or field.get("enum") == [None, ""]:
        assert value_schema == {"type": "string"}
    elif expected_type is None or not isinstance(field.get("type"), list):
        expected = copy.deepcopy(source)
        if expected_type is not None:
            expected["properties"]["value"]["type"] = expected_type
            if expected_type in {"string", "integer", "number", "boolean"}:
                expected["properties"]["value"]["enum"] = [v for v in field["enum"] if v is not None and v != ""]
        for candidate in _PROBES:
            assert Draft202012Validator(out).is_valid({"value": candidate}) == (
                Draft202012Validator(expected).is_valid({"value": candidate})
            )
    if expected_type is not None and field.get("enum"):
        expected_enum = field["enum"]
        if expected_type in {"string", "integer", "number", "boolean"}:
            expected_enum = [v for v in expected_enum if v is not None and v != ""]
        if expected_enum:
            assert value_schema["enum"] == expected_enum
    if "anyOf" in value_schema and expected_type is None:
        _assert_union_shape(value_schema)
