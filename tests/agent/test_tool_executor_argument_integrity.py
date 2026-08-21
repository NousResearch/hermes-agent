"""Fail-closed execution tests for compressed historical tool arguments."""

from __future__ import annotations

import json

import pytest

from agent.tool_argument_integrity import (
    incomplete_tool_arguments_after_schema_decode,
    incomplete_tool_arguments_error_result,
)
from agent.tool_executor import _parse_tool_arguments
from tools.tool_search import resolve_underlying_call


RESERVED = "__hermes_incomplete_tool_arguments__"


@pytest.mark.parametrize(
    "shape",
    [
        {"command": "python -c 'print(1)'"},
        {"path": "/tmp/read-only.txt", "offset": 1, "limit": 100},
        {"path": "/tmp/a", "content": "payload"},
        {"mode": "replace", "path": "/tmp/a", "old_string": "a", "new_string": "b"},
    ],
)
def test_incomplete_provenance_blocks_operation_shapes(shape):
    shape[RESERVED] = {
        "version": 1,
        "reason": "context_compression",
        "replayable": False,
    }
    arguments, error = _parse_tool_arguments(json.dumps(shape))

    assert arguments == {}
    parsed_error = json.loads(error or "{}")
    assert parsed_error["error_type"] == "incomplete_historical_tool_arguments"
    assert "not executed" in parsed_error["message"]


def test_nested_or_malformed_reserved_provenance_fails_closed():
    for payload in (
        {"outer": [{RESERVED: {"version": 1}}]},
        {RESERVED: "malformed"},
        {RESERVED: None},
    ):
        arguments, error = _parse_tool_arguments(json.dumps(payload))
        assert arguments == {}
        assert json.loads(error or "{}")["error_type"] == "incomplete_historical_tool_arguments"


def test_literal_truncation_marker_is_not_treated_as_provenance():
    payload = {
        "path": "/tmp/a",
        "content": "Documentation containing ...[truncated] literally.",
    }
    arguments, error = _parse_tool_arguments(json.dumps(payload))

    assert error is None
    assert arguments == payload


def test_string_encoded_deferred_arguments_are_blocked_after_unwrap():
    from tools.registry import registry

    deferred_name = "mcp__integrity_probe__run"
    registry.register(
        name=deferred_name,
        toolset="mcp-integrity-probe",
        schema={
            "name": deferred_name,
            "description": "Integrity probe",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kwargs: "{}",
    )
    outer = {
        "name": deferred_name,
        "arguments": json.dumps(
            {
                RESERVED: {
                    "version": 1,
                    "reason": "context_compression",
                    "replayable": False,
                }
            }
        ),
    }
    parsed, error = _parse_tool_arguments(json.dumps(outer))
    assert error is None

    _name, underlying, resolve_error = resolve_underlying_call(parsed)
    assert resolve_error is None
    result = incomplete_tool_arguments_error_result(underlying)
    assert result is not None
    parsed_result = json.loads(result)
    assert parsed_result["error_type"] == "incomplete_historical_tool_arguments"
    assert "not executed" in parsed_result["message"]


def test_schema_coercion_cannot_materialize_incomplete_provenance_before_dispatch(monkeypatch):
    import model_tools
    from tools.registry import registry

    tool_name = "integrity_schema_coercion_probe"
    registry.register(
        name=tool_name,
        toolset="integrity-probe",
        schema={
            "name": tool_name,
            "description": "Integrity coercion probe",
            "parameters": {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            },
        },
        handler=lambda args, **kwargs: "should not run",
    )
    dispatches = []
    monkeypatch.setattr(
        registry,
        "dispatch",
        lambda name, args, **kwargs: dispatches.append((name, args)) or "dispatched",
    )
    encoded = json.dumps([
        {
            RESERVED: {
                "version": 1,
                "reason": "context_compression",
                "replayable": False,
            }
        }
    ])

    result = model_tools.handle_function_call(
        tool_name,
        {"items": encoded},
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )

    assert dispatches == []
    parsed = json.loads(result)
    assert parsed["error_type"] == "incomplete_historical_tool_arguments"
    assert "not executed" in parsed["message"]


def test_root_union_without_direct_type_is_not_preview_decoded():
    encoded = json.dumps([{RESERVED: {"version": 1, "replayable": False}}])
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "anyOf": [
                    {"type": "array", "items": {"type": "object"}},
                    {"type": "string"},
                ]
            }
        },
    }

    assert incomplete_tool_arguments_after_schema_decode(
        {"items": encoded}, schema
    ) is None


def test_root_type_list_is_preview_decoded_like_runtime():
    encoded = json.dumps([{RESERVED: {"version": 1, "replayable": False}}])
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": ["array", "string"],
                "items": {"type": "object"},
            }
        },
    }

    result = incomplete_tool_arguments_after_schema_decode(
        {"items": encoded}, schema
    )
    assert json.loads(result or "{}")["error_type"] == "incomplete_historical_tool_arguments"


def test_native_type_list_container_is_not_recursively_preview_decoded():
    encoded_item = json.dumps({RESERVED: {"version": 1, "replayable": False}})
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": ["array", "string"],
                "items": {"type": "object"},
            }
        },
    }

    assert incomplete_tool_arguments_after_schema_decode(
        {"items": [encoded_item]}, schema
    ) is None


def test_nested_union_items_remain_preview_decoded():
    encoded = json.dumps({RESERVED: {"version": 1, "replayable": False}})
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "oneOf": [{"type": "object"}, {"type": "string"}]
                },
            }
        },
    }

    result = incomplete_tool_arguments_after_schema_decode(
        {"items": [encoded]}, schema
    )
    assert json.loads(result or "{}")["error_type"] == "incomplete_historical_tool_arguments"