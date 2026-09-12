"""Codex Responses tool-schema conversion contracts shared by direct and auxiliary paths."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from agent.auxiliary_client import _CodexCompletionsAdapter
from agent.transports.codex import ResponsesApiTransport


def _chat_tool(*, strict: bool | None = None, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": "search_records",
        "description": "Search records.",
        "parameters": parameters or {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }
    if strict is not None:
        function["strict"] = strict
    return {"type": "function", "function": function}


def _auxiliary_tools(tools: list[dict[str, Any]], *, base_url: str = "https://chatgpt.com/backend-api/codex") -> list[dict[str, Any]]:
    adapter = _CodexCompletionsAdapter(SimpleNamespace(base_url=base_url), "gpt-5.6-terra")
    response_kwargs, _, _ = adapter._build_responses_kwargs({
        "messages": [{"role": "user", "content": "find it"}],
        "tools": tools,
    })
    return response_kwargs["tools"]


@pytest.mark.parametrize("strict", [True, False, None])
def test_direct_and_auxiliary_codex_schema_conversion_match_strict_and_optional_contract(strict):
    """Both Responses callers retain intentional strictness and optional fields unchanged."""
    tools = [_chat_tool(strict=strict)]

    direct = ResponsesApiTransport().convert_tools(tools)
    auxiliary = _auxiliary_tools(tools)

    assert direct is not None
    assert direct == auxiliary
    assert direct == [{
        "type": "function", "name": "search_records", "description": "Search records.",
        "strict": strict if strict is not None else False,
        "parameters": tools[0]["function"]["parameters"],
    }]
    assert direct[0]["strict"] is (strict if strict is not None else False)
    assert direct[0]["parameters"]["required"] == ["query"]
    assert "limit" not in direct[0]["parameters"]["required"]


@pytest.mark.parametrize("strict", [True, False, None])
@pytest.mark.parametrize("base_url", ["https://api.x.ai/v1", "https://chatgpt.com/backend-api/codex"])
def test_auxiliary_codex_schema_sanitizes_provider_rejections_without_mutating_caller_schema(strict, base_url):
    """The xAI sanitizer remains active before the shared conversion and owns a private copy."""
    tools = [_chat_tool(
        strict=strict,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "pattern": "^[a-z]+$", "format": "email"},
            },
            "required": [],
        },
    )]
    original = copy.deepcopy(tools)

    converted = _auxiliary_tools(tools, base_url=base_url)

    assert converted[0]["strict"] is (strict if strict is not None else False)
    assert "pattern" not in converted[0]["parameters"]["properties"]["query"]
    assert "format" not in converted[0]["parameters"]["properties"]["query"]
    assert tools == original
