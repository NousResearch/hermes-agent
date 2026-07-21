"""Behavior contracts for Anthropic-native web search and fetch."""

from types import SimpleNamespace

from agent.anthropic_adapter import (
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
)
from agent.transports.anthropic import AnthropicTransport


def _tool(name: str, server_spec: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
            "_anthropic_server_tool": server_spec,
        },
    }


def test_native_endpoint_replaces_web_functions_with_server_tools():
    tools = [
        _tool("web_search", {
            "type": "web_search_20250305", "name": "web_search", "max_uses": 5,
        }),
        _tool("web_extract", {
            "type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 5,
        }),
    ]

    converted = convert_tools_to_anthropic(tools, base_url="https://api.anthropic.com")

    assert converted == [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
        {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 5},
    ]


def test_compatible_third_party_endpoint_keeps_local_function_tools():
    tools = [_tool("web_search", {
        "type": "web_search_20250305", "name": "web_search", "max_uses": 5,
    })]

    converted = convert_tools_to_anthropic(
        tools, base_url="https://api.minimax.io/anthropic"
    )

    assert converted[0]["name"] == "web_search"
    assert converted[0]["input_schema"]["type"] == "object"
    assert "type" not in converted[0]


def test_server_blocks_are_captured_and_replayed_in_original_order():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(
                type="server_tool_use",
                id="srvtoolu_1",
                name="web_search",
                input={"query": "Hermes Agent"},
            ),
            SimpleNamespace(
                type="web_search_tool_result",
                tool_use_id="srvtoolu_1",
                content=[{
                    "type": "web_search_result",
                    "title": "Hermes",
                    "url": "https://example.com/hermes",
                }],
            ),
            SimpleNamespace(
                type="text",
                text="Hermes is an agent.",
                citations=[{
                    "type": "web_search_result_location",
                    "title": "Hermes",
                    "url": "https://example.com/hermes",
                }],
            ),
        ],
    )

    normalized = AnthropicTransport().normalize_response(response)
    stored = {
        "role": "assistant",
        "content": normalized.content,
        "anthropic_content_blocks": normalized.anthropic_content_blocks,
    }
    _, replayed = convert_messages_to_anthropic([
        {"role": "user", "content": "Find Hermes"},
        stored,
    ])

    blocks = replayed[-1]["content"]
    assert [block["type"] for block in blocks] == [
        "server_tool_use", "web_search_tool_result", "text",
    ]
    assert blocks[-1]["citations"][0]["url"] == "https://example.com/hermes"
    assert normalized.content.endswith(
        "Sources:\n- Hermes: https://example.com/hermes"
    )


def test_web_fetch_blocks_use_the_same_round_trip_channel():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(
                type="server_tool_use",
                id="srvtoolu_fetch",
                name="web_fetch",
                input={"url": "https://example.com"},
            ),
            SimpleNamespace(
                type="web_fetch_tool_result",
                tool_use_id="srvtoolu_fetch",
                content={
                    "type": "web_fetch_result",
                    "url": "https://example.com",
                    "content": {"type": "document", "source": {"type": "text", "data": "ok"}},
                },
            ),
            SimpleNamespace(type="text", text="Fetched."),
        ],
    )

    normalized = AnthropicTransport().normalize_response(response)

    assert [block["type"] for block in normalized.anthropic_content_blocks] == [
        "server_tool_use", "web_fetch_tool_result", "text",
    ]


def test_pause_turn_remains_distinct_for_the_conversation_loop():
    transport = AnthropicTransport()
    response = SimpleNamespace(
        stop_reason="pause_turn",
        content=[SimpleNamespace(
            type="server_tool_use",
            id="srvtoolu_1",
            name="web_search",
            input={"query": "long research"},
        )],
    )

    normalized = transport.normalize_response(response)

    assert transport.map_finish_reason("pause_turn") == "pause_turn"
    assert normalized.finish_reason == "pause_turn"
    assert normalized.anthropic_content_blocks[0]["id"] == "srvtoolu_1"


def test_model_key_does_not_implicitly_replace_the_web_backend(monkeypatch):
    from tools import web_tools

    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
    monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
    monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
    monkeypatch.setattr(web_tools, "_list_registered_web_providers", lambda: [])
    keys = {"ANTHROPIC_API_KEY": "sk-ant-api-test"}
    monkeypatch.setattr(web_tools, "_has_env", lambda name: bool(keys.get(name)))

    assert web_tools._get_backend() != "anthropic"
    assert not web_tools.check_web_api_key()

    monkeypatch.setattr(
        web_tools, "_load_web_config", lambda: {"backend": "anthropic"}
    )
    assert web_tools._get_backend() == "anthropic"
    assert web_tools.check_web_api_key()


def test_dynamic_markers_follow_per_capability_selection(monkeypatch):
    from tools import web_tools

    monkeypatch.setattr(web_tools, "_get_search_backend", lambda: "anthropic")
    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "brave-free")

    search = web_tools._anthropic_web_search_schema_overrides()
    extract = web_tools._anthropic_web_fetch_schema_overrides()

    assert search["_anthropic_server_tool"]["name"] == "web_search"
    assert extract == {}


def test_tools_picker_exposes_anthropic_without_requesting_a_second_key():
    from hermes_cli.tools_config import TOOL_CATEGORIES

    providers = TOOL_CATEGORIES["web"]["providers"]
    anthropic = next(p for p in providers if p.get("web_backend") == "anthropic")

    assert anthropic["env_vars"] == []
    assert "already configured" in anthropic["tag"]
