"""Tests for provider-executed (server-side) Anthropic tool injection.

``providers.<name>.server_tools: ["web_search"]`` swaps the client-side
``web_search`` function for Anthropic's native server-tool declaration
(``web_search_20250305``) — the tool several Anthropic-compatible endpoints
(Zhipu, Kimi, DeepSeek, MiniMax) execute themselves.  These tests pin the
three contracts of that swap:

1. Injection is a 1:1 swap, never additive, and absent by default.
2. The config knob flows through provider normalization to the agent attr.
3. A server-tool turn that lands in session history (``server_tool_use`` /
   ``web_search_tool_result`` blocks) replays as schema-valid input instead
   of being silently dropped or rejected with HTTP 400.
"""
from types import SimpleNamespace

from agent.agent_init import (
    _custom_provider_server_tools_for_agent,
    _merge_custom_provider_server_tools,
)
from agent.anthropic_adapter import (
    _apply_server_tools_to_anthropic,
    _convert_assistant_message,
    _convert_content_part_to_anthropic,
    _sanitize_replay_block,
    build_anthropic_kwargs,
)

BASE = "https://open.bigmodel.cn/api/anthropic"


def _tools(*names):
    return [
        {"type": "function", "function": {"name": n, "description": n, "parameters": {}}}
        for n in names
    ]


def _atools(*names):
    """Anthropic-format tool dicts, as convert_tools_to_anthropic emits."""
    return [
        {"name": n, "description": n, "input_schema": {"type": "object", "properties": {}}}
        for n in names
    ]


def _build(tools, **over):
    kwargs = dict(
        model="glm-4.7",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        max_tokens=64,
        reasoning_config=None,
        base_url=BASE,
    )
    kwargs.update(over)
    return build_anthropic_kwargs(**kwargs)


class TestInjection:
    def test_off_by_default(self):
        kwargs = _build(_tools("web_search", "terminal"))
        # Behavior contract: without the knob every tool stays client-side
        # and no server-tool declaration appears.
        names = [t["name"] for t in kwargs["tools"]]
        assert names == ["web_search", "terminal"]

    def test_web_search_swapped_for_server_tool(self):
        kwargs = _build(_tools("web_search", "terminal"), server_tools=["web_search"])
        by_name = {t.get("name"): t for t in kwargs["tools"]}
        # exactly one web_search entry, and it is the server declaration
        assert by_name["web_search"] == {"type": "web_search_20250305", "name": "web_search"}
        # untouched sibling survives
        assert "terminal" in by_name
        assert len(kwargs["tools"]) == 2

    def test_swap_never_additive_without_client_tool(self):
        out = _apply_server_tools_to_anthropic(_atools("terminal"), ["web_search"])
        names = [t.get("name") for t in out]
        # no client web_search was present, so none is granted
        assert names == ["terminal"]

    def test_unknown_name_ignored(self):
        out = _apply_server_tools_to_anthropic(_atools("web_search", "terminal"), ["not_a_server_tool"])
        assert [t["name"] for t in out] == ["web_search", "terminal"]

    def test_no_tools_no_injection(self):
        kwargs = _build(None, server_tools=["web_search"])
        assert "tools" not in kwargs


class TestConfigPlumbing:
    def test_resolved_by_provider_key(self):
        got = _custom_provider_server_tools_for_agent(
            provider="custom:zhipu",
            base_url=BASE + "/",
            custom_providers=[{
                "provider_key": "zhipu",
                "name": "Zhipu Anthropic",
                "base_url": BASE,
                "server_tools": ["web_search"],
            }],
        )
        assert got == ["web_search"]

    def test_none_when_unset(self):
        got = _custom_provider_server_tools_for_agent(
            provider="custom",
            base_url=BASE,
            custom_providers=[{"name": "zhipu", "base_url": BASE}],
        )
        assert got is None

    def test_non_custom_provider_unaffected(self):
        got = _custom_provider_server_tools_for_agent(
            provider="openrouter",
            base_url=BASE,
            custom_providers=[{"name": "zhipu", "base_url": BASE,
                               "server_tools": ["web_search"]}],
        )
        assert got is None

    def test_merge_sets_agent_attr(self):
        agent = SimpleNamespace(provider="custom", base_url=BASE)
        _merge_custom_provider_server_tools(
            agent, [{"name": "zhipu", "base_url": BASE, "server_tools": ["web_search"]}],
        )
        assert agent._anthropic_server_tools == ["web_search"]

    def test_merge_noop_without_knob(self):
        agent = SimpleNamespace(provider="custom", base_url=BASE)
        _merge_custom_provider_server_tools(agent, [{"name": "zhipu", "base_url": BASE}])
        assert not hasattr(agent, "_anthropic_server_tools")

    def test_provider_entry_normalization_keeps_server_tools(self):
        from hermes_cli.config import _normalize_custom_provider_entry

        normalized = _normalize_custom_provider_entry({
            "name": "zhipu",
            "base_url": BASE,
            "api_mode": "anthropic",
            "server_tools": ["web_search"],
        })
        assert normalized["server_tools"] == ["web_search"]
        assert normalized["api_mode"] == "anthropic_messages"


class TestReplaySanitize:
    def test_server_tool_use_becomes_text_marker(self):
        out = _sanitize_replay_block({
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "hermes"},
        })
        assert out == {"type": "text", "text": "[server tool: web_search]"}

    def test_web_search_tool_result_becomes_text(self):
        out = _sanitize_replay_block({
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [
                {"type": "web_search_result", "url": "https://example.com", "title": "Example"},
                {"type": "text", "text": "Example — the example domain."},
                {"type": "text", "text": "  "},
            ],
        })
        # non-text result rows and blank text rows are dropped; text survives
        assert out == {"type": "text", "text": "Example — the example domain."}

    def test_bare_blocks_dropped(self):
        assert _sanitize_replay_block({"type": "server_tool_use", "id": "x"}) is None
        assert _sanitize_replay_block({"type": "web_search_tool_result"}) is None

    def test_content_part_path_also_converted(self):
        out = _convert_content_part_to_anthropic({
            "type": "server_tool_use", "id": "srvtoolu_2", "name": "web_search",
            "input": {"query": "q"},
        })
        assert out == {"type": "text", "text": "[server tool: web_search]"}

    def test_stored_search_turn_replays_as_valid_assistant_content(self):
        # End-to-end replay: a captured turn mixing text + server-tool blocks
        # must come back as text/tool_use blocks the Messages input accepts.
        m = {
            "role": "assistant",
            "anthropic_content_blocks": [
                {"type": "text", "text": "Searching…"},
                {"type": "server_tool_use", "id": "srvtoolu_1",
                 "name": "web_search", "input": {"query": "hermes"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
                 "content": [{"type": "text", "text": "result body"}]},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                 "input": {"path": "a"}},
            ],
        }
        out = _convert_assistant_message(m)
        types = [b["type"] for b in out["content"]]
        assert types == ["text", "text", "text", "tool_use"]
        assert all(b["type"] != "server_tool_use" for b in out["content"])
        assert "result body" in out["content"][2]["text"]
