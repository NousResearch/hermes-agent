"""Provider parity tests: verify that AIAgent builds correct API kwargs
and handles responses properly for all supported providers.

Ensures changes to one provider path don't silently break another.
"""

import json
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from run_agent import AIAgent


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")
    def close(self):
        pass


def _make_agent(monkeypatch, provider, api_mode="chat_completions", base_url="https://openrouter.ai/api/v1"):
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: _tool_defs("web_search", "terminal"))
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    return AIAgent(
        api_key="test-key",
        base_url=base_url,
        provider=provider,
        api_mode=api_mode,
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


# ── _build_api_kwargs tests ─────────────────────────────────────────────────

class TestBuildApiKwargsOpenRouter:
    def test_uses_chat_completions_format(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "messages" in kwargs
        assert "model" in kwargs
        assert kwargs["messages"][-1]["content"] == "hi"

    def test_includes_reasoning_in_extra_body(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        extra = kwargs.get("extra_body", {})
        assert "reasoning" in extra
        assert extra["reasoning"]["enabled"] is True

    def test_includes_tools(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "tools" in kwargs
        tool_names = [t["function"]["name"] for t in kwargs["tools"]]
        assert "web_search" in tool_names

    def test_no_responses_api_fields(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "input" not in kwargs
        assert "instructions" not in kwargs
        assert "store" not in kwargs


class TestBuildApiKwargsNousPortal:
    def test_includes_nous_product_tags(self, monkeypatch):
        agent = _make_agent(monkeypatch, "nous", base_url="https://inference-api.nousresearch.com/v1")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        extra = kwargs.get("extra_body", {})
        assert extra.get("tags") == ["product=hermes-agent"]

    def test_uses_chat_completions_format(self, monkeypatch):
        agent = _make_agent(monkeypatch, "nous", base_url="https://inference-api.nousresearch.com/v1")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "messages" in kwargs
        assert "input" not in kwargs


class TestBuildApiKwargsCustomEndpoint:
    def test_uses_chat_completions_format(self, monkeypatch):
        agent = _make_agent(monkeypatch, "custom", base_url="http://localhost:1234/v1")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "messages" in kwargs
        assert "input" not in kwargs

    def test_no_openrouter_extra_body(self, monkeypatch):
        agent = _make_agent(monkeypatch, "custom", base_url="http://localhost:1234/v1")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        extra = kwargs.get("extra_body", {})
        assert "reasoning" not in extra


class TestBuildApiKwargsCodex:
    def test_uses_responses_api_format(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "input" in kwargs
        assert "instructions" in kwargs
        assert "messages" not in kwargs
        assert kwargs["store"] is False

    def test_includes_reasoning_config(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "reasoning" in kwargs
        assert kwargs["reasoning"]["effort"] == "xhigh"

    def test_includes_encrypted_content_in_include(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "reasoning.encrypted_content" in kwargs.get("include", [])

    def test_tools_converted_to_responses_format(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        tools = kwargs.get("tools", [])
        assert len(tools) > 0
        # Responses format has "name" at top level, not nested under "function"
        assert "name" in tools[0]
        assert "function" not in tools[0]


# ── Message conversion tests ────────────────────────────────────────────────

class TestChatMessagesToResponsesInput:
    """Verify _chat_messages_to_responses_input for Codex mode."""

    def test_user_message_passes_through(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [{"role": "user", "content": "hello"}]
        items = agent._chat_messages_to_responses_input(messages)
        assert items == [{"role": "user", "content": "hello"}]

    def test_system_messages_filtered(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
        ]
        items = agent._chat_messages_to_responses_input(messages)
        assert len(items) == 1
        assert items[0]["role"] == "user"

    def test_assistant_tool_calls_become_function_call_items(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_abc",
                "call_id": "call_abc",
                "function": {"name": "web_search", "arguments": '{"query": "test"}'},
            }],
        }]
        items = agent._chat_messages_to_responses_input(messages)
        fc_items = [i for i in items if i.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["name"] == "web_search"
        assert fc_items[0]["call_id"] == "call_abc"

    def test_tool_results_become_function_call_output(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [{"role": "tool", "tool_call_id": "call_abc", "content": "result here"}]
        items = agent._chat_messages_to_responses_input(messages)
        assert items[0]["type"] == "function_call_output"
        assert items[0]["call_id"] == "call_abc"
        assert items[0]["output"] == "result here"

    def test_encrypted_reasoning_replayed(self, monkeypatch):
        """Encrypted reasoning items from previous turns must be included in input."""
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [
            {"role": "user", "content": "think about this"},
            {
                "role": "assistant",
                "content": "I thought about it.",
                "codex_reasoning_items": [
                    {"type": "reasoning", "id": "rs_abc", "encrypted_content": "gAAAA_test_blob"},
                ],
            },
            {"role": "user", "content": "continue"},
        ]
        items = agent._chat_messages_to_responses_input(messages)
        reasoning_items = [i for i in items if i.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["encrypted_content"] == "gAAAA_test_blob"

    def test_no_reasoning_items_for_non_codex_messages(self, monkeypatch):
        """Messages without codex_reasoning_items should not inject anything."""
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "hello"},
        ]
        items = agent._chat_messages_to_responses_input(messages)
        reasoning_items = [i for i in items if i.get("type") == "reasoning"]
        assert len(reasoning_items) == 0


# ── Response normalization tests ─────────────────────────────────────────────

class TestNormalizeCodexResponse:
    """Verify _normalize_codex_response extracts all fields correctly."""

    def _make_codex_agent(self, monkeypatch):
        return _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                           base_url="https://chatgpt.com/backend-api/codex")

    def test_text_response(self, monkeypatch):
        agent = self._make_codex_agent(monkeypatch)
        response = SimpleNamespace(
            output=[
                SimpleNamespace(type="message", status="completed",
                    content=[SimpleNamespace(type="output_text", text="Hello!")],
                    phase="final_answer"),
            ],
            status="completed",
        )
        msg, reason = agent._normalize_codex_response(response)
        assert msg.content == "Hello!"
        assert reason == "stop"

    def test_reasoning_summary_extracted(self, monkeypatch):
        agent = self._make_codex_agent(monkeypatch)
        response = SimpleNamespace(
            output=[
                SimpleNamespace(type="reasoning",
                    encrypted_content="gAAAA_blob",
                    summary=[SimpleNamespace(type="summary_text", text="Thinking about math")],
                    id="rs_123", status=None),
                SimpleNamespace(type="message", status="completed",
                    content=[SimpleNamespace(type="output_text", text="42")],
                    phase="final_answer"),
            ],
            status="completed",
        )
        msg, reason = agent._normalize_codex_response(response)
        assert msg.content == "42"
        assert "math" in msg.reasoning
        assert reason == "stop"

    def test_encrypted_content_captured(self, monkeypatch):
        agent = self._make_codex_agent(monkeypatch)
        response = SimpleNamespace(
            output=[
                SimpleNamespace(type="reasoning",
                    encrypted_content="gAAAA_secret_blob_123",
                    summary=[SimpleNamespace(type="summary_text", text="Thinking")],
                    id="rs_456", status=None),
                SimpleNamespace(type="message", status="completed",
                    content=[SimpleNamespace(type="output_text", text="done")],
                    phase="final_answer"),
            ],
            status="completed",
        )
        msg, reason = agent._normalize_codex_response(response)
        assert msg.codex_reasoning_items is not None
        assert len(msg.codex_reasoning_items) == 1
        assert msg.codex_reasoning_items[0]["encrypted_content"] == "gAAAA_secret_blob_123"
        assert msg.codex_reasoning_items[0]["id"] == "rs_456"

    def test_no_encrypted_content_when_missing(self, monkeypatch):
        agent = self._make_codex_agent(monkeypatch)
        response = SimpleNamespace(
            output=[
                SimpleNamespace(type="message", status="completed",
                    content=[SimpleNamespace(type="output_text", text="no reasoning")],
                    phase="final_answer"),
            ],
            status="completed",
        )
        msg, reason = agent._normalize_codex_response(response)
        assert msg.codex_reasoning_items is None

    def test_tool_calls_extracted(self, monkeypatch):
        agent = self._make_codex_agent(monkeypatch)
        response = SimpleNamespace(
            output=[
                SimpleNamespace(type="function_call", status="completed",
                    call_id="call_xyz", name="web_search",
                    arguments='{"query":"test"}', id="fc_xyz"),
            ],
            status="completed",
        )
        msg, reason = agent._normalize_codex_response(response)
        assert reason == "tool_calls"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "web_search"


# ── Chat completions response handling (OpenRouter/Nous) ─────────────────────

class TestBuildAssistantMessage:
    """Verify _build_assistant_message works for all provider response formats."""

    def test_openrouter_reasoning_fields(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        msg = SimpleNamespace(
            content="answer",
            tool_calls=None,
            reasoning="I thought about it",
            reasoning_content=None,
            reasoning_details=None,
        )
        result = agent._build_assistant_message(msg, "stop")
        assert result["content"] == "answer"
        assert result["reasoning"] == "I thought about it"
        assert "codex_reasoning_items" not in result

    def test_openrouter_reasoning_details_preserved_unmodified(self, monkeypatch):
        """reasoning_details must be passed back exactly as received for
        multi-turn continuity (OpenRouter, Anthropic, OpenAI all need this)."""
        agent = _make_agent(monkeypatch, "openrouter")
        original_detail = {
            "type": "thinking",
            "thinking": "deep thoughts here",
            "signature": "sig123_opaque_blob",
            "encrypted_content": "some_provider_blob",
            "extra_field": "should_not_be_dropped",
        }
        msg = SimpleNamespace(
            content="answer",
            tool_calls=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=[original_detail],
        )
        result = agent._build_assistant_message(msg, "stop")
        stored = result["reasoning_details"][0]
        # ALL fields must survive, not just type/text/signature
        assert stored["signature"] == "sig123_opaque_blob"
        assert stored["encrypted_content"] == "some_provider_blob"
        assert stored["extra_field"] == "should_not_be_dropped"
        assert stored["thinking"] == "deep thoughts here"

    def test_codex_preserves_encrypted_reasoning(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        msg = SimpleNamespace(
            content="result",
            tool_calls=None,
            reasoning="summary text",
            reasoning_content=None,
            reasoning_details=None,
            codex_reasoning_items=[
                {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA_blob"},
            ],
        )
        result = agent._build_assistant_message(msg, "stop")
        assert result["codex_reasoning_items"] == [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA_blob"},
        ]

    def test_plain_message_no_codex_items(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        msg = SimpleNamespace(
            content="simple",
            tool_calls=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        result = agent._build_assistant_message(msg, "stop")
        assert "codex_reasoning_items" not in result


# ── Auxiliary client provider resolution ─────────────────────────────────────

class TestAuxiliaryClientProviderPriority:
    """Verify auxiliary client resolution doesn't break for any provider."""

    def test_openrouter_always_wins(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        from agent.auxiliary_client import get_text_auxiliary_client
        with patch("agent.auxiliary_client.OpenAI") as mock:
            client, model = get_text_auxiliary_client()
        assert model == "google/gemini-3-flash-preview"
        assert "openrouter" in str(mock.call_args.kwargs["base_url"]).lower()

    def test_nous_when_no_openrouter(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from agent.auxiliary_client import get_text_auxiliary_client
        with patch("agent.auxiliary_client._read_nous_auth", return_value={"access_token": "nous-tok"}), \
             patch("agent.auxiliary_client.OpenAI") as mock:
            client, model = get_text_auxiliary_client()
        assert model == "gemini-3-flash"

    def test_custom_endpoint_when_no_nous(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "local-key")
        from agent.auxiliary_client import get_text_auxiliary_client
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client.OpenAI") as mock:
            client, model = get_text_auxiliary_client()
        assert mock.call_args.kwargs["base_url"] == "http://localhost:1234/v1"

    def test_codex_fallback_last_resort(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from agent.auxiliary_client import get_text_auxiliary_client, CodexAuxiliaryClient
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value="codex-tok"), \
             patch("agent.auxiliary_client.OpenAI"):
            client, model = get_text_auxiliary_client()
        assert model == "gpt-5.3-codex"
        assert isinstance(client, CodexAuxiliaryClient)


# ── Provider routing tests ───────────────────────────────────────────────────

class TestProviderRouting:
    """Verify provider_routing config flows into extra_body.provider."""

    def test_sort_throughput(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        agent.provider_sort = "throughput"
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["provider"]["sort"] == "throughput"

    def test_only_providers(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        agent.providers_allowed = ["anthropic", "google"]
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["provider"]["only"] == ["anthropic", "google"]

    def test_ignore_providers(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        agent.providers_ignored = ["deepinfra"]
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["provider"]["ignore"] == ["deepinfra"]

    def test_order_providers(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        agent.providers_order = ["anthropic", "together"]
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["provider"]["order"] == ["anthropic", "together"]

    def test_require_parameters(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        agent.provider_require_parameters = True
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["provider"]["require_parameters"] is True

    def test_data_collection_deny(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        agent.provider_data_collection = "deny"
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["provider"]["data_collection"] == "deny"

    def test_no_routing_when_unset(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "provider" not in kwargs.get("extra_body", {}).get("provider", {}) or \
               kwargs.get("extra_body", {}).get("provider") is None or \
               "only" not in kwargs.get("extra_body", {}).get("provider", {})

    def test_combined_routing(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        agent.provider_sort = "latency"
        agent.providers_ignored = ["deepinfra"]
        agent.provider_data_collection = "deny"
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        prov = kwargs["extra_body"]["provider"]
        assert prov["sort"] == "latency"
        assert prov["ignore"] == ["deepinfra"]
        assert prov["data_collection"] == "deny"

    def test_routing_not_injected_for_codex(self, monkeypatch):
        """Codex Responses API doesn't use extra_body.provider."""
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        agent.provider_sort = "throughput"
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "extra_body" not in kwargs
        assert "provider" not in kwargs or kwargs.get("provider") is None


# ── Codex reasoning items preflight tests ────────────────────────────────────

class TestCodexReasoningPreflight:
    """Verify reasoning items pass through preflight normalization."""

    def test_reasoning_item_passes_through(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        raw_input = [
            {"role": "user", "content": "hello"},
            {"type": "reasoning", "encrypted_content": "abc123encrypted", "id": "r_001"},
            {"role": "assistant", "content": "hi there"},
        ]
        normalized = agent._preflight_codex_input_items(raw_input)
        reasoning_items = [i for i in normalized if i.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["encrypted_content"] == "abc123encrypted"
        assert reasoning_items[0]["id"] == "r_001"

    def test_reasoning_item_without_id(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        raw_input = [
            {"type": "reasoning", "encrypted_content": "abc123"},
        ]
        normalized = agent._preflight_codex_input_items(raw_input)
        assert len(normalized) == 1
        assert "id" not in normalized[0]

    def test_reasoning_item_empty_encrypted_skipped(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        raw_input = [
            {"type": "reasoning", "encrypted_content": ""},
            {"role": "user", "content": "hello"},
        ]
        normalized = agent._preflight_codex_input_items(raw_input)
        reasoning_items = [i for i in normalized if i.get("type") == "reasoning"]
        assert len(reasoning_items) == 0

    def test_reasoning_items_replayed_from_history(self, monkeypatch):
        """Reasoning items stored in codex_reasoning_items get replayed."""
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "hi",
                "codex_reasoning_items": [
                    {"type": "reasoning", "encrypted_content": "enc123", "id": "r_1"},
                ],
            },
            {"role": "user", "content": "follow up"},
        ]
        items = agent._chat_messages_to_responses_input(messages)
        reasoning_items = [i for i in items if isinstance(i, dict) and i.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["encrypted_content"] == "enc123"


# ── Reasoning effort consistency tests ───────────────────────────────────────

class TestReasoningEffortDefaults:
    """Verify reasoning effort defaults to xhigh across all provider paths."""

    def test_openrouter_default_xhigh(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        reasoning = kwargs["extra_body"]["reasoning"]
        assert reasoning["effort"] == "xhigh"

    def test_codex_default_xhigh(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["reasoning"]["effort"] == "xhigh"

    def test_codex_reasoning_disabled(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        agent.reasoning_config = {"enabled": False}
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "reasoning" not in kwargs
        assert kwargs["include"] == []

    def test_codex_reasoning_low(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        agent.reasoning_config = {"enabled": True, "effort": "low"}
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["reasoning"]["effort"] == "low"

    def test_openrouter_reasoning_config_override(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        agent.reasoning_config = {"enabled": True, "effort": "medium"}
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["reasoning"]["effort"] == "medium"
