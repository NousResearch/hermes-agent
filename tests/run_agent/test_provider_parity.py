"""Provider parity tests: verify that AIAgent builds correct API kwargs
and handles responses properly for all supported providers.

Ensures changes to one provider path don't silently break another.
"""

import base64
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from agent.codex_responses_adapter import _chat_content_to_responses_parts, _chat_messages_to_responses_input, _normalize_codex_response, _preflight_codex_input_items

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


def _fake_invoke_jwt() -> str:
    def _part(payload):
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return (
        f"{_part({'alg': 'none', 'typ': 'JWT'})}."
        f"{_part({'scope': 'inference:invoke', 'exp': 4102444800})}.sig"
    )


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")
    def close(self):
        pass


@pytest.fixture(autouse=True)
def _reset_auxiliary_provider_state():
    from agent.auxiliary_client import _reset_aux_unhealthy_cache

    _reset_aux_unhealthy_cache()
    yield
    _reset_aux_unhealthy_cache()


def _make_agent(monkeypatch, provider, api_mode="chat_completions", base_url="https://openrouter.ai/api/v1", model=None):
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: _tool_defs("web_search", "terminal"))
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    kwargs = dict(
        api_key="test-key",
        base_url=base_url,
        provider=provider,
        api_mode=api_mode,
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    if model:
        kwargs["model"] = model
    elif provider == "nous":
        kwargs["model"] = "gpt-5"
    base_url="https://openrouter.ai/api/v1",
    api_key="test-key",
    base_url="https://openrouter.ai/api/v1",
    return AIAgent(**kwargs)


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
        agent.model = "anthropic/claude-sonnet-4-20250514"
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        extra = kwargs.get("extra_body", {})
        assert "reasoning" in extra
        assert extra["reasoning"]["enabled"] is True


    def test_no_responses_api_fields(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "input" not in kwargs
        assert "instructions" not in kwargs
        assert "store" not in kwargs

    def test_strips_codex_only_tool_call_fields_from_chat_messages(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Checking now.",
                "codex_reasoning_items": [
                    {"type": "reasoning", "id": "rs_1", "encrypted_content": "blob"},
                ],
                "tool_calls": [
                    {
                        "id": "call_123",
                        "call_id": "call_123",
                        "response_item_id": "fc_123",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{\"command\":\"pwd\"}"},
                        "extra_content": {"thought_signature": "opaque"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "/tmp"},
        ]

        kwargs = agent._build_api_kwargs(messages)

        assistant_msg = kwargs["messages"][1]
        tool_call = assistant_msg["tool_calls"][0]

        assert "codex_reasoning_items" not in assistant_msg
        assert tool_call["id"] == "call_123"
        assert tool_call["function"]["name"] == "terminal"
        # extra_content (Gemini thought_signature) is stripped for non-Gemini
        # targets — strict providers like Fireworks 400 on it. The agent here
        # is not a Gemini model, so it must be dropped.
        assert "extra_content" not in tool_call
        assert "call_id" not in tool_call
        assert "response_item_id" not in tool_call

        # Original stored history must remain unchanged (only the outgoing copy
        # is sanitized) — Codex/Responses replay relies on these fields.
        assert messages[1]["tool_calls"][0]["call_id"] == "call_123"
        assert messages[1]["tool_calls"][0]["response_item_id"] == "fc_123"
        assert "codex_reasoning_items" in messages[1]
        assert messages[1]["tool_calls"][0]["extra_content"] == {"thought_signature": "opaque"}

    def test_keeps_extra_content_for_gemini_target(self, monkeypatch):
        """Gemini-family targets must keep extra_content (thought_signature) —
        Gemini 3 thinking models 400 without it replayed on the next turn.
        """
        agent = _make_agent(monkeypatch, "openrouter", model="google/gemini-3-pro-preview")
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Checking now.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "call_id": "call_123",
                        "response_item_id": "fc_123",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{\"command\":\"pwd\"}"},
                        "extra_content": {"google": {"thought_signature": "opaque"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "/tmp"},
        ]

        kwargs = agent._build_api_kwargs(messages)
        tool_call = kwargs["messages"][1]["tool_calls"][0]
        assert tool_call["extra_content"] == {"google": {"thought_signature": "opaque"}}
        # call_id/response_item_id still stripped regardless of model
        assert "call_id" not in tool_call
        assert "response_item_id" not in tool_call

        # Original stored history must remain unchanged for Responses replay mode.
        assert messages[1]["tool_calls"][0]["call_id"] == "call_123"
        assert messages[1]["tool_calls"][0]["response_item_id"] == "fc_123"
        assert messages[1]["tool_calls"][0]["extra_content"] == {
            "google": {"thought_signature": "opaque"}
        }

    def test_gemini_native_passes_base_url_for_top_level_thinking_config(self, monkeypatch):
        agent = _make_agent(
            monkeypatch,
            "gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-3-flash-preview",
        )
        agent.reasoning_config = {"enabled": True, "effort": "high"}
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["thinking_config"] == {
            "includeThoughts": True,
            "thinkingLevel": "high",
        }
        assert "extra_body" not in kwargs["extra_body"]


    def test_should_sanitize_tool_calls_codex_vs_chat(self, monkeypatch):
        """Codex API should NOT sanitize, all other APIs should sanitize."""
        # Codex mode should NOT need sanitization
        codex_agent = _make_agent(monkeypatch, "openrouter")
        codex_agent.api_mode = "codex_responses"
        assert codex_agent._should_sanitize_tool_calls() is False

        # Chat completions mode should need sanitization
        chat_agent = _make_agent(monkeypatch, "openrouter")
        chat_agent.api_mode = "chat_completions"
        assert chat_agent._should_sanitize_tool_calls() is True

        # Anthropic mode should need sanitization
        anthropic_agent = _make_agent(monkeypatch, "openrouter")
        anthropic_agent.api_mode = "anthropic_messages"
        assert anthropic_agent._should_sanitize_tool_calls() is True

    def _api_msg_with_extra_content(self):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "call_id": "call_1", "type": "function",
                 "extra_content": {"google": {"thought_signature": "SIG_123"}},
                 "function": {"name": "t", "arguments": "{}"}},
            ],
        }

    def test_sanitize_tool_calls_strips_extra_content_for_strict_model(self, monkeypatch):
        """Strict providers reject extra_content; strip it for non-Gemini models."""
        agent = _make_agent(monkeypatch, "openrouter")
        api_msg = self._api_msg_with_extra_content()
        result = agent._sanitize_tool_calls_for_strict_api(
            api_msg, model="accounts/fireworks/models/llama-v3p1-70b"
        )
        assert "extra_content" not in result["tool_calls"][0]
        assert "call_id" not in result["tool_calls"][0]




class TestDeveloperRoleSwap:
    """GPT-5 and Codex models should get 'developer' instead of 'system' role."""

    @pytest.mark.parametrize("model", [
        "openai/gpt-5",
        "openai/gpt-5-turbo",
        "openai/gpt-5.4",
        "gpt-5-mini",
        "openai/codex-mini",
        "codex-mini-latest",
        "openai/codex-pro",
    ])
    def test_gpt5_codex_get_developer_role(self, monkeypatch, model):
        agent = _make_agent(monkeypatch, "openrouter")
        agent.model = model
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["messages"][0]["role"] == "developer"
        assert kwargs["messages"][0]["content"] == "You are helpful."
        assert kwargs["messages"][1]["role"] == "user"






class TestBuildApiKwargsChatCompletionsServiceTier:
    """service_tier via request_overrides works on the chat_completions path."""

    def test_includes_service_tier_via_request_overrides(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        agent.model = "gpt-4.1"
        agent.request_overrides = {"service_tier": "priority"}
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["service_tier"] == "priority"




class TestBuildApiKwargsKimiNoTemperatureOverride:
    def test_kimi_for_coding_omits_temperature(self, monkeypatch):
        """Temperature should NOT be set client-side for Kimi models.

        The Kimi gateway selects the correct temperature server-side.
        """
        agent = _make_agent(
            monkeypatch,
            "kimi-coding",
            base_url="https://api.kimi.com/coding/v1",
            model="kimi-for-coding",
        )
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "temperature" not in kwargs


class TestBuildApiKwargsAIGateway:
    def test_uses_chat_completions_format(self, monkeypatch):
        agent = _make_agent(monkeypatch, "ai-gateway", base_url="https://ai-gateway.vercel.sh/v1", model="gpt-4o")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "messages" in kwargs
        assert "model" in kwargs
        assert kwargs["messages"][-1]["content"] == "hi"

    def test_no_responses_api_fields(self, monkeypatch):
        agent = _make_agent(monkeypatch, "ai-gateway", base_url="https://ai-gateway.vercel.sh/v1", model="gpt-4o")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "input" not in kwargs
        assert "instructions" not in kwargs
        assert "store" not in kwargs

    def test_includes_reasoning_in_extra_body(self, monkeypatch):
        agent = _make_agent(monkeypatch, "ai-gateway", base_url="https://ai-gateway.vercel.sh/v1", model="gpt-4o")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        extra = kwargs.get("extra_body", {})
        assert "reasoning" in extra
        assert extra["reasoning"]["enabled"] is True

    def test_includes_tools(self, monkeypatch):
        agent = _make_agent(monkeypatch, "ai-gateway", base_url="https://ai-gateway.vercel.sh/v1", model="gpt-4o")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "tools" in kwargs
        tool_names = [t["function"]["name"] for t in kwargs["tools"]]
        assert "web_search" in tool_names


class TestBuildApiKwargsNousPortal:
    def test_includes_nous_product_tags(self, monkeypatch):
        from agent.portal_tags import nous_portal_tags
        agent = _make_agent(
            monkeypatch,
            "nous",
            base_url="https://inference-api.nousresearch.com/v1",
            model="gpt-5",
        )
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        extra = kwargs.get("extra_body", {})
        assert extra.get("tags") == nous_portal_tags(session_id=agent.session_id)

    def test_uses_chat_completions_format(self, monkeypatch):
        agent = _make_agent(
            monkeypatch,
            "nous",
            base_url="https://inference-api.nousresearch.com/v1",
            model="gpt-5",
        )
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


    def test_fireworks_tool_call_payload_strips_codex_only_fields(self, monkeypatch):
        agent = _make_agent(
            monkeypatch,
            "custom",
            base_url="https://api.fireworks.ai/inference/v1",
        )
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Checking now.",
                "codex_reasoning_items": [
                    {"type": "reasoning", "id": "rs_1", "encrypted_content": "blob"},
                ],
                "tool_calls": [
                    {
                        "id": "call_fw_123",
                        "call_id": "call_fw_123",
                        "response_item_id": "fc_fw_123",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": "{\"command\":\"pwd\"}",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_fw_123", "content": "/tmp"},
        ]

        kwargs = agent._build_api_kwargs(messages)

        assert kwargs["tools"][0]["function"]["name"] == "web_search"
        assert "input" not in kwargs
        assert kwargs.get("extra_body", {}) == {}

        assistant_msg = kwargs["messages"][1]
        tool_call = assistant_msg["tool_calls"][0]

        assert "codex_reasoning_items" not in assistant_msg
        assert tool_call["id"] == "call_fw_123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "terminal"
        assert "call_id" not in tool_call
        assert "response_item_id" not in tool_call


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


    def test_includes_service_tier_via_request_overrides(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        agent.model = "gpt-5.4"
        agent.service_tier = "priority"
        agent.request_overrides = {"service_tier": "priority"}
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["service_tier"] == "priority"



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

    def test_azure_foundry_reasoning_id_survives_full_build_api_kwargs(self, monkeypatch):
        """Regression: ``base_url`` must reach ``build_kwargs()``.

        The Azure Foundry reasoning-id fix is keyed off the ``base_url``
        kwarg that ``build_api_kwargs()`` forwards to the transport. If that
        forwarding is dropped, the transport silently sees ``None``, decides
        the endpoint is not Azure, and strips the replayed reasoning ``id``
        again — reintroducing the very bug this path exists to fix. Asserting
        on ``build_kwargs()`` directly cannot catch that, so drive the whole
        ``_build_api_kwargs`` path here.
        """
        agent = _make_agent(
            monkeypatch, "azure-foundry", api_mode="codex_responses",
            base_url="https://paperclip.services.ai.azure.com/models",
            model="gpt-5.5",
        )
        messages = [
            {
                "role": "assistant",
                "content": "thinking",
                "codex_reasoning_items": [
                    {
                        "type": "reasoning",
                        "id": "rs_123",
                        "encrypted_content": "enc_blob",
                        "summary": [{"type": "summary_text", "text": "brief"}],
                        "status": "completed",
                        "response_id": "resp_123",
                    }
                ],
            },
            {"role": "user", "content": "continue"},
        ]

        kwargs = agent._build_api_kwargs(messages)

        reasoning_item = next(
            item for item in kwargs["input"] if item.get("type") == "reasoning"
        )
        assert reasoning_item["id"] == "rs_123"

    def test_non_azure_reasoning_id_stripped_through_full_build_api_kwargs(self, monkeypatch):
        """The Azure carve-out must not leak onto other Responses endpoints.

        OpenAI/Codex with ``store=False`` returns 404 when a replayed
        reasoning item carries an ``id``, so the id must still be stripped
        everywhere that is not Azure Foundry.
        """
        agent = _make_agent(
            monkeypatch, "openai-codex", api_mode="codex_responses",
            base_url="https://chatgpt.com/backend-api/codex",
        )
        messages = [
            {
                "role": "assistant",
                "content": "thinking",
                "codex_reasoning_items": [
                    {
                        "type": "reasoning",
                        "id": "rs_123",
                        "encrypted_content": "enc_blob",
                        "summary": [{"type": "summary_text", "text": "brief"}],
                    }
                ],
            },
            {"role": "user", "content": "continue"},
        ]

        kwargs = agent._build_api_kwargs(messages)

        reasoning_item = next(
            (item for item in kwargs["input"] if item.get("type") == "reasoning"),
            None,
        )
        if reasoning_item is not None:
            assert "id" not in reasoning_item


# ── Message conversion tests ────────────────────────────────────────────────

class TestChatMessagesToResponsesInput:
    """Verify _chat_messages_to_responses_input for Codex mode."""

    def test_user_message_passes_through(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [{"role": "user", "content": "hello"}]
        items = _chat_messages_to_responses_input(messages)
        assert items == [{"role": "user", "content": "hello"}]

    def test_system_messages_filtered(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
        ]
        items = _chat_messages_to_responses_input(messages)
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
        items = _chat_messages_to_responses_input(messages)
        fc_items = [i for i in items if i.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["name"] == "web_search"
        assert fc_items[0]["call_id"] == "call_abc"






    def test_preflight_preserves_assistant_output_text(self, monkeypatch):
        """_preflight_codex_input_items must preserve output_text for assistant."""
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        raw_input = [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
        ]
        normalized = _preflight_codex_input_items(raw_input)
        user_content = normalized[0]["content"]
        asst_content = normalized[1]["content"]
        assert user_content[0]["type"] == "input_text"
        assert asst_content[0]["type"] == "output_text"

    def test_full_round_trip_with_list_content(self, monkeypatch):
        """End-to-end: user + assistant with list content through both stages."""
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
            {"role": "user", "content": [{"type": "text", "text": "continue"}]},
        ]
        items = _chat_messages_to_responses_input(messages)
        normalized = _preflight_codex_input_items(items)

        # User items use input_text
        assert normalized[0]["content"][0]["type"] == "input_text"
        assert normalized[2]["content"][0]["type"] == "input_text"
        # Assistant item uses output_text
        assert normalized[1]["content"][0]["type"] == "output_text"


class TestChatContentToResponsesParts:
    """Unit tests for _chat_content_to_responses_parts role parameter (#15687)."""

    def test_default_role_emits_input_text(self):
        """Default (user) role emits input_text."""
        result = _chat_content_to_responses_parts([{"type": "text", "text": "hello"}])
        assert result[0]["type"] == "input_text"




    def test_assistant_role_with_mixed_input_output_text_types(self):
        """Parts already marked input_text or output_text get normalized to role's type."""
        parts = [
            {"type": "input_text", "text": "a"},
            {"type": "output_text", "text": "b"},
            {"type": "text", "text": "c"},
        ]
        result = _chat_content_to_responses_parts(parts, role="assistant")
        # All text parts should become output_text regardless of original type
        assert all(p["type"] == "output_text" for p in result)
        assert [p["text"] for p in result] == ["a", "b", "c"]


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
        msg, reason = _normalize_codex_response(response)
        assert msg.content == "Hello!"
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
        msg, reason = _normalize_codex_response(response)
        assert msg.codex_reasoning_items is not None
        assert len(msg.codex_reasoning_items) == 1
        assert msg.codex_reasoning_items[0]["encrypted_content"] == "gAAAA_secret_blob_123"
        assert msg.codex_reasoning_items[0]["id"] == "rs_456"



    def test_message_items_captured_with_id_and_phase(self, monkeypatch):
        """Exact message items (with id/phase) must be captured for cache replay."""
        agent = self._make_codex_agent(monkeypatch)
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message", status="completed", id="msg_abc",
                    phase="commentary",
                    content=[SimpleNamespace(type="output_text", text="Thinking...")],
                ),
                SimpleNamespace(
                    type="message", status="completed", id="msg_def",
                    phase="final_answer",
                    content=[SimpleNamespace(type="output_text", text="Done!")],
                ),
            ],
            status="completed",
        )
        msg, reason = _normalize_codex_response(response)
        assert msg.codex_message_items is not None
        assert len(msg.codex_message_items) == 2
        assert msg.codex_message_items[0]["id"] == "msg_abc"
        assert msg.codex_message_items[0]["phase"] == "commentary"
        assert msg.codex_message_items[0]["content"][0]["text"] == "Thinking..."
        assert msg.codex_message_items[1]["id"] == "msg_def"
        assert msg.codex_message_items[1]["phase"] == "final_answer"
        assert msg.codex_message_items[1]["content"][0]["text"] == "Done!"



class TestChatMessagesToResponsesInputMessageItems:
    """Verify codex_message_items are replayed verbatim instead of reconstructed."""

    def test_replays_exact_message_items(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [
            {
                "role": "assistant",
                "content": "Hello world",
                "codex_message_items": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "id": "msg_123",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "Hello world"}],
                    },
                ],
            },
            {"role": "user", "content": "follow up"},
        ]
        items = _chat_messages_to_responses_input(messages)
        msg_items = [i for i in items if i.get("type") == "message"]
        assert len(msg_items) == 1
        assert msg_items[0]["id"] == "msg_123"
        assert msg_items[0]["phase"] == "final_answer"
        assert msg_items[0]["content"][0]["text"] == "Hello world"

    def test_fallback_to_plain_when_no_message_items(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        messages = [{"role": "assistant", "content": "Hello world"}]
        items = _chat_messages_to_responses_input(messages)
        assert items == [{"role": "assistant", "content": "Hello world"}]



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



# ── Auxiliary client provider resolution ─────────────────────────────────────

class TestAuxiliaryClientProviderPriority:
    """Verify auxiliary client resolution doesn't break for any provider."""

    def test_openrouter_always_wins(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        from agent.auxiliary_client import _OPENROUTER_MODEL, get_text_auxiliary_client
        with patch("agent.auxiliary_client.OpenAI") as mock:
            client, model = get_text_auxiliary_client()
        assert model == _OPENROUTER_MODEL
        assert "openrouter" in str(mock.call_args.kwargs["base_url"]).lower()

    def test_nous_when_no_openrouter(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from agent.auxiliary_client import _NOUS_MODEL, get_text_auxiliary_client
        nous_auth = {
            "access_token": _fake_invoke_jwt(),
            "scope": "inference:invoke",
        }
        with patch("agent.auxiliary_client._read_nous_auth", return_value=nous_auth), \
             patch("agent.auxiliary_client.OpenAI") as mock, \
             patch("hermes_cli.models.get_nous_recommended_aux_model", return_value=None):
            client, model = get_text_auxiliary_client()
        assert model == _NOUS_MODEL

    def test_custom_endpoint_when_no_nous(self, monkeypatch):
        """Custom endpoint is used when no OpenRouter/Nous keys are available.

        Since the March 2026 config refactor, OPENAI_BASE_URL env var is no
        longer consulted — base_url comes from config.yaml via
        resolve_runtime_provider.  Mock _resolve_custom_runtime directly.
        """
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "local-key")
        from agent.auxiliary_client import get_text_auxiliary_client
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._resolve_custom_runtime",
                   return_value=("http://localhost:1234/v1", "local-key")), \
             patch("agent.auxiliary_client.OpenAI") as mock:
            client, model = get_text_auxiliary_client()
        assert mock.call_args.kwargs["base_url"] == "http://localhost:1234/v1"

    def test_codex_not_in_auto_fallback(self, monkeypatch):
        """Codex is deliberately NOT part of the auto fallback chain.

        ChatGPT-account Codex gates which models it accepts via an
        undocumented, shifting allow-list, so falling through to Codex with
        a hardcoded default model breaks silently whenever OpenAI rotates
        the list.  When nothing else is available, ``get_text_auxiliary_client``
        now returns (None, None) rather than guessing a Codex model.
        """
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from agent.auxiliary_client import get_text_auxiliary_client
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value="codex-tok"), \
             patch("agent.auxiliary_client.OpenAI"):
            client, model = get_text_auxiliary_client()
        assert client is None
        assert model is None


# ── Provider routing tests ───────────────────────────────────────────────────

class TestProviderRouting:
    """Verify provider_routing config flows into extra_body.provider."""

    def test_sort_throughput(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        agent.provider_sort = "throughput"
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["provider"]["sort"] == "throughput"






    def test_no_routing_when_unset(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "provider" not in kwargs.get("extra_body", {}).get("provider", {}) or \
               kwargs.get("extra_body", {}).get("provider") is None or \
               "only" not in kwargs.get("extra_body", {}).get("provider", {})




# ── Codex reasoning items preflight tests ────────────────────────────────────

class TestCodexReasoningPreflight:
    """Verify reasoning items pass through preflight normalization."""

    def test_reasoning_item_passes_through(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        raw_input = [
            {"role": "user", "content": "hello"},
            {"type": "reasoning", "encrypted_content": "abc123encrypted", "id": "r_001",
             "summary": [{"type": "summary_text", "text": "Thinking about it"}]},
            {"role": "assistant", "content": "hi there"},
        ]
        normalized = _preflight_codex_input_items(raw_input)
        reasoning_items = [i for i in normalized if i.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["encrypted_content"] == "abc123encrypted"
        # Note: "id" is intentionally excluded from normalized output —
        # with store=False the API returns 404 on server-side id resolution.
        # The id is only used for local deduplication via seen_ids.
        assert "id" not in reasoning_items[0]
        assert reasoning_items[0]["summary"] == [{"type": "summary_text", "text": "Thinking about it"}]

    def test_reasoning_item_without_id(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        raw_input = [
            {"type": "reasoning", "encrypted_content": "abc123"},
        ]
        normalized = _preflight_codex_input_items(raw_input)
        assert len(normalized) == 1
        assert "id" not in normalized[0]
        assert normalized[0]["summary"] == []  # default empty summary


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
        items = _chat_messages_to_responses_input(messages)
        reasoning_items = [i for i in items if isinstance(i, dict) and i.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["encrypted_content"] == "enc123"


# ── Reasoning effort consistency tests ───────────────────────────────────────

class TestReasoningEffortDefaults:
    """Verify reasoning effort defaults to medium across all provider paths."""

    def test_openrouter_default_medium(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter")
        agent.model = "anthropic/claude-sonnet-4-20250514"
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        reasoning = kwargs["extra_body"]["reasoning"]
        assert reasoning["effort"] == "medium"

    def test_codex_default_medium(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openai-codex", api_mode="codex_responses",
                            base_url="https://chatgpt.com/backend-api/codex")
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["reasoning"]["effort"] == "medium"



