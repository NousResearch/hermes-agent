"""User configuration cannot impose generation caps; wire budgets remain internal."""

import json


def test_legacy_user_caps_do_not_change_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MAX_TOKENS", "13")
    config = {
        "model": {"default": "fixture", "provider": "local-fixture", "max_tokens": 17},
        "providers": {"local-fixture": {"api": "http://127.0.0.1:1/v1", "api_key": "fixture", "max_output_tokens": 19}},
    }
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    from gateway.run import _resolve_runtime_agent_kwargs
    from gateway.platforms.api_server import _resolve_request_runtime_agent_kwargs
    from hermes_cli.moa_config import _normalize_preset
    from agent.models_dev import _override_to_catalog_shape

    runtime = _resolve_runtime_agent_kwargs()
    assert runtime.get("max_tokens") is None
    assert runtime["base_url"].startswith("http://127.0.0.1:1")
    request = _resolve_request_runtime_agent_kwargs(provider="local-fixture", target_model="fixture")
    assert request.get("max_tokens") is None
    preset = _normalize_preset({"max_tokens": 23, "reference_max_tokens": 29})
    assert "max_tokens" not in preset and "reference_max_tokens" not in preset
    patch, _ = _override_to_catalog_shape({"context_window": 10000, "max_output_tokens": 31})
    assert patch["limit"] == {"context": 10000}
    from agent.auxiliary_client import _compression_fast_lane_controls
    route = {"provider": "custom", "model": "fixture", "reasoning_effort": "none", "max_output_tokens": 47}
    cap, body = _compression_fast_lane_controls(
        "compression", actual_provider="custom", actual_model="fixture",
        requested_provider="custom", requested_model="fixture", route_config=route,
        leak_guard_config=route, max_tokens=None, extra_body={},
    )
    assert cap is None
    assert body["reasoning"]["enabled"] is False


def test_optional_wire_caps_omitted_required_and_internal_preserved():
    from agent.transports.chat_completions import ChatCompletionsTransport
    from agent.transports.bedrock import BedrockTransport
    from agent.transports.anthropic import AnthropicTransport

    messages = [{"role": "user", "content": "fixture"}]
    chat = ChatCompletionsTransport().build_kwargs(
        "claude-fixture", messages, anthropic_max_output=65536,
        max_tokens_param_fn=lambda value: {"max_tokens": value},
    )
    assert "max_tokens" not in chat
    from providers import get_provider_profile
    custom = ChatCompletionsTransport().build_kwargs(
        "fixture", messages, provider_profile=get_provider_profile("custom"),
        max_tokens_param_fn=lambda value: {"max_tokens": value},
    )
    assert "max_tokens" not in custom
    bedrock = BedrockTransport().build_kwargs("amazon.nova-pro-v1:0", messages)
    assert "maxTokens" not in bedrock.get("inferenceConfig", {})
    native = AnthropicTransport().build_kwargs("claude-sonnet-4-5", messages)
    assert native["max_tokens"] > 0
    bounded = ChatCompletionsTransport().build_kwargs(
        "fixture", messages, max_tokens=43,
        max_tokens_param_fn=lambda value: {"max_tokens": value},
    )
    assert bounded["max_tokens"] == 43


# Route gating of the per-vendor output table --------------------------------------------------
# ``_ANTHROPIC_OUTPUT_LIMITS`` values describe endpoints that REQUIRE the Messages-API
# ``max_tokens`` (Anthropic, plus the Anthropic-compatible MiniMax and DashScope ``qwen3``
# entries). They are therefore resolved on the Anthropic-Messages ROUTE only. A private
# fine-tune whose id merely CONTAINS a table key — ``pplx-qwen38-nvfp4-dflash2`` contains
# ``qwen3`` — served by a self-hosted openai-compatible engine must never inherit the
# DashScope 65536 cap: that engine's window IS 65536, so cap + any prompt > window and
# every request dies with HTTP 400 ("requested 65536 output tokens").

_PRIVATE_VLLM_ID = "pplx-qwen38-nvfp4-dflash2"
_DASHSCOPE_QWEN3_CAP = 65_536
_MESSAGES = [{"role": "user", "content": "fixture"}]


class _CustomOpenAICompatAgent:
    """Minimal agent surface consumed by ``_build_chat_completions_kwargs``."""

    def __init__(self, model: str, max_tokens: int | None = None):
        from agent.transports.chat_completions import ChatCompletionsTransport

        self.model = model
        self.provider = "custom"
        self.base_url = "http://127.0.0.1:8080/v1"
        self._base_url_lower = self.base_url
        self._base_url_hostname = "127.0.0.1"
        self.max_tokens = max_tokens
        self.session_id = "fixture-session"
        self._ollama_num_ctx = None
        self.openrouter_min_coding_score = None
        self._ephemeral_max_output_tokens = None
        self.providers_allowed = self.providers_ignored = self.providers_order = None
        self.provider_sort = self.provider_require_parameters = self.provider_data_collection = None
        self._transport = ChatCompletionsTransport()

    def _get_transport(self):
        return self._transport

    def _is_qwen_portal(self) -> bool:
        return False

    def _is_openrouter_url(self) -> bool:
        return False

    def _supports_reasoning_extra_body(self) -> bool:
        return False

    def _prepare_messages_for_non_vision_model(self, messages):
        return messages

    def _resolved_api_call_timeout(self):
        return None

    def _max_tokens_param(self, value):
        return {"max_tokens": value}


def _build_request(model: str, max_tokens: int | None = None) -> dict:
    from agent.chat_completion_helpers import _build_chat_completions_kwargs

    return _build_chat_completions_kwargs(
        _CustomOpenAICompatAgent(model, max_tokens=max_tokens), _MESSAGES, None, None, {}, None,
    )


def test_dashscope_cap_never_reaches_openai_compatible_requests():
    """(A) Custom openai-compatible provider + a table-key-matching private id: no injection."""
    from agent.transports.chat_completions import ChatCompletionsTransport

    built = _build_request(_PRIVATE_VLLM_ID)
    assert "max_tokens" not in built, (
        "the DashScope/Anthropic output cap must not be injected into a self-hosted "
        f"openai-compatible request; got max_tokens={built.get('max_tokens')!r} for "
        f"{_PRIVATE_VLLM_ID!r} (window is the same {_DASHSCOPE_QWEN3_CAP}, so the request "
        "can never fit)"
    )
    # Same seam one level lower: a caller that still passes the legacy kwarg changes nothing.
    chat = ChatCompletionsTransport().build_kwargs(
        _PRIVATE_VLLM_ID, _MESSAGES, anthropic_max_output=_DASHSCOPE_QWEN3_CAP,
        max_tokens_param_fn=lambda value: {"max_tokens": value},
    )
    assert "max_tokens" not in chat


def test_dashscope_cap_still_applies_on_its_own_route():
    """(B) Protection: the Anthropic-Messages route keeps the DashScope qwen3 cap."""
    from agent.anthropic_adapter import _get_anthropic_max_output, _resolve_anthropic_messages_max_tokens
    from agent.transports.anthropic import AnthropicTransport

    assert _get_anthropic_max_output(_PRIVATE_VLLM_ID) == _DASHSCOPE_QWEN3_CAP
    assert _resolve_anthropic_messages_max_tokens(None, _PRIVATE_VLLM_ID) == _DASHSCOPE_QWEN3_CAP
    native = AnthropicTransport().build_kwargs(_PRIVATE_VLLM_ID, _MESSAGES, max_tokens=None)
    assert native["max_tokens"] == _DASHSCOPE_QWEN3_CAP


def test_plain_model_id_behaviour_unchanged():
    """(C) Ids matching no table key keep the configured cap, or none at all."""
    assert _build_request("some-local-model", max_tokens=512)["max_tokens"] == 512
    assert "max_tokens" not in _build_request("some-local-model")
