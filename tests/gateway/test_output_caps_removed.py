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
