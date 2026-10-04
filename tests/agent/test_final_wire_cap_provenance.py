"""Finite cap provenance through actual builders and final SDK/native bodies."""
import json
import httpx
import pytest
from openai import OpenAI
from anthropic import Anthropic, AnthropicBedrock
from providers.base import ProviderProfile
from agent.transports.chat_completions import ChatCompletionsTransport
from agent.transports.codex import ResponsesApiTransport
from agent.anthropic_adapter import build_anthropic_kwargs, _get_anthropic_max_output
from agent.bedrock_adapter import build_converse_kwargs
from agent.gemini_native_adapter import GeminiNativeClient, GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
from agent.final_wire_admission import (
    COVERED_MAIN, FinalAttemptIdentity, KNOWN_R, PROVIDER_DEFAULT_UNRESOLVED,
    bind_attempt_identity, project_final_body, wrap_httpx_client_transports,
    intercepted_openai_class, intercepted_anthropic_class,
)
from tests.agent.test_final_wire_native_egress import converse_client

ROWS = [{"role": "system", "content": "inert system"}, {"role": "user", "content": "hi"}]
CHAT_LAYERS = ["omitted", "user", "profile", "ephemeral", "override", "sdk_extra"]


@pytest.mark.parametrize("layer", CHAT_LAYERS)
@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_chat_cap_builder_profile_ephemeral_and_sdk_final_authority(layer, field):
    params = {"max_tokens_param_fn": lambda n: {field: n}}
    expected = None
    if layer != "omitted":
        params.update(provider_profile=ProviderProfile(name="inert", default_max_tokens=23))
        expected = 23
    if layer in {"user", "ephemeral", "override", "sdk_extra"}:
        params["max_tokens"] = 19
        expected = 19
    if layer == "ephemeral":
        params["ephemeral_max_output_tokens"] = 17
        expected = 17
    if layer in {"override", "sdk_extra"}:
        params["request_overrides"] = {field: 31}
        expected = 31
    if layer == "sdk_extra":
        params["request_overrides"]["extra_body"] = {field: 41, "messages": [{"role": "user", "content": "replacement"}]}
        expected = 41
    kwargs = ChatCompletionsTransport().build_kwargs("inert", ROWS, **params)
    exercise_http("chat_completions", kwargs, expected, replacement=layer == "sdk_extra")


@pytest.mark.parametrize("layer", ["omitted", "user", "override_before_configured", "sdk_extra", "consumer_omission", "consumer_explicit"])
def test_responses_configured_override_consumer_and_sdk_cap(layer):
    consumer = layer.startswith("consumer")
    expected = None if layer in {"omitted", "consumer_omission"} else 19
    params = {"instructions": "inert system", "is_codex_backend": consumer}
    if layer != "omitted":
        params["max_tokens"] = 19
    if layer == "override_before_configured":
        params["request_overrides"] = {"max_output_tokens": 31}
    if layer == "consumer_explicit":
        params["request_overrides"] = {"max_output_tokens": 31}
        expected = 31
    if layer == "sdk_extra":
        params["request_overrides"] = {"extra_body": {"max_output_tokens": 41, "input": "replacement"}}
        expected = 41
    kwargs = ResponsesApiTransport().build_kwargs("inert", ROWS, **params)
    # Consumer omission is preserved, not converted to a new invented default.
    if layer == "consumer_omission":
        assert "max_output_tokens" not in kwargs
    exercise_http("codex_responses", kwargs, expected, replacement=layer == "sdk_extra")


@pytest.mark.parametrize("family", ["anthropic_messages", "anthropic_bedrock"])
@pytest.mark.parametrize("layer", ["user", "default", "thinking", "sdk_extra"])
def test_anthropic_actual_default_post_thinking_and_sdk_reservation(family, layer):
    reasoning = {"enabled": True, "effort": "high"} if layer == "thinking" else None
    cap = None if layer == "default" else 19
    kwargs = build_anthropic_kwargs("claude-sonnet-4-5", ROWS, [], cap, reasoning)
    expected = _get_anthropic_max_output("claude-sonnet-4-5") if layer == "default" else 19
    if layer == "thinking":
        expected = kwargs["max_tokens"]
        assert expected > 19
    if layer == "sdk_extra":
        kwargs["extra_body"] = {"max_tokens": 41, "messages": [{"role": "user", "content": "replacement"}]}
        expected = 41
    exercise_http(family, kwargs, expected, replacement=layer == "sdk_extra")


@pytest.mark.parametrize("layer", ["omitted", "user", "before_send_override"])
def test_converse_final_cap_after_native_serialization_and_events(layer):
    client, delegate = converse_client()
    expected = None if layer == "omitted" else 19
    kwargs = build_converse_kwargs("inert", ROWS, max_tokens=expected)
    if layer == "before_send_override":
        expected = 41
        def mutate(request, **kw):
            body = json.loads(request.body)
            body["inferenceConfig"]["maxTokens"] = 41
            request.body = json.dumps(body).encode()
        client.meta.events.register_last("before-send.bedrock-runtime.Converse", mutate)
    try:
        ident = identity("bedrock_converse")
        with bind_attempt_identity(ident):
            client.converse(**kwargs)
        assert len(delegate.seen) == 1
        assert_reservation(delegate.seen[0][0], ident, expected)
    finally:
        client.close()


@pytest.mark.parametrize("layer", ["default", "user", "profile", "ephemeral", "thinking", "ignored_extra_cap"])
def test_gemini_facade_cap_layers_and_native_final_default(layer):
    params = {"provider_profile": ProviderProfile(name="inert", default_max_tokens=23), "max_tokens_param_fn": lambda n: {"max_tokens": n}}
    expected = 23
    if layer in {"default", "user", "thinking", "ignored_extra_cap"}:
        params.pop("provider_profile")
        expected = GEMINI_DEFAULT_MAX_OUTPUT_TOKENS if layer == "default" else 19
    if layer in {"user", "thinking", "ignored_extra_cap", "ephemeral"}:
        params["max_tokens"] = 19
    if layer == "ephemeral":
        params["ephemeral_max_output_tokens"] = 17
        expected = 17
    kwargs = ChatCompletionsTransport().build_kwargs("gemini-inert", ROWS, **params)
    if layer == "thinking":
        kwargs["extra_body"] = {"thinking_config": {"thinkingBudget": 8192}}
        expected = GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    if layer == "ignored_extra_cap":
        kwargs["extra_body"] = {"max_tokens": 41}
    exercise_http("gemini_native", kwargs, expected)


def identity(family):
    return FinalAttemptIdentity(COVERED_MAIN, family, "inert", "https://inert.invalid", 1000000, "provenance")


def assert_reservation(body, ident, expected):
    snap = project_final_body(body, ident)
    assert snap.coverage != "UNSUPPORTED"
    assert snap.resolved_r == expected
    assert snap.reservation_state == (PROVIDER_DEFAULT_UNRESOLVED if expected is None else KNOWN_R)
    assert snap.estimated_input == sum(n for _, n in snap.bucket_totals)


def exercise_http(family, kwargs, expected, replacement=False):
    bodies = []
    def receive(request):
        bodies.append(json.loads(request.content))
        if family == "gemini_native":
            payload = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        elif family.startswith("anthropic"):
            payload = {"id": "inert", "type": "message", "role": "assistant", "model": "inert", "content": [], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
        elif family == "codex_responses":
            payload = {"id": "inert", "object": "response", "created_at": 0, "status": "completed", "output": []}
        else:
            payload = {"id": "inert", "object": "chat.completion", "created": 0, "model": "inert", "choices": []}
        return httpx.Response(200, json=payload)
    with httpx.Client(transport=httpx.MockTransport(receive)) as http:
        wrap_httpx_client_transports(http)
        if family == "gemini_native":
            sdk = GeminiNativeClient(api_key="inert", base_url="https://inert.invalid", http_client=http)
            call = lambda: sdk.chat.completions.create(**kwargs)
        elif family.startswith("anthropic"):
            cls = intercepted_anthropic_class(AnthropicBedrock if family == "anthropic_bedrock" else Anthropic)
            auth = {"aws_access_key": "inert", "aws_secret_key": "inert", "aws_region": "us-east-1"} if family == "anthropic_bedrock" else {"api_key": "inert"}
            sdk = cls(**auth, base_url="https://inert.invalid", http_client=http, timeout=600)
            call = lambda: sdk.messages.create(**kwargs)
        else:
            sdk = intercepted_openai_class(OpenAI)(api_key="inert", base_url="https://inert.invalid", http_client=http)
            call = (lambda: sdk.responses.create(**kwargs)) if family == "codex_responses" else (lambda: sdk.chat.completions.create(**kwargs))
        ident = identity(family)
        with bind_attempt_identity(ident):
            call()
        assert len(bodies) == 1
        body = bodies[0]
        assert_reservation(body, ident, expected)
        if replacement:
            assert "hi" not in json.dumps(body.get("messages", body.get("input")))
            assert "replacement" in json.dumps(body)
