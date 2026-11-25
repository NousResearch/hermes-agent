"""Bedrock Ultrafast is a Responses capability, not a Converse tier alias.

Offline config -> runtime -> CLI/gateway handoff -> AIAgent -> OpenAI SDK HTTP
recordings. AWS replies below are fixtures, not live service/entitlement evidence.
"""
from types import SimpleNamespace
import json

import httpx
import pytest

from hermes_cli.models import model_supports_fast_mode, resolve_fast_mode_overrides

MODEL = "openai.gpt-6.1-sol"
BASE = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"


def test_bedrock_ultrafast_is_exact_model_route_and_tier():
    for provider in ("bedrock", "aws", "aws-bedrock", "amazon-bedrock", "amazon"):
        assert resolve_fast_mode_overrides(MODEL, provider=provider, base_url=BASE, tier="ultrafast") == {
            "service_tier": "ultrafast"}
    for tier in (None, "priority", "auto", "cold", "flex", "reserved", "default"):
        assert resolve_fast_mode_overrides(MODEL, provider="bedrock", base_url=BASE, tier=tier) is None
    for model in ("gpt-6.1-sol", "openai.gpt-6.1-sol-pro", "openai.gpt-6-astra",
                  "us." + MODEL, "global." + MODEL, "anthropic.claude-opus-5-5", MODEL + ":0"):
        assert resolve_fast_mode_overrides(model, provider="bedrock", base_url=BASE, tier="ultrafast") is None
    for base in (None, "", BASE.replace("us-east-1", "us-west-2"),
                 BASE.replace(".api.aws", ".api.aws.attacker.test"),
                 BASE.replace("https:", "http:"), BASE.replace("/openai/v1", "/v1"),
                 BASE + "?route=other", BASE + "#fragment", BASE.replace(".api.aws/", ".api.aws:8443/"),
                 "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
                 "https://bedrock-mantle.us-east-1.api.aws@attacker.test/openai/v1"):
        assert resolve_fast_mode_overrides(MODEL, provider="bedrock", base_url=base, tier="ultrafast") is None
    for provider in (None, "custom", "openai-api", "openrouter"):
        for tier in ("priority", "ultrafast"):
            for base in (BASE, None, "https://api.openai.com/v1"):
                assert resolve_fast_mode_overrides(MODEL, provider=provider, base_url=base, tier=tier) is None


def _configure(tmp_path, *, model=MODEL, region="us-east-1", tier="ultrafast"):
    from hermes_cli.config import atomic_config_write, load_config
    from hermes_constants import get_hermes_home
    atomic_config_write(get_hermes_home() / "config.yaml", {
        "model": {"provider": "bedrock", "default": model}, "bedrock": {"region": region},
        "agent": {"service_tier": tier}, "compression": {"enabled": False},
    })
    return load_config()


def _turn_route(cfg, runtime, surface):
    from agent.fast_mode import parse_service_tier
    from cli import HermesCLI
    from gateway.run import GatewayRunner
    tier = parse_service_tier(cfg["agent"]["service_tier"])
    model = cfg["model"]["default"]
    if surface == "cli":
        shell = SimpleNamespace(model=model, api_key=runtime["api_key"], base_url=runtime["base_url"],
                                provider=runtime["provider"], api_mode=runtime["api_mode"],
                                acp_command=None, acp_args=[], _credential_pool=None, service_tier=tier)
        return HermesCLI._resolve_turn_agent_config(shell, "hello")
    runner = object.__new__(GatewayRunner)
    runner._service_tier = tier
    runtime = {**runtime, "command": None, "args": [], "credential_pool": None, "max_tokens": None}
    return runner._resolve_turn_agent_config("hello", model, runtime)


@pytest.mark.parametrize("surface", ["cli", "gateway"])
@pytest.mark.parametrize("auth", ["bearer", "iam"])
@pytest.mark.parametrize("stream", [False, True])
def test_config_tier_reaches_real_sdk_preserving_off_body(tmp_path, monkeypatch, surface, auth, stream):
    pytest.importorskip("botocore.credentials", reason="botocore (bedrock extra) required")
    pytest.importorskip("boto3", reason="boto3 (bedrock extra) required")
    from botocore.credentials import Credentials
    import boto3
    from agent.codex_runtime import _consume_codex_event_stream
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent

    cfg = _configure(tmp_path)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    if auth == "bearer":
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bedrock-bearer")
    else:
        monkeypatch.setattr(boto3.Session, "get_credentials", lambda self: Credentials("testing", "testing"))
    runtime = resolve_runtime_provider(requested="bedrock", target_model=MODEL)
    assert runtime["api_mode"] == "codex_responses" and runtime["base_url"] == BASE
    requests = []

    def record(_transport, request):
        body = json.loads(request.content)
        requests.append(body)
        assert str(request.url) == BASE + "/responses"
        if auth == "bearer":
            assert request.headers["authorization"] == "Bearer test-bedrock-bearer"
        else:
            assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
            assert "/us-east-1/bedrock/aws4_request" in request.headers["authorization"]
        response = {"id": "resp_test", "object": "response", "created_at": 1, "model": MODEL,
                    "status": "completed", "output": [], "service_tier": "default",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
        if stream:
            events = [{"type": "response.output_text.delta", "delta": "ok"},
                      {"type": "response.completed", "response": response}]
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content="".join("data: " + json.dumps(e) + "\n\n" for e in events).encode())
        return httpx.Response(200, json=response)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", record)
    route = _turn_route(cfg, runtime, surface)
    agent = AIAgent(model=MODEL, provider="bedrock", api_mode=runtime["api_mode"],
                    base_url=runtime["base_url"], api_key=runtime["api_key"],
                    service_tier="ultrafast", request_overrides=route["request_overrides"],
                    quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
                    session_id="bedrock-ultrafast-test")
    messages = [{"role": "system", "content": "Keep the prefix stable."}, {"role": "user", "content": "hello"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_test", "type": "function",
                 "function": {"name": "lookup", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_test", "content": "offline result"}]
    tools = [{"type": "function", "function": {"name": "lookup", "description": "Test lookup",
              "parameters": {"type": "object", "properties": {}}}}]
    try:
        for tier, explicit in (("ultrafast", None), ("off", None), ("default", None),
                               ("ultrafast", {"service_tier": "default"}),
                               ("ultrafast", {"extra_body": {"service_tier": "default"}})):
            cfg["agent"]["service_tier"] = tier
            agent.request_overrides = explicit or _turn_route(cfg, runtime, surface)["request_overrides"] or {}
            kwargs = agent._build_api_kwargs(messages, tools)
            result = agent.client.responses.create(**kwargs, stream=stream)
            if stream:
                result = _consume_codex_event_stream(result, model=MODEL)
            assert result.service_tier == "default"  # requested != served
    finally:
        agent.client.close()
    print(json.dumps({"surface": surface, "auth": auth, "stream": stream, "requests": requests}))
    ultra, off, default, explicit, extra_body = requests
    assert ultra.pop("service_tier", None) == "ultrafast"
    assert ultra == off == default  # exact body including tool history/schema/cache prefix
    assert explicit.pop("service_tier") == "default" and explicit == off
    assert extra_body.pop("service_tier") == "default" and extra_body == off
    assert any(item["type"] == "function_call_output" for item in off["input"])
    assert off["tools"][0]["name"] == "lookup"


@pytest.mark.parametrize("model,region", [("us." + MODEL, "us-east-1"), ("global." + MODEL, "us-east-1"),
                                         (MODEL, "us-west-2"), ("openai.gpt-6-astra", "us-east-1")])
def test_config_unsupported_routes_do_not_gain_tier_or_change_transport(tmp_path, monkeypatch, model, region):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent
    cfg = _configure(tmp_path, model=model, region=region)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bedrock-bearer")
    runtime = resolve_runtime_provider(requested="bedrock", target_model=model)
    expected = "bedrock_converse" if model.startswith(("us.", "global.")) else "codex_responses"
    assert runtime["api_mode"] == expected
    for surface in ("cli", "gateway"):
        overrides = _turn_route(cfg, runtime, surface)["request_overrides"]
        assert not overrides
        agent = AIAgent(model=model, provider="bedrock", api_mode=runtime["api_mode"],
                        base_url=runtime["base_url"], api_key=runtime["api_key"],
                        service_tier="ultrafast", request_overrides=overrides,
                        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[])
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hello"}], [])
        assert "service_tier" not in kwargs and "serviceTier" not in kwargs
        assert agent.api_mode == expected
        if expected == "codex_responses":
            agent.client.close()


def test_cli_explicit_ultrafast_is_available_on_supported_mantle_route(monkeypatch):
    from cli import HermesCLI
    shell = SimpleNamespace(model=MODEL, provider="bedrock", base_url=BASE, api_key="test-bedrock-bearer",
                            service_tier=None, agent=None)
    shell._fast_command_available = lambda: HermesCLI._fast_command_available(shell)
    monkeypatch.setattr("hermes_cli.cli_commands_mixin._retire_agent", lambda cli: None)
    HermesCLI._handle_fast_command(shell, "/fast ultrafast")
    assert shell.service_tier == "ultrafast"


def test_fast_capability_is_reported_only_on_the_mantle_route():
    """``/fast`` and the ACP speed picker ask ``model_supports_fast_mode``: Sol is fast only where the
    request builder sends the tier, never by model name alone."""
    from cli import HermesCLI
    assert model_supports_fast_mode(MODEL, provider="bedrock", base_url=BASE)
    for provider, base in ((None, None), ("bedrock", None), ("bedrock", BASE.replace("us-east-1", "us-west-2")),
                           ("openai", "https://api.openai.com/v1"), ("openrouter", BASE)):
        assert not model_supports_fast_mode(MODEL, provider=provider, base_url=base), (provider, base)
        shell = SimpleNamespace(model=MODEL, provider=provider, base_url=base, agent=None)
        assert not HermesCLI._fast_command_available(shell), (provider, base)
