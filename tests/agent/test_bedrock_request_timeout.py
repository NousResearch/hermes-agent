"""Bedrock request timeouts reach the clients that serve the call.

``providers.bedrock.request_timeout_seconds`` and ``providers.bedrock.models.<id>.timeout_seconds``
resolved correctly but never reached a client: the boto3 Converse client kept botocore's 60s read
timeout and the AnthropicBedrock SDK client its 900s default. These tests read the real botocore /
httpx client config (dummy credentials, IMDS off) and intercept the botocore API call itself, so no
request leaves the process.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("boto3")
pytest.importorskip("botocore.client")

import boto3
import botocore.client

from agent import bedrock_adapter
from agent.bedrock_adapter import _get_bedrock_runtime_client, invalidate_runtime_client

REGION = "us-east-1"
RUNTIME_URL = f"https://bedrock-runtime.{REGION}.amazonaws.com"
CLAUDE = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"  # model override 300
NOVA = "us.amazon.nova-pro-v1:0"  # no model override -> provider 600
LLAMA = "us.meta.llama4-maverick-17b-instruct-v1:0"  # model override 120
OVERRIDES = {"providers": {"bedrock": {
    "request_timeout_seconds": 600,
    "models": {CLAUDE: {"timeout_seconds": 300}, LLAMA: {"timeout_seconds": 120}},
}}}
BOTOCORE_DEFAULT_TIMEOUT = 60


@pytest.fixture(autouse=True)
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-AMBIENT")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-secret")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    for var in ("AWS_PROFILE", "AWS_SESSION_TOKEN", "AWS_BEARER_TOKEN_BEDROCK"):
        monkeypatch.delenv(var, raising=False)
    bedrock_adapter.reset_client_cache()
    yield
    bedrock_adapter.reset_client_cache()


@pytest.fixture
def config(monkeypatch):
    """Install ``cfg`` as the loaded config (the operator report's offline technique)."""
    def install(cfg):
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
    install({})
    return install


@pytest.fixture
def botocore_calls(monkeypatch):
    """Intercept every botocore API call: record (operation, serving client, its read timeout)."""
    calls = []

    def fake_make_api_call(client, operation_name, params):
        calls.append((operation_name, client, client.meta.config.read_timeout, params.get("modelId")))
        if operation_name == "ConverseStream":
            return {"stream": []}
        return {"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
                "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}
    monkeypatch.setattr(botocore.client.BaseClient, "_make_api_call", fake_make_api_call)
    return calls


def _converse_kwargs(model):
    return {"__bedrock_converse__": True, "__bedrock_region__": REGION, "modelId": model,
            "messages": [{"role": "user", "content": [{"text": "hi"}]}]}


# --- client construction ----------------------------------------------------------------------

def test_unset_config_builds_exactly_the_botocore_default_client(config):
    baseline = boto3.client("bedrock-runtime", region_name=REGION).meta.config

    built = _get_bedrock_runtime_client(REGION, CLAUDE).meta.config

    assert (built.read_timeout, built.connect_timeout) == (BOTOCORE_DEFAULT_TIMEOUT, BOTOCORE_DEFAULT_TIMEOUT)
    assert (built.read_timeout, built.connect_timeout, built.retries, built.max_pool_connections) == (
        baseline.read_timeout, baseline.connect_timeout, baseline.retries, baseline.max_pool_connections)


@pytest.mark.parametrize("model, expected", [(NOVA, 600), (CLAUDE, 300)], ids=["provider", "model-wins"])
def test_configured_timeout_is_the_client_read_timeout_and_connect_stays_bounded(config, model, expected):
    config(OVERRIDES)

    client_config = _get_bedrock_runtime_client(REGION, model).meta.config

    assert client_config.read_timeout == expected
    assert client_config.connect_timeout == BOTOCORE_DEFAULT_TIMEOUT, "a long read budget must not stretch connect"
    assert client_config.retries == boto3.client("bedrock-runtime", region_name=REGION).meta.config.retries


@pytest.mark.parametrize("provider_value, model_value", [("soon", -5), (0, "later"), ("", None)])
def test_invalid_values_behave_like_unset(config, provider_value, model_value):
    config({"providers": {"bedrock": {"request_timeout_seconds": provider_value,
                                      "models": {CLAUDE: {"timeout_seconds": model_value}}}}})

    client = _get_bedrock_runtime_client(REGION, CLAUDE)

    assert client.meta.config.read_timeout == BOTOCORE_DEFAULT_TIMEOUT
    assert client is _get_bedrock_runtime_client(REGION), "invalid must share the unset client, not a new one"


def test_models_with_different_overrides_never_share_a_client_in_one_region(config):
    config(OVERRIDES)

    claude, nova, llama = (_get_bedrock_runtime_client(REGION, m) for m in (CLAUDE, NOVA, LLAMA))

    assert [c.meta.config.read_timeout for c in (claude, nova, llama)] == [300, 600, 120]
    assert len({id(claude), id(nova), id(llama)}) == 3
    # Bounded: same region + same effective timeout reuses one client.
    assert _get_bedrock_runtime_client(REGION, CLAUDE) is claude
    assert _get_bedrock_runtime_client(REGION, "us.amazon.nova-lite-v1:0") is nova  # provider-level 600 too

    config({})  # unset must not collide with any configured slot
    unset = _get_bedrock_runtime_client(REGION, CLAUDE)
    assert unset not in (claude, nova, llama)
    assert unset.meta.config.read_timeout == BOTOCORE_DEFAULT_TIMEOUT


def test_profile_scoped_clients_carry_their_own_profile_timeouts(tmp_path, monkeypatch):
    """A routed profile's clients come from its own config.yaml and own cache slot, per timeout."""
    from agent import secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "profile-home"
    home.mkdir()
    (home / ".env").write_text("AWS_ACCESS_KEY_ID=AKIA-PROFILE\nAWS_SECRET_ACCESS_KEY=profile-secret\n")
    (home / "config.yaml").write_text(
        "providers:\n  bedrock:\n    request_timeout_seconds: 600\n"
        f"    models:\n      {CLAUDE}:\n        timeout_seconds: 300\n")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)

    def in_profile(fn):
        h_tok = set_hermes_home_override(str(home))
        s_tok = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
        try:
            return fn()
        finally:
            secret_scope.reset_secret_scope(s_tok)
            reset_hermes_home_override(h_tok)

    claude, nova, claude_again = in_profile(lambda: (
        _get_bedrock_runtime_client(REGION, CLAUDE), _get_bedrock_runtime_client(REGION, NOVA),
        _get_bedrock_runtime_client(REGION, CLAUDE)))

    assert (claude.meta.config.read_timeout, nova.meta.config.read_timeout) == (300, 600)
    assert claude.meta.config.connect_timeout == nova.meta.config.connect_timeout == BOTOCORE_DEFAULT_TIMEOUT
    assert claude is claude_again and claude is not nova
    assert claude._request_signer._credentials.access_key == "AKIA-PROFILE"
    # The launch context (no override, its own unset config) keeps botocore's default client.
    ambient = _get_bedrock_runtime_client(REGION, CLAUDE)
    assert ambient is not claude and ambient.meta.config.read_timeout == BOTOCORE_DEFAULT_TIMEOUT
    # Stale-pool eviction drops every timeout slot of the profile's region, never the launch context's.
    assert in_profile(lambda: invalidate_runtime_client(REGION)) is True
    assert in_profile(lambda: _get_bedrock_runtime_client(REGION, CLAUDE)) is not claude
    assert _get_bedrock_runtime_client(REGION, CLAUDE) is ambient


# --- dispatch ---------------------------------------------------------------------------------

def test_converse_and_converse_stream_are_served_with_the_requested_models_timeout(config, botocore_calls):
    from agent.chat_completion_helpers import _BedrockStream, _dispatch_nonstreaming_api_request
    config(OVERRIDES)

    agent = SimpleNamespace(api_mode="bedrock_converse", provider="bedrock", model=CLAUDE,
                            _disable_streaming=False, _safe_print=MagicMock())
    _dispatch_nonstreaming_api_request(agent, _converse_kwargs(CLAUDE), make_client=None)
    _BedrockStream(agent, _converse_kwargs(NOVA), on_first_delta=None)._open_stream(_converse_kwargs(NOVA))

    assert [(op, timeout, model) for op, _client, timeout, model in botocore_calls] == [
        ("Converse", 300, CLAUDE), ("ConverseStream", 600, NOVA)]


def test_auxiliary_call_converse_uses_the_models_timeout(config, botocore_calls):
    from agent.bedrock_adapter import call_converse
    config(OVERRIDES)

    call_converse(region=REGION, model=LLAMA, messages=[{"role": "user", "content": "hi"}])

    assert [(op, timeout) for op, _c, timeout, _m in botocore_calls] == [("Converse", 120)]


def _agent():
    from agent.context_compressor import ContextCompressor
    from run_agent import AIAgent
    agent = AIAgent.__new__(AIAgent)
    agent.model, agent.provider = "claude-opus-4.8", "anthropic"
    agent.base_url, agent.api_key, agent.api_mode = "https://api.anthropic.com", "sk-ant", "anthropic_messages"
    agent.client, agent._anthropic_client, agent._client_kwargs = None, MagicMock(), {}
    agent.quiet_mode, agent._config_context_length, agent._primary_runtime = True, None, {}
    agent.max_tokens = None
    agent.context_compressor = ContextCompressor(
        model=agent.model, threshold_percent=0.5, base_url=agent.base_url, api_key="sk-ant",
        provider="anthropic", quiet_mode=True, config_context_length=None,
    )
    return agent


def _dispatch_turn(agent):
    from agent.chat_completion_helpers import _build_bedrock_kwargs, _dispatch_nonstreaming_api_request
    kwargs = _build_bedrock_kwargs(agent, [{"role": "user", "content": "hi"}], None, None)
    _dispatch_nonstreaming_api_request(agent, kwargs, make_client=None)


def test_model_switch_and_fallback_dispatch_with_the_new_models_timeout(config, botocore_calls, monkeypatch):
    from agent.client_lifecycle import _swap_fallback_clients
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **k: 200_000)
    config(OVERRIDES)
    agent = _agent()

    agent.switch_model(LLAMA, "bedrock", api_key="aws-sdk", base_url=RUNTIME_URL, api_mode="bedrock_converse")
    _dispatch_turn(agent)
    # Fallback onto another Bedrock model, bound the way route_binding.bind_route_entry binds it.
    agent.model, agent.provider, agent.base_url, agent.api_mode = NOVA, "bedrock", RUNTIME_URL, "bedrock_converse"
    _swap_fallback_clients(agent, MagicMock(base_url=RUNTIME_URL, api_key="aws-sdk"), "bedrock", NOVA,
                           RUNTIME_URL, "bedrock_converse")
    _dispatch_turn(agent)
    agent.switch_model(LLAMA, "bedrock", api_key="aws-sdk", base_url=RUNTIME_URL, api_mode="bedrock_converse")
    _dispatch_turn(agent)

    assert [(timeout, model) for _op, _c, timeout, model in botocore_calls] == [
        (120, LLAMA), (600, NOVA), (120, LLAMA)]
    first, second, third = (client for _op, client, _t, _m in botocore_calls)
    assert first is third and first is not second


# --- AnthropicBedrock (Claude via the anthropic SDK) ------------------------------------------

def test_anthropic_bedrock_client_takes_the_timeout_and_keeps_default_when_unset():
    pytest.importorskip("anthropic")
    from agent.anthropic_adapter import build_anthropic_bedrock_client

    configured = build_anthropic_bedrock_client(REGION, timeout=300.0).timeout
    unset = build_anthropic_bedrock_client(REGION).timeout

    assert (configured.read, configured.connect) == (300.0, 10.0)
    assert (unset.read, unset.connect) == (900.0, 10.0)


def test_anthropic_bedrock_request_client_follows_the_active_model(config, monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **k: 200_000)
    other_claude = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # provider-level 600
    config(OVERRIDES)
    agent = _agent()

    agent.switch_model(CLAUDE, "bedrock", api_key="aws-sdk", base_url=RUNTIME_URL, api_mode="anthropic_messages")
    shared = agent._anthropic_client.timeout
    first = agent._create_request_anthropic_client(reason="test").timeout
    agent.model = other_claude  # e.g. a fallback onto another Bedrock Claude in the same region
    second = agent._create_request_anthropic_client(reason="test").timeout

    assert (shared.read, first.read, second.read) == (300.0, 300.0, 600.0)
    assert first.connect == second.connect == 10.0
