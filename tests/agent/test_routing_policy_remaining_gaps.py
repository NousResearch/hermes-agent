"""Regression coverage for terminal policy checks in remaining auxiliary paths."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _policy(**deny):
    return {
        "enabled": True,
        "require_explicit": False,
        "deny": {"providers": [], "models": [], "base_url_hosts": [], **deny},
    }


def _client(calls):
    return SimpleNamespace(
        base_url="https://allowed.invalid/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(choices=[]),
        )),
    )


def test_explicit_openrouter_route_keeps_provider_identity_after_sync_and_async_conversion(monkeypatch):
    """A concrete OpenRouter selection must reach both wire clients as OpenRouter, not blank."""
    from agent import auxiliary_client as aux

    sync = _client([])
    async_client = _client([])
    monkeypatch.setattr(aux, "_try_openrouter", lambda **_kw: (sync, "openrouter/default"))
    monkeypatch.setattr(aux, "_to_async_client", lambda client, model, **_kw: (async_client, model))

    sync_client, _ = aux.resolve_provider_client("openrouter", model="openrouter/model")
    async_route_client, _ = aux.resolve_provider_client("openrouter", model="openrouter/model", async_mode=True)

    assert sync_client._hermes_aux_effective_provider == "openrouter"
    assert async_route_client._hermes_aux_effective_provider == "openrouter"


def test_raw_auxiliary_stream_denied_provider_never_sends(monkeypatch):
    """The Relay stream callback is a real send boundary, including its provider identity."""
    from agent import auxiliary_client as aux
    from hermes_cli import routing_policy

    calls = []
    client = _client(calls)
    client._hermes_aux_effective_provider = "denied-provider"
    monkeypatch.setattr(routing_policy, "current_routing_policy", lambda: _policy(providers=["denied-provider"]))

    with pytest.raises(routing_policy.RoutingPolicyError, match="denies provider"):
        aux._relay_sync_stream(client, {"model": "allowed", "messages": [], "stream": True})

    assert calls == []


def test_codex_adapter_checks_translated_extra_body_model_before_responses_send(monkeypatch):
    """Responses translation must preserve the effective model for its final wire guard."""
    from agent import auxiliary_client as aux
    from hermes_cli import routing_policy

    sent = []
    real_client = SimpleNamespace(
        api_key="k", base_url="https://codex.invalid/v1",
        _hermes_aux_effective_provider="openai-codex",
        responses=SimpleNamespace(create=lambda **kwargs: sent.append(kwargs)),
    )
    adapter = aux._CodexCompletionsAdapter(real_client, "allowed-model")
    monkeypatch.setattr(routing_policy, "current_routing_policy", lambda: _policy(models=["denied-model"]))

    with pytest.raises(routing_policy.RoutingPolicyError, match="selected model"):
        adapter.create(model="allowed-model", messages=[], extra_body={"model": "denied-model"})

    assert sent == []


def test_moa_slot_policy_rejection_is_not_downgraded_to_bare_runtime(monkeypatch):
    """A denied slot resolution aborts before the MoA fan-out can send or continue."""
    from agent import moa_loop
    from hermes_cli.routing_policy import RoutingPolicyError

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kw: (_ for _ in ()).throw(RoutingPolicyError("denied slot")),
    )

    with pytest.raises(RoutingPolicyError, match="denied slot"):
        moa_loop._slot_runtime({"provider": "denied", "model": "denied-model"})


def test_iteration_summary_policy_error_propagates_without_sdk_send(monkeypatch):
    """Late summary effective-model overrides are checked inside Relay's callback."""
    from agent import chat_completion_helpers as helpers
    from hermes_cli import routing_policy

    calls = []
    client = _client(calls)
    transport = SimpleNamespace(normalize_response=lambda response, **_kw: response)
    agent = SimpleNamespace(
        provider="allowed-provider", model="allowed", base_url="https://allowed.invalid/v1",
        api_mode="chat_completions", max_iterations=1, suppress_status_output=True,
        _build_api_kwargs=lambda _messages: {
            "model": "allowed", "messages": [], "extra_body": {"model": "denied-model"},
        },
        _ensure_primary_openai_client=lambda **_kw: client,
        _get_transport=lambda: transport,
        _sanitize_api_messages=lambda messages: messages,
        _drop_thinking_only_and_merge_users=lambda messages: messages,
        _should_sanitize_tool_calls=lambda: False,
        _force_ascii_payload=False,
    )
    monkeypatch.setattr(routing_policy, "current_routing_policy", lambda: _policy(models=["denied-model"]))
    monkeypatch.setattr(helpers, "_iteration_summary_api_messages", lambda _agent, _messages: [])

    with pytest.raises(routing_policy.RoutingPolicyError, match="selected model"):
        helpers.handle_max_iterations(agent, [], 1)

    assert calls == []


def test_background_review_policy_rejection_never_falls_back_to_parent(monkeypatch):
    """A denied routed review must not silently consume the parent model."""
    from agent import background_review
    from hermes_cli.routing_policy import RoutingPolicyError

    parent = SimpleNamespace(
        provider="parent", model="parent-model", max_tokens=None, acp_command=None, acp_args=[],
        request_overrides={}, _credential_pool=None, _current_main_runtime=lambda: {},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kw: (_ for _ in ()).throw(RoutingPolicyError("denied review")),
    )

    with pytest.raises(RoutingPolicyError, match="denied review"):
        background_review._resolve_review_runtime(parent, {"provider": "denied", "model": "denied-model"})


def test_compressor_policy_rejection_never_retries_the_main_route(monkeypatch):
    """A denied summary route is terminal rather than a reason to retry it on main."""
    from agent import context_compressor
    from hermes_cli.routing_policy import RoutingPolicyError

    compressor = context_compressor.ContextCompressor.__new__(context_compressor.ContextCompressor)
    compressor.summary_model = "denied-summary"
    compressor.model = "main-model"
    compressor.provider = "denied-provider"
    compressor.base_url = "https://denied.invalid/v1"
    compressor._fallback_to_main_for_compression = lambda *_args, **_kwargs: pytest.fail("main fallback")
    compressor._generate_summary = lambda *_args, **_kwargs: pytest.fail("main retry")
    monkeypatch.setattr("agent.conversation_compression._raise_if_stale_attempt", lambda _self: None)

    with pytest.raises(RoutingPolicyError, match="denied summary"):
        compressor._on_summary_failure(RoutingPolicyError("denied summary"), [], None, "")


def test_summary_relay_rewrite_to_denied_model_never_sends(monkeypatch):
    """The summary's final callback, not only its pre-Relay request, is a wire boundary."""
    from agent import chat_completion_helpers as helpers
    from hermes_cli import routing_policy

    sent = []
    agent = SimpleNamespace(provider="allowed", model="allowed", base_url="https://allowed.invalid")
    monkeypatch.setattr(routing_policy, "current_routing_policy", lambda: _policy(models=["denied"]))
    monkeypatch.setattr("agent.relay_llm.execute_current", lambda request, callback, **_kw: callback({**request, "model": "denied"}))

    with pytest.raises(routing_policy.RoutingPolicyError, match="selected model"):
        helpers._managed_summary_call(agent, "id", {"model": "allowed"}, lambda request: sent.append(request), retry_count=0)
    assert sent == []


def test_moa_reference_policy_error_is_terminal_before_aggregator(monkeypatch):
    """Denied reference resolution cannot become a degraded advisory note or invoke aggregation."""
    from agent import moa_loop
    from hermes_cli.routing_policy import RoutingPolicyError

    monkeypatch.setattr(moa_loop, "_slot_runtime", lambda _slot: (_ for _ in ()).throw(RoutingPolicyError("denied ref")))
    monkeypatch.setattr(moa_loop, "call_llm", lambda **_kw: pytest.fail("no provider send"))
    with pytest.raises(RoutingPolicyError, match="denied ref"):
        moa_loop.aggregate_moa_context(user_prompt="u", api_messages=[], reference_models=[{"provider": "x", "model": "m"}], aggregator={"provider": "ok", "model": "ok"})


def test_moa_reference_post_resolution_policy_error_is_terminal(monkeypatch):
    """A policy error raised by the actual reference send cannot become a failed note."""
    from agent import moa_loop
    from hermes_cli.routing_policy import RoutingPolicyError

    monkeypatch.setattr(moa_loop, "_slot_runtime", lambda _slot: {"provider": "allowed", "model": "allowed"})
    monkeypatch.setattr(moa_loop, "call_llm", lambda **_kw: (_ for _ in ()).throw(RoutingPolicyError("denied reference wire")))

    with pytest.raises(RoutingPolicyError, match="denied reference wire"):
        moa_loop._run_reference({"provider": "allowed", "model": "allowed"}, [{"role": "user", "content": "u"}])


def test_moa_aggregator_post_resolution_policy_error_is_terminal(monkeypatch):
    """An aggregator wire policy error must not fall back to raw reference guidance."""
    from agent import moa_loop
    from hermes_cli.routing_policy import RoutingPolicyError

    monkeypatch.setattr(moa_loop, "_run_references_parallel", lambda *_a, **_kw: [("ref", "advice", SimpleNamespace())])
    monkeypatch.setattr(moa_loop, "_slot_runtime", lambda _slot: {"provider": "allowed", "model": "allowed"})
    monkeypatch.setattr(moa_loop, "call_llm", lambda **_kw: (_ for _ in ()).throw(RoutingPolicyError("denied aggregator wire")))

    with pytest.raises(RoutingPolicyError, match="denied aggregator wire"):
        moa_loop.aggregate_moa_context(user_prompt="u", api_messages=[], reference_models=[{"provider": "ref", "model": "m"}], aggregator={"provider": "allowed", "model": "allowed"})


@pytest.mark.parametrize("async_mode", [False, True])
def test_gemini_native_wire_model_override_is_denied_without_send(monkeypatch, async_mode):
    """Native Gemini writes the translated top-level model, so it must guard that value."""
    import asyncio
    from agent.gemini_native_adapter import AsyncGeminiNativeClient, GeminiNativeClient
    from hermes_cli import routing_policy

    sent = []
    class RecordingTransport:
        def post(self, *args, **kwargs):
            sent.append((args, kwargs))
            pytest.fail("native Gemini transport must not send")
    monkeypatch.setattr(routing_policy, "current_routing_policy", lambda: _policy(models=["allowed-model"]))
    sync = GeminiNativeClient(api_key="key", http_client=RecordingTransport())
    client = AsyncGeminiNativeClient(sync) if async_mode else sync
    create = client.chat.completions.create
    kwargs = {"model": "allowed-model", "messages": [], "extra_body": {"model": "denied-at-entry"}}

    with pytest.raises(routing_policy.RoutingPolicyError, match="selected model"):
        asyncio.run(create(**kwargs)) if async_mode else create(**kwargs)
    assert sent == []


def test_codex_app_server_fails_closed_under_enabled_policy_before_session_dispatch(monkeypatch):
    """The subprocess route is opaque, so an enabled policy cannot permit its turn or compaction traffic."""
    from agent import codex_runtime
    from hermes_cli import routing_policy

    agent = SimpleNamespace(model="allowed-agent-model", _codex_session=None)
    monkeypatch.setattr(routing_policy, "current_routing_policy", lambda: _policy())

    with pytest.raises(routing_policy.RoutingPolicyError, match="cannot verify"):
        codex_runtime._ensure_codex_session(agent, messages=[])


@pytest.mark.parametrize("provider,model", [(None, None), ("auto", "model"), ("provider", None)])
def test_aux_call_resolution_requires_explicit_route_before_discovery(monkeypatch, provider, model):
    """The direct resolver must not let vision/cache branches discover credentials under strict policy."""
    from agent import auxiliary_client as aux
    from hermes_cli import routing_policy

    monkeypatch.setattr(routing_policy, "current_routing_policy", lambda: {**_policy(), "require_explicit": True})
    with pytest.raises(routing_policy.RoutingPolicyError, match="explicit"):
        aux._resolve_call_client(
            None, provider=provider, model=model, base_url=None, api_key=None,
            resolved_provider="auto", resolved_model=None, resolved_base_url=None,
            resolved_api_key=None, resolved_api_mode=None, main_runtime=None, async_mode=False,
        )


def test_init_fallback_policy_denial_stops_before_second_fallback(monkeypatch):
    """An unavailable primary followed by a denied fallback cannot skip onward to a usable second fallback."""
    from agent import agent_init
    from hermes_cli.routing_policy import RoutingPolicyError

    agent = SimpleNamespace(provider="primary", model="primary-model")
    attempted = []
    monkeypatch.setattr(agent_init, "_fallback_entries", lambda _model: [
        {"provider": "denied", "model": "denied-model"}, {"provider": "usable", "model": "usable-model"},
    ])
    monkeypatch.setattr("hermes_cli.fallback_config.resolve_entry_api_key", lambda _entry: None)
    def resolve(provider, **_kwargs):
        attempted.append(provider)
        if provider == "primary":
            return None, None
        if provider == "denied":
            raise RoutingPolicyError("denied configured fallback")
        pytest.fail("second fallback must not be resolved")
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", resolve)

    with pytest.raises(RoutingPolicyError, match="denied configured fallback"):
        agent_init._routed_client_kwargs(agent, None, None)
    assert attempted == ["primary", "denied"]


def test_bedrock_aux_adapter_checks_extra_body_effective_model_before_send(monkeypatch):
    """Native Bedrock translation must guard the wire model selected through extra_body."""
    from agent import auxiliary_client as aux
    from hermes_cli import routing_policy

    sent = []
    monkeypatch.setattr(routing_policy, "current_routing_policy", lambda: _policy(models=["denied"]))
    monkeypatch.setattr("agent.bedrock_adapter.call_converse", lambda **kwargs: sent.append(kwargs))
    with pytest.raises(routing_policy.RoutingPolicyError, match="selected model"):
        aux._BedrockCompletionsAdapter("us-east-1", "allowed").create(
            model="allowed", messages=[], extra_body={"model": "denied"})
    assert sent == []


def test_runtime_primary_denial_precedes_credential_resolution_and_fallback(monkeypatch):
    """A forbidden explicit primary route cannot probe credentials or enter its fallback chain."""
    from hermes_cli import config, runtime_provider as rp
    from hermes_cli.routing_policy import RoutingPolicyError

    policy = _policy(providers=["gemini"])
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: {"routing_policy": policy})
    monkeypatch.setattr(rp, "resolve_provider", lambda *_a, **_kw: pytest.fail("credential resolution ran"))
    monkeypatch.setattr("hermes_cli.fallback_config.get_fallback_chain", lambda _config: [
        {"provider": "openrouter", "model": "usable-model"},
    ])

    with pytest.raises(RoutingPolicyError, match="denies provider"):
        rp.resolve_runtime_with_fallback(
            {"fallback_providers": [{"provider": "openrouter", "model": "usable-model"}]},
            requested="gemini", target_model="gemini-allowed-model",
        )


class _NoSendGeminiTransport:
    def __init__(self):
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        pytest.fail("Gemini transport must not send under denied provider policy")


@pytest.mark.parametrize("route", ["auto", "custom:gemini-native"])
@pytest.mark.parametrize("async_mode", [False, True])
def test_registry_gemini_identity_survives_auto_and_named_custom_routing(monkeypatch, route, async_mode):
    """Registry discovery must retain Gemini identity through native sync/async wire clients."""
    import asyncio
    from types import SimpleNamespace
    from agent import auxiliary_client as aux
    from agent.gemini_native_adapter import GeminiNativeClient
    from hermes_cli import auth, routing_policy
    from hermes_cli import runtime_provider as rp

    transport = _NoSendGeminiTransport()
    native_base = "https://generativelanguage.googleapis.com/v1beta"
    pconfig = SimpleNamespace(auth_type="api_key", inference_base_url=native_base, name="Gemini", api_key_env_vars=[])
    monkeypatch.setattr(auth, "PROVIDER_REGISTRY", {"gemini": pconfig})
    monkeypatch.setattr(auth, "resolve_api_key_provider_credentials", lambda _provider: {"api_key": "key", "base_url": native_base})
    monkeypatch.setattr(aux, "_get_aux_model_for_provider", lambda provider: "gemini-allowed-model" if provider == "gemini" else None)
    monkeypatch.setattr(aux, "_try_openrouter", lambda **_kw: (None, None))
    monkeypatch.setattr(aux, "_try_nous", lambda **_kw: (None, None))
    monkeypatch.setattr(aux, "_try_custom_endpoint", lambda **_kw: (None, None))
    monkeypatch.setattr(aux, "_discovery_chain_allowed", lambda *_a, **_kw: True)
    monkeypatch.setattr(rp, "_get_named_custom_provider", lambda name: (
        {"name": "gemini-native", "base_url": native_base, "api_key": "key", "model": "gemini-allowed-model"}
        if name == "gemini-native" else None
    ))
    real_init = GeminiNativeClient.__init__
    def recording_init(self, **kwargs):
        kwargs["http_client"] = transport
        real_init(self, **kwargs)
    monkeypatch.setattr(GeminiNativeClient, "__init__", recording_init)
    monkeypatch.setattr(routing_policy, "current_routing_policy", lambda: _policy(providers=["gemini"]))

    client, model = aux.resolve_provider_client(route, model="gemini-allowed-model", async_mode=async_mode)

    assert model == "gemini-allowed-model"
    assert client._hermes_aux_effective_provider == "gemini"
    with pytest.raises(routing_policy.RoutingPolicyError, match="denies provider"):
        result = client.chat.completions.create(model=model, messages=[])
        if async_mode:
            asyncio.run(result)
    assert transport.calls == []


def test_bedrock_final_model_id_denial_precedes_converse_and_stream_sends(monkeypatch):
    """The final translated modelId is checked before either Bedrock boto operation."""
    from agent import bedrock_adapter
    from agent import chat_completion_helpers as helpers
    from hermes_cli import routing_policy

    calls = []
    client = SimpleNamespace(
        converse=lambda **kwargs: calls.append(("converse", kwargs)),
        converse_stream=lambda **kwargs: calls.append(("converse_stream", kwargs)),
    )
    monkeypatch.setattr(bedrock_adapter, "_get_bedrock_runtime_client", lambda _region: client)
    monkeypatch.setattr(bedrock_adapter, "recover_from_cache_point_rejection",
                        lambda *_a, **_kw: pytest.fail("policy denial must not enter cache-point recovery"))
    monkeypatch.setattr(routing_policy, "current_routing_policy", lambda: _policy(models=["forbidden-final-model"]))
    translated = {"__bedrock_region__": "us-east-1", "model": "allowed-earlier-model", "modelId": "forbidden-final-model", "messages": []}

    for stream in (False, True):
        with pytest.raises(routing_policy.RoutingPolicyError, match="selected model"):
            helpers._bedrock_converse_call(dict(translated), stream=stream)
    assert calls == []


def test_bedrock_stream_fallback_checks_final_model_id_before_converse_send(monkeypatch):
    """A stream-denial fallback cannot catch a policy denial and resend via converse."""
    from agent import chat_completion_helpers as helpers
    from hermes_cli import routing_policy

    calls = []
    client = SimpleNamespace(converse=lambda **kwargs: calls.append(kwargs))
    agent = SimpleNamespace(_bedrock_region="us-east-1", _disable_streaming=False, _safe_print=lambda *_a, **_kw: None,
                            model="allowed-earlier-model", provider="bedrock")
    stream = helpers._BedrockStream(agent, {"modelId": "allowed-earlier-model"}, on_first_delta=None)
    monkeypatch.setattr(routing_policy, "current_routing_policy", lambda: _policy(models=["forbidden-final-model"]))

    with pytest.raises(routing_policy.RoutingPolicyError, match="selected model"):
        stream._fall_back_to_converse(client, {"modelId": "forbidden-final-model", "messages": []}, Exception("IAM"))
    assert calls == []
