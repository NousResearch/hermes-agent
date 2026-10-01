"""Contract tests for the canonical provider routing domain."""

from __future__ import annotations

from providers import (
    InvocationRequest,
    ProviderProfile,
    canonicalize_api_mode,
    resolve_invocation_route,
)


def test_canonicalize_api_mode_preserves_unknown_values_and_maps_aliases():
    assert canonicalize_api_mode(" openai-chat ") == "chat_completions"
    assert canonicalize_api_mode("responses") == "codex_responses"
    assert canonicalize_api_mode("anthropic") == "anthropic_messages"
    assert canonicalize_api_mode("vendor_native") == "vendor_native"
    assert canonicalize_api_mode(None) == ""


def test_endpoint_policy_is_exact_and_spoof_resistant():
    cases = {
        "https://us.api.openai.com/v1": "codex_responses",
        "https://eu.api.openai.com/v1": "codex_responses",
        "https://api.openai.com.attacker.test/v1": "chat_completions",
        "https://proxy.test/api.openai.com/v1": "chat_completions",
        "https://api.anthropic.com/v1": "anthropic_messages",
        "https://proxy.test/v1/anthropic": "anthropic_messages",
        "https://proxy.test/v1/anthropic/v1": "anthropic_messages",
        "https://proxy.test/v1/anthropic/messages": "chat_completions",
        "https://api.kimi.com/coding": "anthropic_messages",
        "https://api.kimi.com/coding/v1": "anthropic_messages",
        "https://api.kimi.com/coding-extra": "chat_completions",
        "https://api.kimi.com.attacker.test/coding": "chat_completions",
        "https://bedrock-runtime.us-east-1.amazonaws.com": "bedrock_converse",
        "https://bedrock-runtime.us-east-1.amazonaws.com.attacker.test": "chat_completions",
        "https://api.meta.ai/v1": "codex_responses",
        "https://api.router.com/v1": "codex_responses",
        "https://api.x.ai/v1": "codex_responses",
        "https://api.actual.inc/v1": "chat_completions",
    }

    for base_url, expected in cases.items():
        route = resolve_invocation_route(
            InvocationRequest(provider="custom", model="model", base_url=base_url)
        )
        assert route.api_mode == expected, base_url



def test_actual_route_mandate_overrides_stale_explicit_mode():
    route = resolve_invocation_route(
        InvocationRequest(
            provider="custom",
            model="test-model",
            base_url="https://api.actual.inc/v1",
            explicit_api_mode="codex_responses",
        )
    )
    assert route.api_mode == "chat_completions"
    assert route.source == "provider_mandate"


def test_route_precedence_is_explicit_then_endpoint_then_policy_then_config_then_profile(monkeypatch):
    class PolicyProfile(ProviderProfile):
        def resolve_route_policy(self, model: str, base_url: str = "", *, options=None) -> str | None:
            return "anthropic_messages" if model == "policy-model" else None

    profile = PolicyProfile(
        name="precedence-probe",
        api_mode="codex_responses",
        base_url="https://probe.invalid/v1",
    )
    monkeypatch.setattr("providers.routing._get_profile", lambda _provider: profile)

    def route(**kwargs):
        return resolve_invocation_route(
            InvocationRequest(
                provider="precedence-probe",
                model=kwargs.pop("model", "ordinary-model"),
                **kwargs,
            )
        )

    assert route(configured_api_mode="anthropic", configured_provider="precedence-probe").source == "configured"
    assert route(
        model="policy-model",
        configured_api_mode="anthropic",
        configured_provider="precedence-probe",
    ).source == "provider_policy"
    assert route(
        base_url="https://api.openai.com/v1",
        model="policy-model",
        configured_api_mode="anthropic",
        configured_provider="precedence-probe",
    ).source == "endpoint_mandate"
    assert route(
        explicit_api_mode="openai",
        base_url="https://api.openai.com/v1",
        model="policy-model",
        configured_api_mode="anthropic",
        configured_provider="precedence-probe",
    ).source == "explicit"
    assert route().api_mode == "codex_responses"


def test_generic_gpt5_model_policy_is_owned_by_routing(monkeypatch):
    profiles = {
        "external": ProviderProfile(name="external", auth_type="external_process"),
        "copilot": ProviderProfile(name="copilot", auth_type="copilot"),
    }
    monkeypatch.setattr("providers.routing._get_profile", profiles.get)

    generic = resolve_invocation_route(
        InvocationRequest(
            provider="generic-http",
            model="openai/gpt-5.6",
            base_url="https://proxy.example.invalid/v1",
        )
    )
    assert generic.api_mode == "codex_responses"
    assert generic.source == "model_policy"

    configured = resolve_invocation_route(
        InvocationRequest(
            provider="generic-http",
            model="gpt-5.6",
            base_url="https://proxy.example.invalid/v1",
            configured_provider="generic-http",
            configured_api_mode="chat_completions",
        )
    )
    assert configured.api_mode == "chat_completions"
    assert configured.source == "configured"

    azure = resolve_invocation_route(
        InvocationRequest(
            provider="generic-http",
            model="gpt-5.6",
            base_url="https://resource.openai.azure.com/openai/v1",
        )
    )
    assert azure.api_mode == "chat_completions"

    custom = resolve_invocation_route(
        InvocationRequest(provider="custom", model="gpt-5.6", base_url="https://proxy.invalid/v1")
    )
    assert custom.api_mode == "chat_completions"

    external = resolve_invocation_route(
        InvocationRequest(provider="external", model="gpt-5.6", base_url="https://proxy.invalid/v1")
    )
    assert external.api_mode == "chat_completions"
    assert external.runtime_kind == "external_process"

    external_official_marker = resolve_invocation_route(
        InvocationRequest(provider="external", model="gpt-5.6", base_url="https://api.openai.com/v1")
    )
    assert external_official_marker.api_mode == "chat_completions"
    assert external_official_marker.runtime_kind == "external_process"

    copilot = resolve_invocation_route(
        InvocationRequest(provider="copilot", model="gpt-5-mini")
    )
    assert copilot.api_mode == "chat_completions"


def test_configured_mode_does_not_cross_provider_boundary(monkeypatch):
    profile = ProviderProfile(name="route-provider", api_mode="chat_completions")
    monkeypatch.setattr("providers.routing._get_profile", lambda _provider: profile)

    route = resolve_invocation_route(
        InvocationRequest(
            provider="route-provider",
            model="model",
            configured_provider="other-provider",
            configured_api_mode="anthropic_messages",
        )
    )
    assert route.api_mode == "chat_completions"
    assert route.source == "profile"


def test_runtime_kind_and_aggregator_declaration(monkeypatch):
    class ClientProfile(ProviderProfile):
        def create_client(self, **kwargs):
            return object()

    profiles = {
        "external": ProviderProfile(name="external", auth_type="external_process", base_url="acp://external"),
        "client": ClientProfile(name="client", base_url="https://client.invalid/v1"),
        "aggregator": ProviderProfile(name="aggregator", is_aggregator=True, base_url="https://agg.invalid/v1"),
    }
    monkeypatch.setattr("providers.routing._get_profile", profiles.__getitem__)

    assert resolve_invocation_route(InvocationRequest(provider="external")).runtime_kind == "external_process"
    assert resolve_invocation_route(InvocationRequest(provider="client")).runtime_kind == "provider_client"
    route = resolve_invocation_route(InvocationRequest(provider="aggregator"))
    assert route.runtime_kind == "http"
    assert route.is_routing_aggregator is True


def test_app_server_runtime_is_an_openai_runtime_overlay(monkeypatch):
    profile = ProviderProfile(name="openai-codex", api_mode="codex_responses")
    monkeypatch.setattr("providers.routing._get_profile", lambda _provider: profile)

    route = resolve_invocation_route(
        InvocationRequest(provider="openai-codex", model="gpt-5", openai_runtime="codex_app_server")
    )
    assert route.api_mode == "codex_responses"
    assert route.runtime_kind == "app_server"
    assert route.source in {"provider_policy", "profile"}


def test_historical_openai_runtime_identity_uses_direct_api_profile():
    route = resolve_invocation_route(InvocationRequest(provider="openai", model="gpt-5.6"))
    assert route.provider == "openai"
    assert route.base_url == "https://api.openai.com/v1"
    assert route.api_mode == "codex_responses"


def test_hard_endpoint_mandate_beats_app_server_overlay():
    route = resolve_invocation_route(
        InvocationRequest(
            provider="openai",
            base_url="https://api.anthropic.com/v1",
            openai_runtime="codex_app_server",
        )
    )
    assert route.api_mode == "anthropic_messages"
    assert route.runtime_kind == "http"


def test_named_custom_identity_is_preserved_and_routes_as_aggregator(monkeypatch):
    profile = ProviderProfile(name="custom", base_url="https://custom.invalid/v1")
    monkeypatch.setattr("providers.routing._get_profile", lambda _provider: profile)

    route = resolve_invocation_route(
        InvocationRequest(provider="custom:my-gateway", model="vendor/model")
    )
    assert route.provider == "custom:my-gateway"
    assert route.is_routing_aggregator is True