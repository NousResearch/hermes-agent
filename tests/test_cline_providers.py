"""Behavior contracts for the native Cline billing routes."""

from types import SimpleNamespace

from agent.transports.chat_completions import ChatCompletionsTransport
from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_MODELS
from providers import get_provider_profile


def test_cline_providers_should_be_first_class_onboarding_choices():
    slugs = {provider.slug for provider in CANONICAL_PROVIDERS}
    assert {"cline-api", "cline-pass"} <= slugs
    assert PROVIDER_REGISTRY["cline-api"].api_key_env_vars == ("CLINE_API_KEY",)
    assert PROVIDER_REGISTRY["cline-pass"].api_key_env_vars == (
        "CLINEPASS_API_KEY",
        "CLINE_API_KEY",
    )


def test_cline_api_should_preserve_provider_model_wire_ids():
    profile = get_provider_profile("cline-api")
    assert profile is not None
    assert profile.fetch_models() == list(_PROVIDER_MODELS["cline-api"])
    assert "zai/glm-5.2" in profile.fetch_models()
    _, top_level = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "xhigh"}
    )
    assert top_level == {"reasoning_effort": "xhigh"}


def test_cline_pass_should_use_static_subscription_catalog_without_discovery():
    profile = get_provider_profile("cline-pass")
    assert profile is not None
    models = profile.fetch_models(api_key="sk-test")
    assert models == list(_PROVIDER_MODELS["cline-pass"])
    assert all(model.startswith("cline-pass/") for model in models)


def test_cline_pass_should_clamp_mimo_effort_but_preserve_other_xhigh():
    profile = get_provider_profile("cline-pass")
    assert profile is not None
    _, mimo = profile.build_api_kwargs_extras(
        model="cline-pass/mimo-v2.5",
        reasoning_config={"enabled": True, "effort": "xhigh"},
    )
    _, glm = profile.build_api_kwargs_extras(
        model="cline-pass/glm-5.2",
        reasoning_config={"enabled": True, "effort": "xhigh"},
    )
    assert mimo == {"reasoning_effort": "high"}
    assert glm == {"reasoning_effort": "xhigh"}


def test_chat_normalization_should_capture_cline_delta_reasoning_from_model_extra():
    message = SimpleNamespace(
        content="done",
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
        refusal=None,
        model_extra={"reasoning": "cline thought"},
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
    )
    normalized = ChatCompletionsTransport.__new__(
        ChatCompletionsTransport
    ).normalize_response(response)
    assert normalized.reasoning == "cline thought"
