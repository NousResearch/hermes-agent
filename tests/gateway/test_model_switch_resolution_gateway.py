"""Gateway-owned /model resolution and request-policy seams."""

import pytest

from application_model_command_request import parse_model_command, resolve_model_persistence
from gateway.model_switch_resolution import resolve_model_switch


def _cheap_enrichment(monkeypatch):
    monkeypatch.setattr("application_model_switch_enrichment.validate_model_switch", lambda *_a, **_k: "")
    monkeypatch.setattr(
        "agent.models_dev.query_model_metadata", lambda *_a, **_k: None
    )
    monkeypatch.setattr("agent.models_dev.get_model_info", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "application_model_switch_enrichment.resolve_native_compaction_capabilities",
        lambda **_k: {},
    )


def test_model_command_parser_owns_scope_and_reasoning_validation():
    request = parse_model_command(
        "gpt-5.5 --provider openrouter --reasoning high --session"
    )
    assert request.target == "gpt-5.5"
    assert request.explicit_provider == "openrouter"
    assert request.reasoning_effort == "high"
    assert request.scope == "session"
    assert request.errors == ()
    assert resolve_model_persistence(
        {"model": {"default": "old", "provider": "openrouter"}},
        request,
    ) is False

    invalid = parse_model_command("gpt-5.5 --once --global")
    assert invalid.errors == ("once_with_global",)


def test_same_custom_route_uses_canonical_selection_without_credential_probe(
    monkeypatch,
):
    _cheap_enrichment(monkeypatch)
    monkeypatch.setattr(
        "gateway.session_model_resolution._resolve_runtime_credentials",
        lambda **_k: (_ for _ in ()).throw(
            AssertionError("credential acquisition should not run")
        ),
    )
    result = resolve_model_switch(
        config={},
        raw_input="new-local-model",
        explicit_provider="",
        current_provider="custom",
        current_model="old-local-model",
        current_base_url="http://127.0.0.1:8000/v1",
        current_api_key="local-key",
    )
    assert result.new_model == "new-local-model"
    assert result.target_provider == "custom"
    assert result.base_url == "http://127.0.0.1:8000/v1"
    assert result.api_key == "local-key"
    assert result.provider_changed is False


def test_named_configured_provider_preserves_request_overrides(monkeypatch):
    _cheap_enrichment(monkeypatch)
    calls = []

    def credentials(**kwargs):
        calls.append(kwargs)
        return {
            "provider": "custom:named-local",
            "base_url": "http://127.0.0.1:4141/v1",
            "api_key": "local-key",
            "api_mode": "codex_responses",
            "runtime_kind": "http",
        }

    monkeypatch.setattr(
        "gateway.session_model_resolution._resolve_runtime_credentials",
        credentials,
    )
    config = {
        "providers": {
            "named-local": {
                "name": "Local",
                "base_url": "http://127.0.0.1:4141/v1",
                "models": ["rotator-code"],
                "extra_body": {"text": {"verbosity": "low"}},
            }
        }
    }
    result = resolve_model_switch(
        config=config,
        raw_input="rotator-code",
        explicit_provider="named-local",
        current_provider="openrouter",
        current_model="old",
        current_base_url="https://openrouter.ai/api/v1",
        current_api_key="old-key",
    )
    assert calls[0]["requested"] == "named-local"
    assert result.target_provider == "custom:named-local"
    assert result.new_model == "rotator-code"
    assert result.request_overrides == {
        "extra_body": {"text": {"verbosity": "low"}}
    }


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, True),
        ({"model": "existing"}, False),
        ({"model": {"default": "old"}}, False),
    ],
)
def test_default_persistence_is_config_owned(config, expected):
    request = parse_model_command("new-model")
    assert resolve_model_persistence(config, request) is expected