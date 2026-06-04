import base64
import json
import time
from types import SimpleNamespace

from agent.agent_init import _merge_custom_provider_extra_body
from hermes_cli import runtime_provider as rp


def _encode_jwt_part(payload):
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _nous_invoke_jwt():
    return ".".join(
        [
            _encode_jwt_part({"alg": "none", "typ": "JWT"}),
            _encode_jwt_part(
                {
                    "scope": rp.auth_mod.NOUS_INFERENCE_INVOKE_SCOPE,
                    "exp": int(time.time()) + 3600,
                }
            ),
            "signature",
        ]
    )


def test_custom_provider_extra_body_merges_into_request_overrides():
    agent = SimpleNamespace(
        provider="custom",
        model="google/gemma-4-31b-it",
        base_url="https://example.test/v1",
        request_overrides={"service_tier": "priority"},
    )

    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "name": "gemma",
                "base_url": "https://example.test/v1/",
                "model": "google/gemma-4-31b-it",
                "extra_body": {
                    "enable_thinking": True,
                    "reasoning_effort": "high",
                },
            }
        ],
    )

    assert agent.request_overrides == {
        "service_tier": "priority",
        "extra_body": {
            "enable_thinking": True,
            "reasoning_effort": "high",
        },
    }


def test_custom_provider_extra_body_preserves_caller_override():
    agent = SimpleNamespace(
        provider="custom",
        model="google/gemma-4-31b-it",
        base_url="https://example.test/v1",
        request_overrides={
            "extra_body": {
                "reasoning_effort": "low",
                "caller_only": True,
            }
        },
    )

    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "name": "gemma",
                "base_url": "https://example.test/v1",
                "model": "google/gemma-4-31b-it",
                "extra_body": {
                    "enable_thinking": True,
                    "reasoning_effort": "high",
                },
            }
        ],
    )

    assert agent.request_overrides["extra_body"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
        "caller_only": True,
    }


def test_custom_provider_extra_body_ignores_other_custom_models():
    agent = SimpleNamespace(
        provider="custom",
        model="other-model",
        base_url="https://example.test/v1",
        request_overrides={},
    )

    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "name": "gemma",
                "base_url": "https://example.test/v1",
                "model": "google/gemma-4-31b-it",
                "extra_body": {"enable_thinking": True},
            }
        ],
    )

    assert agent.request_overrides == {}


def test_named_custom_provider_key_wins_over_nous_invoke_jwt(monkeypatch):
    nous_jwt = _nous_invoke_jwt()
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None)
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "third-party",
                    "base_url": "https://api.thirdparty.example/v1",
                    "api_key": "custom-provider-key",
                }
            ]
        },
    )

    resolved = rp.resolve_runtime_provider(
        requested="custom:third-party",
        explicit_api_key=nous_jwt,
    )

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "https://api.thirdparty.example/v1"
    assert resolved["api_key"] == "custom-provider-key"


def test_bare_custom_model_key_wins_over_nous_invoke_jwt(monkeypatch):
    nous_jwt = _nous_invoke_jwt()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CUSTOM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None)
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "model": {
                "provider": "custom",
                "base_url": "https://api.thirdparty.example/v1",
                "api_key": "model-custom-key",
            }
        },
    )

    resolved = rp.resolve_runtime_provider(
        requested="custom",
        explicit_api_key=nous_jwt,
    )

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "https://api.thirdparty.example/v1"
    assert resolved["api_key"] == "model-custom-key"


def test_named_custom_provider_pool_key_wins_over_nous_invoke_jwt(monkeypatch):
    nous_jwt = _nous_invoke_jwt()
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "pooled",
                    "base_url": "https://api.pooled.example/v1",
                    "api_key": "config-key",
                }
            ]
        },
    )

    def fake_pool_result(base_url, provider_label, api_mode_override=None, provider_name=None):
        return {
            "provider": provider_label,
            "api_mode": api_mode_override or "chat_completions",
            "base_url": base_url,
            "api_key": "pool-key",
            "source": f"pool:custom:{provider_name}",
        }

    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", fake_pool_result)

    resolved = rp.resolve_runtime_provider(
        requested="custom:pooled",
        explicit_api_key=nous_jwt,
    )

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "https://api.pooled.example/v1"
    assert resolved["api_key"] == "pool-key"


def test_custom_provider_without_key_does_not_substitute_nous_jwt(monkeypatch):
    nous_jwt = _nous_invoke_jwt()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None)
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "third-party",
                    "base_url": "https://api.thirdparty.example/v1",
                }
            ]
        },
    )

    resolved = rp.resolve_runtime_provider(
        requested="custom:third-party",
        explicit_api_key=nous_jwt,
    )

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "https://api.thirdparty.example/v1"
    assert resolved["api_key"] == "no-key-required"


def test_custom_provider_nous_host_can_use_nous_invoke_jwt(monkeypatch):
    nous_jwt = _nous_invoke_jwt()
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None)
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "nous-compatible",
                    "base_url": "https://inference-api.nousresearch.com/v1",
                    "api_key": "custom-provider-key",
                }
            ]
        },
    )

    resolved = rp.resolve_runtime_provider(
        requested="custom:nous-compatible",
        explicit_api_key=nous_jwt,
    )

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "https://inference-api.nousresearch.com/v1"
    assert resolved["api_key"] == nous_jwt


def test_nous_provider_still_resolves_nous_invoke_jwt(monkeypatch):
    nous_jwt = _nous_invoke_jwt()
    monkeypatch.setattr(rp, "load_config", lambda: {})
    monkeypatch.setattr(
        rp,
        "load_pool",
        lambda provider: type("Pool", (), {"has_credentials": lambda self: False})(),
    )
    monkeypatch.setattr(
        rp,
        "resolve_nous_runtime_credentials",
        lambda **kwargs: {
            "base_url": "https://inference-api.nousresearch.com/v1",
            "api_key": nous_jwt,
            "source": "portal",
            "expires_at": None,
        },
    )

    resolved = rp.resolve_runtime_provider(requested="nous")

    assert resolved["provider"] == "nous"
    assert resolved["base_url"] == "https://inference-api.nousresearch.com/v1"
    assert resolved["api_key"] == nous_jwt
