"""Declared custom endpoint credentials outrank matching-host globals (#57547)."""

import pytest

import hermes_cli.runtime_provider as rp


@pytest.mark.parametrize("binding", ["key_env", "api_key_env"])
@pytest.mark.parametrize("present", [True, False], ids=["bound-key", "missing-key"])
def test_custom_openrouter_endpoint_preserves_declared_credential(monkeypatch, binding, present):
    monkeypatch.setattr(rp, "resolve_provider", lambda *args, **kwargs: "openrouter")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "custom", "base_url": "https://openrouter.ai/api/v1",
        binding: "TENANT_OPENROUTER_KEY",
    })
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-unrelated-global")
    if present:
        monkeypatch.setenv("TENANT_OPENROUTER_KEY", "sk-or-declared-tenant")
        resolved = rp.resolve_runtime_provider(requested="custom")
        assert resolved["api_key"] == "sk-or-declared-tenant"
        assert resolved["provider"] == "custom"
    else:
        monkeypatch.delenv("TENANT_OPENROUTER_KEY", raising=False)
        with pytest.raises(rp.AuthError, match="TENANT_OPENROUTER_KEY") as error:
            rp.resolve_runtime_provider(requested="custom")
        assert error.value.code == "declared_key_env_unresolved"


@pytest.mark.parametrize("endpoint_override", [None, "https://my-vllm-server.example.com/v1"])
def test_custom_endpoint_uses_model_key_env(monkeypatch, endpoint_override):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "custom",
            "base_url": "https://my-vllm-server.example.com/v1",
            "key_env": "MY_VLLM_KEY",
        },
    )
    monkeypatch.setenv("MY_VLLM_KEY", "sk-vllm-env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-wrong-key")

    resolved = rp.resolve_runtime_provider(requested="custom", explicit_base_url=endpoint_override)

    assert resolved["api_key"] == "sk-vllm-env"


@pytest.mark.parametrize("endpoint_override", [None, "https://api.deepseek.com/v1"])
def test_custom_endpoint_missing_declared_key_env_fails_closed(monkeypatch, endpoint_override):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "custom",
            "base_url": "https://api.deepseek.com/v1",
            "key_env": "TENANT_B_API_KEY",
        },
    )
    monkeypatch.delenv("TENANT_B_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-wrong-tenant")

    with pytest.raises(rp.AuthError, match="TENANT_B_API_KEY"):
        rp.resolve_runtime_provider(requested="custom", explicit_base_url=endpoint_override)


def test_named_custom_missing_declared_key_env_fails_closed(monkeypatch):
    monkeypatch.setattr(
        rp,
        "_get_named_custom_provider",
        lambda _name: {
            "name": "tenant-b",
            "base_url": "https://api.deepseek.com/v1",
            "key_env": "TENANT_B_API_KEY",
        },
    )
    monkeypatch.delenv("TENANT_B_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-wrong-tenant")

    with pytest.raises(rp.AuthError, match="TENANT_B_API_KEY"):
        rp._resolve_named_custom_runtime(requested_provider="tenant-b")


def test_configured_key_env_resolving_empty_is_refused(monkeypatch, caplog):
    """#67453: a declared ``key_env`` whose variable is unset used to be laundered silently into
    ``no-key-required`` and surface only as the provider's 403; a keyless block stays silent."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "custom")
    monkeypatch.setattr(rp, "load_config", lambda: {"custom_providers": [
        {"name": "scw", "base_url": "https://api.example.test/v1", "key_env": "UNSET_LLM_KEY", "model": "m"},
        {"name": "local", "base_url": "http://127.0.0.1:8080/v1", "model": "m"}]})
    monkeypatch.delenv("UNSET_LLM_KEY", raising=False)
    with caplog.at_level("WARNING", logger="hermes_cli.runtime_provider"):
        with pytest.raises(rp.AuthError, match="UNSET_LLM_KEY") as exc:
            rp.resolve_runtime_provider(requested="custom:scw")
        assert exc.value.code == "declared_key_env_unresolved"
        assert rp.resolve_runtime_provider(requested="custom:local")["api_key"] == "no-key-required"
    assert not caplog.records
