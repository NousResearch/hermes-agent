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
