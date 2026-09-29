"""ZAI vision must honor the configured billing endpoint."""

import pytest


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("https://api.z.ai/api/coding/paas/v4", "https://api.z.ai/api/coding/paas/v4"),
        ("https://api.z.ai/api/paas/v4", "https://api.z.ai/api/paas/v4"),
        ("https://api.z.ai/api/anthropic", "https://api.z.ai/api/coding/paas/v4"),
    ],
)
def test_zai_vision_uses_configured_openai_compatible_base_url(
    monkeypatch, tmp_path, configured, expected
):
    """Vision preserves the configured billing surface, converting only its wire path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GLM_API_KEY", "sk-test-coding-plan")
    monkeypatch.setenv("GLM_BASE_URL", configured)

    import agent.auxiliary_client as aux

    # Avoid bootstrapping the host HTTP transport; exercise the real resolver and SDK URL.
    monkeypatch.setattr(aux, "_openai_http_client_kwargs", lambda *_args: {})
    monkeypatch.setattr(aux, "_client_cache", {})
    provider, client, model = aux.resolve_vision_provider_client(
        provider="zai", model="glm-5v-turbo"
    )

    assert provider == "zai"
    assert model == "glm-5v-turbo"
    assert client is not None
    assert str(client.base_url).rstrip("/") == expected
