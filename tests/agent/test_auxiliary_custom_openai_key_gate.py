"""#91308: ``resolve_provider_client("custom", explicit_base_url=...)`` must not send OPENAI_API_KEY
to an unrelated host, nor let it shadow the host-matched main key.

Precedence: explicit api_key → main key on the same origin → OPENAI_API_KEY (OpenAI hosts or the
paired OPENAI_BASE_URL only) → ``no-key-required``.
"""

import pytest

ENV_KEY = "sk-openai-env-secret"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", ENV_KEY)


def _key_for(base_url):
    from agent.auxiliary_client import resolve_provider_client
    client, _model = resolve_provider_client("custom", model="test-model", explicit_base_url=base_url)
    assert client is not None, base_url
    return client.api_key


@pytest.mark.parametrize("base_url", [
    "https://api.fireworks.ai/inference/v1",
    "http://127.0.0.1:8000/v1",
    "https://api.openai.com.evil.test/v1",      # suffix spoof
    "https://openai.azure.com.evil.test/v1",    # azure suffix spoof
    "https://evilopenai.com/v1",                # no dot boundary
    "https://api.openai.com@evil.test/v1",      # userinfo trick
])
def test_env_key_not_sent_to_unrelated_host(base_url):
    assert _key_for(base_url) == "no-key-required"


@pytest.mark.parametrize("base_url", [
    "https://api.openai.com/v1",
    "https://my-resource.openai.azure.com/openai/deployments/x",
])
def test_env_key_still_used_on_openai_hosts(base_url):
    assert _key_for(base_url) == ENV_KEY


def test_env_key_used_on_its_paired_openai_base_url(monkeypatch):
    """OPENAI_BASE_URL names the proxy the key was issued for — same rule as the main resolver."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm-proxy.corp.test/v1")
    assert _key_for("https://llm-proxy.corp.test/v1") == ENV_KEY
    assert _key_for("https://other.corp.test/v1") == "no-key-required"


def test_same_origin_main_key_wins_over_env_key():
    from agent.auxiliary_client import _RUNTIME_MAIN_CONTEXT, set_runtime_main
    token = set_runtime_main("custom", "m", base_url="https://api.fireworks.ai/inference/v1",
                             api_key="fw-configured-key")
    try:
        assert _key_for("https://api.fireworks.ai/inference/v1") == "fw-configured-key"
    finally:
        _RUNTIME_MAIN_CONTEXT.reset(token)


def test_explicit_key_wins():
    from agent.auxiliary_client import resolve_provider_client
    client, _ = resolve_provider_client("custom", model="m", explicit_base_url="https://api.openai.com/v1",
                                        explicit_api_key="sk-explicit")
    assert client.api_key == "sk-explicit"
