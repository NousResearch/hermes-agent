"""API-key provider endpoints and keys resolve from the same active profile."""
import pytest


@pytest.fixture
def endpoint_provider(monkeypatch):
    from hermes_cli import auth

    config = auth.ProviderConfig(
        id="scoped-endpoint-test", name="Scoped endpoint", auth_type="api_key",
        inference_base_url="https://default.example.invalid/v1",
        api_key_env_vars=("SCOPED_ENDPOINT_TEST_KEY",),
        base_url_env_var="SCOPED_ENDPOINT_TEST_URL",
    )
    monkeypatch.setattr(auth, "_registry_lookup", lambda _: config)
    return config


@pytest.mark.parametrize("scoped_url", ["https://profile.example.invalid/v1", None])
def test_provider_endpoint_uses_bound_scope(monkeypatch, endpoint_provider, scoped_url):
    from agent import secret_scope
    from hermes_cli import auth

    monkeypatch.setenv("SCOPED_ENDPOINT_TEST_URL", "https://launch.example.invalid/v1")
    mapping = {"SCOPED_ENDPOINT_TEST_KEY": "synthetic-profile-key"}
    if scoped_url is not None:
        mapping["SCOPED_ENDPOINT_TEST_URL"] = scoped_url
    mode = secret_scope.set_multiplex_context(True)
    scope = secret_scope.set_secret_scope(mapping)
    try:
        credentials = auth.resolve_api_key_provider_credentials(endpoint_provider.id)
        status = auth.get_api_key_provider_status(endpoint_provider.id)
        assert credentials["api_key"] == "synthetic-profile-key"
        assert credentials["base_url"] == (scoped_url or endpoint_provider.inference_base_url)
        assert status["base_url"] == credentials["base_url"]
    finally:
        secret_scope.reset_secret_scope(scope)
        secret_scope.reset_multiplex_context(mode)


def test_single_profile_endpoint_retains_process_environment(monkeypatch, endpoint_provider):
    from hermes_cli import auth

    monkeypatch.setenv("SCOPED_ENDPOINT_TEST_KEY", "synthetic-process-key")
    monkeypatch.setenv("SCOPED_ENDPOINT_TEST_URL", " https://process.example.invalid/v1/ ")
    credentials = auth.resolve_api_key_provider_credentials(endpoint_provider.id)
    assert credentials["api_key"] == "synthetic-process-key"
    assert credentials["base_url"] == "https://process.example.invalid/v1"


def test_actual_config_endpoint_keeps_precedence(monkeypatch):
    from hermes_cli import auth

    config = auth.ProviderConfig(
        id="actual", name="Actual", auth_type="api_key", base_url_env_var="ACTUAL_BASE_URL",
    )
    monkeypatch.setattr(auth, "read_raw_config", lambda: {
        "model": {"provider": "actual", "base_url": "https://configured.example.invalid/v1"},
    })
    monkeypatch.setenv("ACTUAL_BASE_URL", "https://process.example.invalid/v1")
    assert auth._provider_env_base_url(config) == "https://configured.example.invalid/v1"
