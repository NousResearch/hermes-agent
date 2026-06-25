"""Provider-scoped header regression tests for OpenAI-compatible providers."""

from providers.base import ProviderProfile


def _install_unit_test_provider_profile(monkeypatch) -> ProviderProfile:
    profile = ProviderProfile(
        name="unit-test-profile-provider",
        aliases=("unit-test-profile", "utpp"),
        display_name="Unit Test Profile Provider",
        description="ProviderProfile-only provider used by header tests.",
        signup_url="https://unit-test-provider.example/signup",
        env_vars=("UTPP_API_KEY", "UTPP_BASE_URL"),
        base_url="https://unit-test-provider.example/v1",
        api_mode="anthropic_messages",
        fallback_models=("unit-profile-model",),
    )

    def fake_get_provider_profile(name):
        if name in {profile.name, *profile.aliases}:
            return profile
        return None

    import providers as provider_registry

    monkeypatch.setattr(provider_registry, "get_provider_profile", fake_get_provider_profile)
    return profile


def test_provider_profile_catalog_fetch_receives_provider_scoped_headers(monkeypatch, tmp_path):
    """Live profile catalog fetches should receive matching provider headers."""
    profile = _install_unit_test_provider_profile(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider: {"api_key": "profile-key", "base_url": profile.base_url},
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider_headers": {profile.name: {"X-Provider": "profile"}}}},
    )
    captured = {}

    def fake_fetch_models(*, api_key=None, base_url=None, timeout=8.0, headers=None):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        captured["headers"] = headers
        return ["unit-profile-model"]

    monkeypatch.setattr(profile, "fetch_models", fake_fetch_models)

    from hermes_cli.models import provider_model_ids

    assert provider_model_ids(profile.name, force_refresh=True) == ["unit-profile-model"]
    assert captured["api_key"] == "profile-key"
    assert captured["base_url"] == profile.base_url
    assert captured["headers"]["X-Provider"] == "profile"


def test_provider_profile_catalog_fetch_keeps_legacy_fetch_models_signature(monkeypatch, tmp_path):
    """Out-of-tree ProviderProfile.fetch_models overrides need not accept headers."""
    profile = _install_unit_test_provider_profile(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider: {"api_key": "profile-key", "base_url": profile.base_url},
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider_headers": {profile.name: {"X-Provider": "profile"}}}},
    )
    captured = {}

    def legacy_fetch_models(*, api_key=None, base_url=None, timeout=8.0):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return ["legacy-live-model"]

    monkeypatch.setattr(profile, "fetch_models", legacy_fetch_models)

    from hermes_cli.models import provider_model_ids

    assert provider_model_ids(profile.name, force_refresh=True) == ["legacy-live-model"]
    assert captured == {"api_key": "profile-key", "base_url": profile.base_url}


def test_custom_validation_probe_receives_base_url_inferred_scoped_headers(monkeypatch):
    """Custom endpoint validation should pass provider-scoped headers to /models probes."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider_headers": {"gmi": {"X-Provider": "gmi"}}}},
    )
    captured = {}

    def fake_probe(api_key, base_url, timeout=5.0, api_mode=None, request_headers=None):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        captured["api_mode"] = api_mode
        captured["request_headers"] = request_headers
        return {"models": ["demo-model"], "probed_url": base_url.rstrip("/") + "/models"}

    monkeypatch.setattr("hermes_cli.models.probe_api_models", fake_probe)
    from hermes_cli.models import validate_requested_model

    result = validate_requested_model(
        "demo-model",
        "custom",
        api_key="test-key",
        base_url="https://api.gmi-serving.com/v1",
    )

    assert result["accepted"] is True
    assert captured["request_headers"] == {"X-Provider": "gmi"}


def test_provider_model_cache_invalidates_when_configured_headers_change(monkeypatch, tmp_path):
    """Header-dependent /models caches must refresh when tenant headers change."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"model": {"provider_headers": {"gmi": {"X-Tenant": "tenant-a"}}}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider: {"api_key": "test-key", "base_url": "https://api.gmi-serving.com/v1"},
    )
    captured_headers = []

    def fake_probe(api_key, base_url, timeout=5.0, api_mode=None, request_headers=None):
        headers = dict(request_headers or {})
        captured_headers.append(headers)
        return {
            "models": [f"{headers.get('X-Tenant')}-model"],
            "probed_url": base_url.rstrip("/") + "/models",
        }

    monkeypatch.setattr("hermes_cli.models.probe_api_models", fake_probe)
    from hermes_cli.models import cached_provider_model_ids

    assert cached_provider_model_ids("gmi") == ["tenant-a-model"]

    config["model"]["provider_headers"]["gmi"]["X-Tenant"] = "tenant-b"
    assert cached_provider_model_ids("gmi") == ["tenant-b-model"]
    assert captured_headers == [
        {"X-Tenant": "tenant-a"},
        {"X-Tenant": "tenant-b"},
    ]
