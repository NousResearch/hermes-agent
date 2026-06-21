from gateway import run as gateway_run


def _runner():
    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    runner._session_model_overrides = {}
    runner._last_resolved_model = {}
    return runner


def test_runtime_agent_kwargs_preserves_explicit_provider_model(monkeypatch):
    def fake_resolve_runtime_provider():
        return {
            "api_key": "sk-test",
            "base_url": "https://example.test/v1",
            "provider": "custom-runtime",
            "api_mode": "openai",
            "model": "runtime/model-from-provider",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {"default": ""},
    )

    kwargs = gateway_run._resolve_runtime_agent_kwargs()

    assert kwargs["provider"] == "custom-runtime"
    assert kwargs["model"] == "runtime/model-from-provider"


def test_session_runtime_uses_provider_model_when_config_model_empty(monkeypatch):
    def fake_resolve_runtime_provider():
        return {
            "api_key": "sk-test",
            "base_url": "https://example.test/v1",
            "provider": "custom-runtime",
            "api_mode": "openai",
            "model": "runtime/model-from-provider",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {"default": ""},
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda _cfg=None: "")

    model, runtime_kwargs = _runner()._resolve_session_agent_runtime(session_key="sid")

    assert model == "runtime/model-from-provider"
    assert "model" not in runtime_kwargs
    assert runtime_kwargs["provider"] == "custom-runtime"


def test_consume_runtime_model_leaves_kwargs_safe_for_agent_constructor():
    model, runtime_kwargs = gateway_run._consume_runtime_model(
        "",
        {
            "provider": "custom-runtime",
            "model": "runtime/model-from-provider",
        },
    )

    assert model == "runtime/model-from-provider"
    assert runtime_kwargs == {"provider": "custom-runtime"}


def test_session_runtime_keeps_provider_default_fallback_without_runtime_model(monkeypatch):
    def fake_resolve_runtime_provider():
        return {
            "api_key": "sk-test",
            "base_url": "https://example.test/v1",
            "provider": "custom-runtime",
            "api_mode": "openai",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {"default": ""},
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda _cfg=None: "")
    monkeypatch.setattr(
        "hermes_cli.models.get_default_model_for_provider",
        lambda provider: "catalog/default" if provider == "custom-runtime" else "",
    )

    model, runtime_kwargs = _runner()._resolve_session_agent_runtime(session_key="sid")

    assert model == "catalog/default"
    assert runtime_kwargs["provider"] == "custom-runtime"
