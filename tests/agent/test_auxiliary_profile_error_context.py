"""Auxiliary recovery preserves model context for third-party error classifiers."""

from types import SimpleNamespace


def test_model_specific_provider_policy_stops_the_actual_auxiliary_ladder(monkeypatch):
    import providers
    from agent.auxiliary_client import _aux_recovery_ladder

    provider, model = "fixture-model-policy", "restricted-model"
    endpoint = "https://provider.invalid/v1"
    observed_models = []

    def classify(error, *, model="", **context):
        observed_models.append(model)
        if model == "restricted-model":
            return {"reason": "provider_policy_blocked", "retryable": False,
                    "should_rotate_credential": False, "should_fallback": False}
        return None

    providers._discover_providers()
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    providers.register_provider(providers.ProviderProfile(
        name=provider, auth_type="api_key", base_url=endpoint, classify_api_error=classify,
    ))
    error = RuntimeError("Unsupported parameter: temperature")
    error.status_code = 400
    client = SimpleNamespace(_hermes_aux_effective_provider=provider, base_url=endpoint)
    ladder = _aux_recovery_ladder(
        error, client=client, kwargs={"model": model, "temperature": 0.5}, task="compression",
        async_mode=False, base_info=endpoint, resolved_provider=provider, resolved_model=model,
        resolved_base_url=endpoint, resolved_api_key="fixture-key", resolved_api_mode="chat_completions",
        final_model=model, max_tokens=None, main_runtime=None, route_info=None,
    )
    caught, recovery_step = None, None
    try:
        recovery_step = next(ladder)
    except RuntimeError as actual:
        caught = actual
    finally:
        ladder.close()

    assert observed_models == [model]
    assert caught is error
    assert recovery_step is None
