"""OAuth plugin cooldowns must never turn into an OpenRouter route."""
import time
from types import SimpleNamespace

import pytest

from providers.base import ProviderProfile


@pytest.fixture
def plugin_pool(tmp_path, monkeypatch):
    import providers
    from hermes_cli import auth
    from hermes_cli.auth_plugin_providers import register_plugin_provider, PLUGIN_MIRRORED_PROVIDERS
    from agent.credential_pool import PooledCredential, load_pool
    from hermes_cli.config import atomic_config_write

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(auth, "PROVIDER_REGISTRY", dict(auth.PROVIDER_REGISTRY))
    monkeypatch.setattr("hermes_cli.runtime_provider.PROVIDER_REGISTRY", auth.PROVIDER_REGISTRY)
    monkeypatch.setattr("hermes_cli.auth_plugin_providers.PLUGIN_MIRRORED_PROVIDERS", set(PLUGIN_MIRRORED_PROVIDERS))

    class FixtureProfile(ProviderProfile):
        def fetch_models(self, **kwargs):
            return ["fixture-a", "fixture-b"]

    profile = FixtureProfile(name="fixture-oauth", aliases=("fixture-login",), auth_type="oauth_external",
                             base_url="https://relay.invalid", fallback_models=("fixture-a", "fixture-b"))
    providers.register_provider(profile)
    register_plugin_provider(profile)
    atomic_config_write(tmp_path / "config.yaml", {"model": {
        "provider": profile.name, "default": "fixture-a", "base_url": profile.base_url}})
    pool = load_pool(profile.name)
    pool.add_entry(PooledCredential(provider=profile.name, id="seat", label="fixture", auth_type="oauth",
                                   priority=0, source="manual:fixture", access_token="fixture-token",
                                   base_url=profile.base_url))
    return profile, pool


@pytest.mark.parametrize("state", ["empty", "exhausted", "model_cooldown"])
def test_unavailable_oauth_plugin_never_resolves_as_openrouter(plugin_pool, state):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.auth import AuthError
    profile, pool = plugin_pool
    if state == "empty":
        from hermes_cli.auth import write_credential_pool
        write_credential_pool(profile.name, [], removed_ids=["seat"])
    elif state == "exhausted":
        pool.mark_exhausted_and_rotate(status_code=402, failure_reason="billing", model="fixture-a")
    else:
        pool.mark_exhausted_and_rotate(status_code=400, failure_reason="model_entitlement", model="fixture-a")
    with pytest.raises(AuthError, match="No available credentials"):
        resolve_runtime_provider(requested="fixture-login", target_model="fixture-a")
    if state == "model_cooldown":
        rt = resolve_runtime_provider(requested=profile.name, target_model="fixture-b")
        assert (rt["provider"], rt["base_url"]) == (profile.name, profile.base_url)


def test_provider_model_throttle_preserves_sibling_route_and_persistence(plugin_pool):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.model_switch import switch_model, persist_model_selection
    from hermes_cli.config import load_config
    profile, pool = plugin_pool
    assert pool.mark_exhausted_and_rotate(
        status_code=402, failure_reason="rate_limit", model="fixture-a",
        error_context={"rate_limit_scope": "model", "reset_at": time.time() + 600}) is None
    assert pool.entries()[0].last_status is None
    assert pool.select(model="fixture-a") is None
    assert pool.select(model="fixture-b") is not None
    rt = resolve_runtime_provider(requested=profile.name, target_model="fixture-b")
    assert (rt["provider"], rt["base_url"]) == (profile.name, profile.base_url)
    result = switch_model(raw_input="fixture-b", current_provider=profile.name, current_model="fixture-a",
                          current_base_url=profile.base_url, current_api_key="fixture-token", is_global=True)
    assert result.success, result.error_message
    assert (result.target_provider, result.base_url) == (profile.name, profile.base_url)
    persist_model_selection(result)
    config = load_config()["model"]
    assert (config["provider"], config["base_url"]) == (profile.name, profile.base_url)


def test_switch_rejects_a_runtime_for_another_provider_before_adopting_its_key(plugin_pool, monkeypatch):
    from hermes_cli import runtime_provider as rp
    from hermes_cli.model_switch import _Switch
    profile, _pool = plugin_pool
    monkeypatch.setattr(rp, "resolve_runtime_provider", lambda **kwargs: {
        "provider": "openrouter", "api_key": "foreign-key", "base_url": "https://openrouter.ai/api/v1"})
    state = _Switch(raw_input="fixture-b", current_provider=profile.name, current_model="fixture-a",
                    current_base_url=profile.base_url, current_api_key="fixture-token", is_global=True,
                    explicit_provider="", user_providers=None, custom_providers=None,
                    target_provider=profile.name, new_model="fixture-b", api_key="fixture-token",
                    base_url=profile.base_url)
    with pytest.raises(rp.AuthError, match="different provider"):
        state.resolve_runtime(requested=profile.name)
    assert (state.api_key, state.base_url) == ("fixture-token", profile.base_url)


def test_turn_passes_classifier_scope_and_reset_to_the_real_pool(plugin_pool, monkeypatch):
    from agent import turn_api_error as handler
    from agent.error_classifier import FailoverReason
    profile, pool = plugin_pool
    until = time.time() + 600
    profile.classify_api_error = lambda error, **kwargs: {
        "reason": "rate_limit", "should_rotate_credential": True,
        "error_context": {"rate_limit_scope": "model", "reset_at": until}}
    monkeypatch.setattr(handler, "recover_before_classification", lambda *args, **kwargs: (False, "system"))

    def recover(agent, error, classified, retry, *, error_context, **kwargs):
        assert classified.reason == FailoverReason.rate_limit
        assert error_context["message"] == "original diagnostic"
        pool.mark_exhausted_and_rotate(status_code=402, model="fixture-a", failure_reason=classified.reason.value,
                                       error_context=error_context)
        return True, False

    monkeypatch.setattr(handler, "recover_after_classification", recover)
    agent = SimpleNamespace(thinking_callback=None, provider=profile.name, model="fixture-a",
                            base_url=profile.base_url, api_key="fixture-token",
                            _extract_api_error_context=lambda error: {"message": "original diagnostic"},
                            _invoke_api_request_error_hook=lambda **kwargs: None)
    error = RuntimeError("upstream model RPM limit")
    error.status_code = 402
    verdict = handler.handle_api_error(
        agent, api_error=error, _retry=None, thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
        system_message="system", active_system_prompt="system", conversation_history=[], approx_tokens=0,
        retry_count=0, max_retries=3, compression_attempts=0, max_compression_attempts=1, api_call_count=1,
        api_request_id="fixture", api_start_time=time.time(), effective_task_id="fixture", turn_id="fixture")
    assert verdict.action == "continue"
    assert pool.entries()[0].model_cooldowns["fixture-a"] == until
    assert pool.select(model="fixture-b") is not None
