"""Hyphenated provider ids must not be mangled into impossible env-var names (#114405).

``_resolve_call_client`` built its "no API key" hint with ``_explicit.upper()_API_KEY``,
which for ``minimax-oauth`` suggests ``MINIMAX-OAUTH_API_KEY`` — not a valid env var name,
and not the provider's key. The hint must come from
``PROVIDER_REGISTRY[provider_id].api_key_env_vars`` (which #114404 additionally restores
for ``minimax-oauth``). Providers with no registered key env vars (pure OAuth) get no
env-var hint at all.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _keyless_environment(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/nonexistent/hermes-home")
    for key in ("MINIMAX_API_KEY", "MINIMAX_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


def _raise_for_lane(monkeypatch, provider: str, model: str = "m1"):
    """Drive call_llm with a task lane pinned to `provider` and no credentials available."""
    from agent import auxiliary_client as ac

    lane = {"provider": provider, "model": model}
    monkeypatch.setattr(
        ac, "_get_auxiliary_task_config",
        lambda task: dict(lane) if task == "title_generation" else {})
    monkeypatch.setattr(
        ac, "_try_configured_fallback_for_unavailable_client",
        lambda task, provider_id: (None, None, None))
    return ac.call_llm("title_generation", messages=[{"role": "user", "content": "hi"}])


def test_hyphenated_provider_hint_uses_registry_env_var(monkeypatch):
    """minimax-oauth with no credentials → hint names MINIMAX_API_KEY, never MINIMAX-OAUTH_API_KEY."""
    from agent.auxiliary_unavailable import AuxiliaryClientUnavailable

    with pytest.raises(AuxiliaryClientUnavailable) as excinfo:
        _raise_for_lane(monkeypatch, "minimax-oauth", "MiniMax-M2.5")

    message = str(excinfo.value)
    assert "MINIMAX_API_KEY" in message, message
    assert "MINIMAX-OAUTH_API_KEY" not in message, message
    assert "hermes model" in message, message


def test_provider_without_registered_env_vars_gets_no_env_hint(monkeypatch):
    """A provider whose registry entry has no api_key_env_vars must not get a made-up name."""
    from agent.auxiliary_unavailable import AuxiliaryClientUnavailable
    from hermes_cli import auth as auth_mod

    monkeypatch.setitem(
        auth_mod.PROVIDER_REGISTRY, "some-pure-oauth",
        auth_mod.ProviderConfig("some-pure-oauth", "Pure OAuth", "oauth_device_code"))

    with pytest.raises(AuxiliaryClientUnavailable) as excinfo:
        _raise_for_lane(monkeypatch, "some-pure-oauth")

    message = str(excinfo.value)
    assert "SOME-PURE-OAUTH_API_KEY" not in message, message
    assert "Set the" not in message, message
    assert "hermes model" in message, message


def test_existing_api_key_providers_keep_the_hint(monkeypatch):
    """Sanity: a provider with a registered env var still gets the actionable hint."""
    from agent.auxiliary_unavailable import AuxiliaryClientUnavailable

    with pytest.raises(AuxiliaryClientUnavailable) as excinfo:
        _raise_for_lane(monkeypatch, "opencode-zen")

    message = str(excinfo.value)
    assert "OPENCODE_ZEN_API_KEY" in message, message
    assert "hermes model" in message, message
