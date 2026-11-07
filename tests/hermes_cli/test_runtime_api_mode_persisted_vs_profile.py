"""A persisted ``model.api_mode`` never overrides a provider profile's per-model wire at runtime
resolution (#94881): setup stamps the wire of whichever model was picked first, so a pinned
``chat_completions`` must not send a Responses-only model to /chat/completions. The hook's None
still keeps the persisted value."""

from types import SimpleNamespace

import pytest

HOST = "https://api.perwire.test/v1"


@pytest.fixture
def per_wire(monkeypatch):
    import hermes_cli.auth
    import hermes_cli.runtime_provider as rp
    import providers
    from providers.base import ProviderProfile

    class PerWire(ProviderProfile):
        def api_mode_for_model(self, model, *, api_key=None):
            if model.startswith("broken"):
                raise RuntimeError("live catalog unavailable")
            return "codex_responses" if model.startswith("gpt-5") else None

    providers.list_providers()  # discovery first, so the fake survives it
    providers.register_provider(PerWire(
        name="perwire", base_url=HOST, api_mode="chat_completions", env_vars=("PERWIRE_API_KEY",)))
    monkeypatch.setenv("PERWIRE_API_KEY", "sk-fake")
    monkeypatch.setattr(rp, "load_pool", lambda _provider: SimpleNamespace(has_credentials=lambda: False))
    yield rp
    providers._REGISTRY.pop("perwire", None)
    providers._SOURCES.pop("perwire", None)
    providers._PROVIDER_LIST_CACHE = None
    hermes_cli.auth.PROVIDER_REGISTRY.pop("perwire", None)


def _pin(rp, monkeypatch, api_mode):
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "perwire", "default": "claude-y", "api_mode": api_mode})


@pytest.mark.parametrize(("persisted", "model", "api_mode"), [
    ("chat_completions", "gpt-5.6-sol", "codex_responses"),  # the hook's wire beats the pinned one
    ("codex_responses", "claude-y", "codex_responses"),      # hook None keeps the persisted value
    ("anthropic_messages", "claude-y", "anthropic_messages"),
    ("codex_responses", "broken-z", "chat_completions"),     # a raising hook degrades to the declared wire
])
def test_persisted_api_mode_yields_to_the_per_model_wire(per_wire, monkeypatch, persisted, model, api_mode):
    _pin(per_wire, monkeypatch, persisted)

    assert per_wire.resolve_runtime_provider(requested="perwire", target_model=model)["api_mode"] == api_mode


def test_runtime_and_model_switch_agree_on_a_pinned_provider(per_wire, monkeypatch):
    """``/model`` already ignores the persisted wire for per-model providers; runtime resolution must match."""
    from hermes_cli.model_switch import model_derived_api_mode

    _pin(per_wire, monkeypatch, "chat_completions")

    resolved = per_wire.resolve_runtime_provider(requested="perwire", target_model="gpt-5.5")["api_mode"]
    assert model_derived_api_mode("perwire", "gpt-5.5", "sk-fake") == resolved == "codex_responses"


def test_pool_credentials_follow_the_per_model_wire(per_wire, monkeypatch):
    entry = SimpleNamespace(runtime_api_key="pool-key", access_token="pool-key", base_url=HOST, source="manual")
    pool = SimpleNamespace(has_credentials=lambda: True, select=lambda **_kwargs: entry)
    monkeypatch.setattr(per_wire, "load_pool", lambda _provider: pool)
    _pin(per_wire, monkeypatch, "chat_completions")

    resolved = per_wire.resolve_runtime_provider(requested="perwire", target_model="gpt-5.6-sol")

    assert resolved["api_key"] == "pool-key"
    assert resolved["credential_pool"] is pool
    assert resolved["api_mode"] == "codex_responses"
