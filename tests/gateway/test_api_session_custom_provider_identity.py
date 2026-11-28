"""API-server: a resumed turn keeps the named custom endpoint identity.

A named custom endpoint (``model.provider: custom:<name>``) resolves to the bare provider
class ``custom`` with its identity in ``requested_provider``. The session-persisted-model
branch of ``APIServerAdapter._select_agent_runtime`` re-resolved the bare class, which lands
on the default ``https://openrouter.ai/api/v1``. Runs the real method; only the provider
resolver is replaced by a recorder.
"""
from __future__ import annotations

import gateway.platforms.api_server as api

NAMED = "custom:local-runtime"
LOCAL = "http://127.0.0.1:11434/v1"


def _select(monkeypatch, runtime, session_model="model-a:Q5"):
    calls = []

    def resolver(provider, target_model=None):
        calls.append(provider)
        url = LOCAL if provider == NAMED else ("https://openrouter.ai/api/v1" if provider == "custom" else "x")
        return {"provider": "custom" if provider.startswith("custom") else provider, "base_url": url}

    monkeypatch.setattr(api, "_resolve_request_runtime_agent_kwargs", resolver)
    adapter = api.APIServerAdapter.__new__(api.APIServerAdapter)
    adapter._session_model_override_for = lambda _key: None
    adapter._model_name = "model-a:Q5"
    kwargs = dict(runtime)
    adapter._select_agent_runtime(
        kwargs, "model-a:Q5", requested_model=None, requested_provider=None, route=None,
        session_model=session_model, confirmed_runtime_lock=False, gateway_session_key="k", session_id="s")
    return calls, kwargs


def test_resumed_turn_resolves_the_named_custom_identity(monkeypatch):
    calls, kwargs = _select(monkeypatch, {"provider": "custom", "requested_provider": NAMED, "base_url": LOCAL})
    assert calls == [NAMED]
    assert kwargs["base_url"] == LOCAL


def test_bare_custom_without_named_identity_is_unchanged(monkeypatch):
    calls, _kwargs = _select(monkeypatch, {"provider": "custom", "requested_provider": "custom", "base_url": LOCAL})
    assert calls == ["custom"]


def test_other_providers_are_unchanged(monkeypatch):
    calls, _kwargs = _select(monkeypatch, {"provider": "openrouter", "requested_provider": "openrouter"})
    assert calls == ["openrouter"]


def test_first_turn_without_session_model_does_not_re_resolve(monkeypatch):
    calls, kwargs = _select(monkeypatch, {"provider": "custom", "requested_provider": NAMED, "base_url": LOCAL},
                            session_model=None)
    assert calls == [] and kwargs["base_url"] == LOCAL
