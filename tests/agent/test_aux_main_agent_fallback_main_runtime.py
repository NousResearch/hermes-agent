"""Regression: the main-agent aux fallback must reuse the main runtime, not an ambient key.

`_try_main_agent_model_fallback` resolved the main provider from the bare name, so a
`custom` main provider (a relay reached via base_url) fell through to
`_resolve_api_key_provider()`, which adopts whatever provider key is in the dotenv —
sending the aux payload (image bytes, prompt text) to a provider the user never
configured. The `elif main_runtime:` reuse arm in `resolve_provider_client` exists for
exactly this, but the kwarg was never passed, so the arm was dead here.
"""
from __future__ import annotations

import pytest

from agent import auxiliary_client as ac


class _FakeClient:
    """Stand-in for a resolved client; records the endpoint it would talk to."""

    def __init__(self, base_url: str = ""):
        self.base_url = base_url


@pytest.fixture
def bare_custom_main(monkeypatch):
    """A bare `custom` main provider with NO custom endpoint env — the leaking shape."""
    monkeypatch.setattr(ac, "_read_main_provider", lambda: "custom")
    monkeypatch.setattr(ac, "_read_main_model", lambda: "chat-model-under-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def test_fallback_reuses_main_runtime_endpoint(bare_custom_main, monkeypatch):
    """With main_runtime supplied, the fallback must resolve the MAIN agent's endpoint."""
    seen = {}

    def _fake_resolve(provider, model, main_runtime=None, **kwargs):
        seen["main_runtime"] = main_runtime
        return _FakeClient(base_url=(main_runtime or {}).get("base_url", "")), model

    monkeypatch.setattr(ac, "resolve_provider_client", _fake_resolve)

    main_runtime = {"base_url": "http://main-relay.invalid/v1", "api_key": "sk-test", "model": "chat-model-under-test"}
    client, model, label = ac._try_main_agent_model_fallback(
        "exhausted-provider", task="vision", reason="error", main_runtime=main_runtime,
    )

    assert seen["main_runtime"] == main_runtime, (
        "the fallback must forward main_runtime so resolve_provider_client can reuse the "
        "main agent's resolved endpoint instead of walking the ambient API-key registry"
    )
    assert client is not None
    assert label == "main-agent(custom)"
    assert model == "chat-model-under-test"


def test_fallback_without_main_runtime_still_resolves(bare_custom_main, monkeypatch):
    """No runtime supplied (legacy callers) must not crash — the call still happens."""
    seen = {}

    def _fake_resolve(provider, model, main_runtime=None, **kwargs):
        seen["main_runtime"] = main_runtime
        return _FakeClient(), model

    monkeypatch.setattr(ac, "resolve_provider_client", _fake_resolve)

    client, model, label = ac._try_main_agent_model_fallback(
        "exhausted-provider", task="vision", reason="error",
    )

    assert seen["main_runtime"] is None
    assert client is not None


def test_fallback_does_not_adopt_an_unrelated_api_key_provider(bare_custom_main, monkeypatch):
    """The leak itself: with a runtime available, a `custom` main provider must NOT be
    re-resolved to whatever provider key the dotenv happens to carry.

    This pins the CONSUMER-visible outcome (which endpoint the aux payload would be sent
    to), not just that a kwarg was passed, so a refactor that keeps the signature but
    stops forwarding the runtime still fails here.
    """
    # An unrelated provider key is present — the condition that made the bug pick Gemini.
    monkeypatch.setenv("GEMINI_API_KEY", "sk-unrelated-test-key")

    def _fake_resolve(provider, model, main_runtime=None, **kwargs):
        if main_runtime and main_runtime.get("base_url"):
            return _FakeClient(base_url=main_runtime["base_url"]), model
        # The bare-custom path: the real resolve_provider_client would walk the API-key
        # registry here and adopt the ambient provider. Mirror that so the test is a
        # faithful stand-in rather than a mock that hides the leak.
        return _FakeClient(base_url="https://generativelanguage.googleapis.com/v1beta"), model

    monkeypatch.setattr(ac, "resolve_provider_client", _fake_resolve)

    client, model, _label = ac._try_main_agent_model_fallback(
        "exhausted-provider", task="vision", reason="error",
        main_runtime={"base_url": "http://main-relay.invalid/v1", "api_key": "sk-test",
                      "model": "chat-model-under-test"},
    )

    assert client.base_url == "http://main-relay.invalid/v1", (
        "aux payload would be sent to an unrelated provider instead of the main agent's "
        "own endpoint"
    )
