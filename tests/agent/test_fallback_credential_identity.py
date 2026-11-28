"""#151321 at the call site: a second account on one endpoint is tried, not skipped.

The predicate itself is covered in ``test_backend_identity.py``; this file pins that
``_should_skip_fallback_candidate`` actually feeds it the two configured API keys, so the
reported symptom (a 429 against account A skipping account B) is gone end to end.
"""

from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as cch

URL = "https://apihub.agnes-ai.com/v1"


def _agent(api_key=None):
    return SimpleNamespace(provider="custom", model="agnes-3.0-flash", base_url=URL, api_key=api_key)


def _entry(api_key=None):
    entry = {"provider": "custom", "model": "agnes-3.0-flash", "base_url": URL}
    if api_key is not None:
        entry["api_key"] = api_key
    return entry


@pytest.fixture(autouse=True)
def _no_unrelated_skips(monkeypatch):
    """Isolate the identity comparison from entitlement/pool/local-usability skips."""
    monkeypatch.setattr("agent.fallback_cooldown._is_entitlement_rejected", lambda *a, **k: False)
    monkeypatch.setattr(cch, "_candidate_pool_exhausted", lambda *a, **k: False)
    monkeypatch.setattr(cch, "_fallback_entry_unavailable_without_network", lambda *a, **k: None)


def _skip(agent, entry):
    return cch._should_skip_fallback_candidate(
        agent, entry, ("custom", "agnes-3.0-flash"), "custom", "agnes-3.0-flash", set())


def test_second_account_on_the_same_endpoint_is_tried():
    assert _skip(_agent("sk-account-a"), _entry("sk-account-b")) is False


def test_the_same_account_is_still_skipped():
    assert _skip(_agent("sk-account-a"), _entry("sk-account-a")) is True


def test_entry_without_a_configured_key_stays_conservative():
    """No static key on the entry = unknown credential = today's answer (skip)."""
    assert _skip(_agent("sk-account-a"), _entry()) is True


def test_callable_primary_key_stays_conservative():
    """Entra ID resolves a callable per call: never fingerprinted, so the skip survives."""
    assert _skip(_agent(lambda: "sk-rotating"), _entry("sk-account-b")) is True
