"""Tests for canonical provider route identity helpers."""

from types import SimpleNamespace

from providers import route_identity


def _profile(base_url):
    return SimpleNamespace(base_url=base_url)


def test_foreign_provider_endpoint_detects_another_registered_route(monkeypatch):
    monkeypatch.setattr(
        "providers.registry.get_provider_profile",
        lambda provider: _profile("https://chatgpt.com/backend-api/codex")
        if provider == "openai-codex"
        else None,
    )
    monkeypatch.setattr(
        "providers.registry.list_providers",
        lambda: [
            _profile("https://chatgpt.com/backend-api/codex"),
            _profile("https://inference-api.nousresearch.com/v1"),
        ],
    )

    assert route_identity.is_foreign_provider_endpoint(
        "openai-codex",
        "https://inference-api.nousresearch.com/v1/",
    )


def test_foreign_provider_endpoint_accepts_own_route(monkeypatch):
    monkeypatch.setattr(
        "providers.registry.get_provider_profile",
        lambda _provider: _profile("https://chatgpt.com/backend-api/codex"),
    )
    monkeypatch.setattr(
        "providers.registry.list_providers",
        lambda: [_profile("https://chatgpt.com/backend-api/codex")],
    )

    assert not route_identity.is_foreign_provider_endpoint(
        "openai-codex",
        "https://chatgpt.com/backend-api/codex/",
    )


def test_foreign_provider_endpoint_ignores_unregistered_custom_route(monkeypatch):
    monkeypatch.setattr(
        "providers.registry.get_provider_profile",
        lambda _provider: None,
    )
    monkeypatch.setattr(
        "providers.registry.list_providers",
        lambda: [_profile("https://api.example.test/v1")],
    )

    assert not route_identity.is_foreign_provider_endpoint(
        "custom:proxy",
        "https://api.example.test/v1",
    )
