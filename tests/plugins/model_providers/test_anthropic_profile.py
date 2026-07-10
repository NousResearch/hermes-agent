"""Tests for the native Anthropic provider profile."""

from __future__ import annotations

import sys

import pytest


class _FakeAnthropicModelsResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.read_sizes: list[int] = []

    def __enter__(self) -> "_FakeAnthropicModelsResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size is None or size < 0:
            return self.body
        return self.body[:size]


@pytest.fixture
def anthropic_profile():
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("anthropic")
    assert profile is not None, "anthropic provider profile must be registered"
    return profile


@pytest.fixture
def anthropic_module(anthropic_profile):
    import model_tools  # noqa: F401

    return sys.modules[anthropic_profile.__class__.__module__]


def test_fetch_models_bounds_response_read(
    monkeypatch,
    anthropic_module,
    anthropic_profile,
):
    response = _FakeAnthropicModelsResponse(
        b'{"data": [{"id": "claude-test"}]}'
    )
    captured = []

    def _urlopen(req, timeout):
        captured.append((req, timeout))
        return response

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    assert anthropic_profile.fetch_models(api_key="sk-test", timeout=2.0) == [
        "claude-test"
    ]
    assert response.read_sizes == [
        anthropic_module._ANTHROPIC_MODELS_RESPONSE_BODY_MAX_BYTES + 1
    ]
    assert captured[0][0].full_url == "https://api.anthropic.com/v1/models"
    assert captured[0][1] == 2.0


def test_fetch_models_rejects_oversized_response(
    monkeypatch,
    anthropic_module,
    anthropic_profile,
):
    response = _FakeAnthropicModelsResponse(
        b"x" * (anthropic_module._ANTHROPIC_MODELS_RESPONSE_BODY_MAX_BYTES + 1)
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: response,
    )

    assert anthropic_profile.fetch_models(api_key="sk-test") is None
    assert response.read_sizes == [
        anthropic_module._ANTHROPIC_MODELS_RESPONSE_BODY_MAX_BYTES + 1
    ]
