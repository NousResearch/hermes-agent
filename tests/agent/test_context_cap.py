"""The global context cap is route-independent and never inflates small windows."""

from __future__ import annotations

from agent.context_cap import cap_context_length, configured_context_cap
from agent.model_metadata import get_model_context_length


# Upstream already pins the Kimi Coding endpoint to 1,048,576 for kimi-k3
# (_ENDPOINT_SCOPED_CONTEXT), so this route resolves without network access.
KIMI_CODING = "https://api.kimi.com/coding"


def _config(monkeypatch, model):
    cfg = {"model": model}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
    return cfg


def _native_window(monkeypatch, window):
    """Pin the uncapped resolver so tests never depend on live model metadata."""
    monkeypatch.setattr(
        "agent.model_metadata._get_model_context_length_uncapped",
        lambda *args, **kwargs: window,
    )


def test_large_route_window_is_capped(monkeypatch):
    _config(monkeypatch, {
        "default": "some-other-model",
        "provider": "kimi-coding",
        "context_cap": 350_000,
    })

    assert get_model_context_length(
        "kimi-k3", provider="kimi-coding", base_url=KIMI_CODING,
    ) == 350_000


def test_larger_synthetic_window_is_capped(monkeypatch):
    _config(monkeypatch, {"context_cap": 350_000})
    _native_window(monkeypatch, 2_000_000)

    assert get_model_context_length("any-model") == 350_000


def test_smaller_native_window_is_not_inflated(monkeypatch):
    _config(monkeypatch, {"context_cap": 350_000})
    _native_window(monkeypatch, 200_000)

    assert get_model_context_length("any-model") == 200_000


def test_smaller_explicit_pin_wins_and_larger_pin_is_capped(monkeypatch):
    _config(monkeypatch, {"context_cap": 350_000})

    assert get_model_context_length("any-model", config_context_length=128_000) == 128_000
    assert get_model_context_length("any-model", config_context_length=900_000) == 350_000


def test_no_cap_preserves_route_window(monkeypatch):
    _config(monkeypatch, {})

    assert get_model_context_length(
        "kimi-k3", provider="kimi-coding", base_url=KIMI_CODING,
    ) == 1_048_576


def test_cap_validation_and_disable(monkeypatch):
    for raw in (None, 0, -1, False, "not-a-number"):
        _config(monkeypatch, {"context_cap": raw})
        assert configured_context_cap() is None
        assert cap_context_length(1_048_576) == 1_048_576
