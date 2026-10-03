"""Regression tests for runtime-footer producer/consumer wiring."""

from pathlib import Path
from types import SimpleNamespace

import gateway.runtime_footer_usage as usage
import hermes_constants
from gateway.run_turn_runner import _resolve_runtime_footer_metadata


def test_runner_resolves_footer_usage_inside_runtime_scope(monkeypatch):
    calls = []
    monkeypatch.setattr(
        usage,
        "get_cached",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "snapshot",
    )
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: Path("/profiles/routed"))

    result = _resolve_runtime_footer_metadata(
        SimpleNamespace(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_key="runtime-secret",
        ),
        {"display": {"runtime_footer": {"enabled": True, "fields": ["account"]}}},
        "feishu",
    )

    assert result["provider"] == "anthropic"
    assert result["base_url"] == "https://api.anthropic.com"
    assert result["account_usage"] == "snapshot"
    assert result["footer_config"]["fields"] == ["account"]
    assert "api_key" not in result
    assert calls == [
        (
            ("anthropic",),
            {
                "base_url": "https://api.anthropic.com",
                "api_key": "runtime-secret",
                "hermes_home": "/profiles/routed",
            },
        )
    ]


def test_runner_does_not_fetch_when_footer_usage_is_not_requested(monkeypatch):
    monkeypatch.setattr(
        usage,
        "get_cached",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected refresh")),
    )

    result = _resolve_runtime_footer_metadata(
        SimpleNamespace(provider="anthropic", base_url="", api_key="secret"),
        {"display": {"runtime_footer": {"enabled": True, "fields": ["model"]}}},
        "feishu",
    )

    assert result["account_usage"] is None
    assert "api_key" not in result
