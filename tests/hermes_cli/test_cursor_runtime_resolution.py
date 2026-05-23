"""Runtime resolution tests for the Cursor SDK provider."""

import pytest

from hermes_cli import runtime_provider as rp


def test_cursor_provider_resolves_to_cursor_sdk_runtime(monkeypatch):
    monkeypatch.setenv("CURSOR_API_KEY", "cursor_test_key")
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "cursor")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "cursor", "default": "composer-2.5"},
    )

    resolved = rp.resolve_runtime_provider(requested="cursor")

    assert resolved["provider"] == "cursor"
    assert resolved["api_mode"] == "cursor_sdk_runtime"
    assert resolved["api_key"] == "cursor_test_key"


def test_valid_api_modes_includes_cursor_sdk_runtime():
    assert "cursor_sdk_runtime" in rp._VALID_API_MODES
