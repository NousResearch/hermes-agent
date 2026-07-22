"""Cursor provider must resolve without relying on plugin auto-extend alone."""

from __future__ import annotations

import pytest

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider
from hermes_cli.providers import get_provider, resolve_provider_full
from hermes_cli import runtime_provider as rp


def test_cursor_is_hardcoded_in_provider_registry():
    assert "cursor" in PROVIDER_REGISTRY
    cfg = PROVIDER_REGISTRY["cursor"]
    assert cfg.id == "cursor"
    assert cfg.inference_base_url == "cursor://agent"
    assert "CURSOR_API_KEY" in cfg.api_key_env_vars
    assert PROVIDER_REGISTRY["cursor-agent"].id == "cursor"
    assert PROVIDER_REGISTRY["cursor-sdk"].id == "cursor"


def test_resolve_provider_accepts_cursor_aliases():
    assert resolve_provider("cursor") == "cursor"
    assert resolve_provider("cursor-agent") == "cursor"
    assert resolve_provider("cursor-sdk") == "cursor"


def test_overlay_and_full_resolve_agree():
    overlay = get_provider("cursor")
    assert overlay is not None
    assert overlay.id == "cursor"
    assert overlay.base_url == "cursor://agent"

    full = resolve_provider_full("cursor")
    assert full is not None
    assert full.id == "cursor"


def test_resolve_runtime_provider_cursor_before_auth_registry(monkeypatch):
    """Early short-circuit must not depend on resolve_provider succeeding."""
    monkeypatch.setenv("CURSOR_API_KEY", "crsr_test_key_for_resolve")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "cursor", "default": "auto"},
    )

    # Even if auth registry somehow lacked cursor, early path still works.
    def _boom(*_a, **_k):
        raise rp.AuthError("Unknown provider 'cursor'.", code="invalid_provider")

    monkeypatch.setattr(rp, "resolve_provider", _boom)

    runtime = rp.resolve_runtime_provider(requested="cursor", target_model="auto")
    assert runtime["provider"] == "cursor"
    assert runtime["base_url"] == "cursor://agent"
    assert runtime["api_key"] == "crsr_test_key_for_resolve"
    assert runtime["api_mode"] == "chat_completions"


def test_resolve_runtime_provider_cursor_rejects_unusable_secret(monkeypatch):
    monkeypatch.setenv("CURSOR_API_KEY", "op://Infrastructure/Cursor API Key/credential")
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value_prefer_dotenv",
        lambda _name: "op://Infrastructure/Cursor API Key/credential",
    )
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "cursor", "default": "auto"},
    )

    with pytest.raises(rp.AuthError, match="No usable CURSOR_API_KEY"):
        rp.resolve_runtime_provider(requested="cursor", target_model="auto")
