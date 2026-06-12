from __future__ import annotations

from unittest.mock import patch


def test_cursor_provider_registries() -> None:
    from providers import get_provider_profile
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_MODELS
    from hermes_cli.providers import get_label, get_provider, normalize_provider

    profile = get_provider_profile("cursor")
    assert profile is not None
    assert profile.name == "cursor"
    assert profile.auth_type == "external_process"
    assert profile.base_url == "cursor://agent"
    assert "composer-2.5" in profile.fallback_models

    overlay = get_provider("cursor")
    assert overlay is not None
    assert overlay.auth_type == "external_process"
    assert overlay.base_url == "cursor://agent"
    assert normalize_provider("cursor-agent") == "cursor"
    assert normalize_provider("anysphere") == "cursor"
    assert get_label("cursor") == "Cursor"

    assert "cursor" in PROVIDER_REGISTRY
    assert any(entry.slug == "cursor" for entry in CANONICAL_PROVIDERS)
    assert "composer-2.5" in _PROVIDER_MODELS["cursor"]


def test_cursor_external_process_status_and_runtime_resolution(monkeypatch) -> None:
    from hermes_cli.auth import (
        get_external_process_provider_status,
        resolve_external_process_provider_credentials,
    )
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    with patch("shutil.which", return_value="/fake/bin/cursor-agent"), patch(
        "subprocess.check_output", return_value="✓ Logged in as alice@example.com\n"
    ):
        status = get_external_process_provider_status("cursor")
        assert status["configured"] is True
        assert status["logged_in"] is True
        assert status["email"] == "alice@example.com"

        creds = resolve_external_process_provider_credentials("cursor")
        assert creds["provider"] == "cursor"
        assert creds["base_url"] == "cursor://agent"
        assert creds["command"] == "/fake/bin/cursor-agent"
        assert creds["api_key"] == "cursor-agent-login"

        runtime = resolve_runtime_provider(requested="cursor")
        assert runtime["provider"] == "cursor"
        assert runtime["api_mode"] == "chat_completions"
        assert runtime["base_url"] == "cursor://agent"
        assert runtime["command"] == "/fake/bin/cursor-agent"


def test_cursor_provider_model_ids_falls_back_when_cli_missing() -> None:
    from hermes_cli.models import provider_model_ids

    with patch("shutil.which", return_value=None):
        models = provider_model_ids("cursor")
    assert "auto" in models
    assert "composer-2.5" in models
