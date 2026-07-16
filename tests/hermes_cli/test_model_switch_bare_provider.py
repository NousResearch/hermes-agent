"""Bare `/model <provider>` must switch provider, not treat the slug as a model name."""

from __future__ import annotations

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def test_bare_cursor_acp_switches_provider(monkeypatch):
    """`/model cursor-acp` from a custom Gemma endpoint must flip provider."""
    monkeypatch.setattr(
        "hermes_cli.auth.shutil.which",
        lambda command: f"/usr/local/bin/{command}",
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *a, **k: dict(_MOCK_VALIDATION),
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "cursor-acp",
            "api_mode": "chat_completions",
            "base_url": "acp://cursor",
            "api_key": "cursor-acp",
            "command": "/usr/local/bin/agent",
            "args": ["acp"],
            "source": "process",
        },
    )

    result = switch_model(
        raw_input="cursor-acp",
        current_provider="custom",
        current_model="gemma-4",
        current_base_url="http://213.221.10.157:1234/v1",
        current_api_key="lm-token",
        is_global=True,
    )

    assert result.success is True
    assert result.target_provider == "cursor-acp"
    assert result.new_model in {"agent", "cursor-acp"}
    assert str(result.base_url or "").startswith("acp://cursor")


def test_bare_cursor_alias_switches_provider(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.shutil.which",
        lambda command: f"/usr/local/bin/{command}",
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *a, **k: dict(_MOCK_VALIDATION),
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "cursor-acp",
            "api_mode": "chat_completions",
            "base_url": "acp://cursor",
            "api_key": "cursor-acp",
            "command": "/usr/local/bin/agent",
            "args": ["acp"],
            "source": "process",
        },
    )

    result = switch_model(
        raw_input="cursor",
        current_provider="custom",
        current_model="gemma-4",
        current_base_url="http://127.0.0.1:1234/v1",
        is_global=False,
    )

    assert result.success is True
    assert result.target_provider == "cursor-acp"


def test_provider_flag_without_model_uses_catalog_default(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.shutil.which",
        lambda command: f"/usr/local/bin/{command}",
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *a, **k: dict(_MOCK_VALIDATION),
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "cursor-acp",
            "api_mode": "chat_completions",
            "base_url": "acp://cursor",
            "api_key": "cursor-acp",
            "command": "/usr/local/bin/agent",
            "args": ["acp"],
            "source": "process",
        },
    )

    result = switch_model(
        raw_input="",
        current_provider="custom",
        current_model="gemma-4",
        current_base_url="http://127.0.0.1:1234/v1",
        explicit_provider="cursor-acp",
    )

    assert result.success is True
    assert result.target_provider == "cursor-acp"
    assert result.new_model == "agent"
