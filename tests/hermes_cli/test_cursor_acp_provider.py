"""Tests for Cursor ACP provider registration and runtime resolution."""

from unittest.mock import patch

from agent.cursor_acp_client import CursorACPClient, _normalize_cursor_model
from hermes_cli import runtime_provider as rp
from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.models import provider_model_ids


def test_cursor_acp_model_catalog_contains_composer_models():
    models = provider_model_ids("cursor-acp")

    assert models == [
        "cursor/composer-2.5",
        "cursor/composer-2",
        "cursor/default",
        "cursor-acp",
    ]


def test_cursor_acp_normalizes_provider_prefixed_model_for_agent_cli():
    assert _normalize_cursor_model("cursor/composer-2.5") == "composer-2.5"
    assert _normalize_cursor_model("composer-2.5") == "composer-2.5"
    assert _normalize_cursor_model("cursor/default") is None
    assert _normalize_cursor_model("cursor-acp") is None


def test_cursor_acp_client_passes_configured_model_to_agent_cli():
    client = CursorACPClient(
        command="agent",
        args=["acp"],
        acp_model="cursor/composer-2.5",
    )

    assert client._acp_args == ["--model", "composer-2.5", "acp"]


def test_cursor_alias_resolves_to_cursor_acp(monkeypatch):
    monkeypatch.setattr(rp.auth_mod, "_load_auth_store", lambda: {})
    assert rp.resolve_provider("cursor") == "cursor-acp"
    assert rp.resolve_provider("cursor-agent") == "cursor-acp"


def test_resolve_runtime_provider_cursor_acp(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "cursor-acp")
    monkeypatch.setattr(
        rp,
        "resolve_external_process_provider_credentials",
        lambda provider: {
            "provider": provider,
            "api_key": "cursor-acp",
            "base_url": "acp://cursor",
            "command": "/usr/local/bin/agent",
            "args": ["acp"],
            "source": "process",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="cursor-acp")

    assert resolved["provider"] == "cursor-acp"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "acp://cursor"
    assert resolved["api_key"] == "cursor-acp"
    assert resolved["command"] == "/usr/local/bin/agent"
    assert resolved["args"] == ["acp"]
    assert resolved["source"] == "process"
    assert resolved["requested_provider"] == "cursor-acp"


def test_cursor_acp_picker_shows_static_catalog_when_cli_available():
    with patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("hermes_cli.auth.get_external_process_provider_status", return_value={"configured": True}), \
         patch("shutil.which", return_value="/usr/local/bin/agent"):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    cursor = next((p for p in providers if p["slug"] == "cursor-acp"), None)

    assert cursor is not None
    assert cursor["models"] == [
        "cursor/composer-2.5",
        "cursor/composer-2",
        "cursor/default",
        "cursor-acp",
    ]
    assert cursor["total_models"] == 4
