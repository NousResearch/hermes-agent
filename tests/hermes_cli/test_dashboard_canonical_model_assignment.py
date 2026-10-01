"""Dashboard main assignment resolves canonically without CLI model-switch semantics."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import application_dashboard_model_selection as selection
from application_model_switch_persistence import apply_model_selection


@pytest.fixture(autouse=True)
def _no_network_validator(monkeypatch):
    monkeypatch.setattr(selection, "validate_model_switch", lambda *_a, **_kw: "")


def test_native_provider_uses_canonical_model_and_wire(monkeypatch):
    seen = []
    def acquire(**kw):
        seen.append(kw)
        return {"provider": "anthropic", "api_mode": "chat_completions",
                "base_url": "https://api.anthropic.com", "api_key": "secret"}
    monkeypatch.setattr(selection, "_acquire", acquire)
    result = selection.select_dashboard_main_model(
        config={"model": {"provider": "openrouter", "default": "legacy"}},
        provider="anthropic", model="anthropic/claude-sonnet-5",
    )
    assert result.target_provider == "anthropic"
    assert result.new_model == "claude-sonnet-5"
    assert result.api_mode == "anthropic_messages"
    assert seen[0]["requested"] == "anthropic"
    assert seen[0]["explicit_api_key"] is None


def test_unknown_provider_rejected_before_acquisition(monkeypatch):
    monkeypatch.setattr(selection, "_acquire", lambda **kw: pytest.fail("acquired unknown"))
    with pytest.raises(ValueError, match="Unknown provider"):
        selection.select_dashboard_main_model(
            config={}, provider="unknown:missing", model="private-model",
        )


def test_named_provider_config_keeps_declared_bare_key(monkeypatch):
    seen = []
    def acquire(**kw):
        seen.append(kw)
        return {"provider": "custom", "base_url": kw["explicit_base_url"],
                "api_mode": "chat_completions"}
    monkeypatch.setattr(selection, "_acquire", acquire)
    config = {"providers": {
        "relay": {"name": "Relay", "base_url": "https://relay.example/v1",
                  "models": ["private-model"], "default_model": "private-model"}
    }}
    result = selection.select_dashboard_main_model(
        config=config, provider="relay", model="private-model",
    )
    assert result.target_provider == "relay"
    assert result.base_url == "https://relay.example/v1"
    assert seen[0]["explicit_base_url"] == "https://relay.example/v1"
    block = apply_model_selection(
        {"default": "old", "provider": "custom", "base_url": "http://old/v1",
         "api_key": "old-secret", "context_length": 40000, "model_slots": {"fast": "keep"}},
        result,
    )
    assert block["model_slots"] == {"fast": "keep"}
    assert "api_key" not in block and "context_length" not in block


def test_custom_submitted_endpoint_overrides_stale_config_without_key_probe(monkeypatch):
    monkeypatch.setattr(selection, "_acquire", lambda **kw: pytest.fail("must use submitted host"))
    result = selection.select_dashboard_main_model(
        config={"model": {"provider": "custom", "default": "old",
                          "base_url": "https://stale.example/v1"}},
        provider="custom", model="local-model",
        base_url="https://current.example/anthropic/v1", api_key="explicit-key",
    )
    assert result.target_provider == "custom"
    assert result.base_url == "https://current.example/anthropic/v1"
    assert result.api_mode == "anthropic_messages"


def test_dashboard_uses_only_canonical_model_and_provider_domains():
    root = Path(__file__).resolve().parents[2]
    modules = set()
    for path in ("application_dashboard_model_selection.py",
                 "application_dashboard_model_detection.py"):
        for node in ast.walk(ast.parse((root / path).read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
    assert "models.selection" in modules
    assert "providers.routing" in modules
    assert "hermes_cli.model_switch" not in modules
    assert "hermes_cli.models" not in modules


def test_main_assignment_owner_never_imports_cli_switch_or_cli_apply():
    source = (Path(__file__).resolve().parents[2] /
              "hermes_cli" / "web_server_config.py").read_text(encoding="utf-8")
    section = source.split("def _validated_main_model_selection(", 1)[1].split(
        "def _normalize_config_for_web(", 1
    )[0]
    assert "hermes_cli.model_switch" not in section
    assert "application_dashboard_model_selection" in section
    assert "application_model_switch_persistence" in section

def test_missing_provider_credentials_return_dashboard_bad_request(monkeypatch):
    from fastapi import HTTPException
    from hermes_cli.auth_constants import AuthError
    from hermes_cli.web_server_config import _validated_main_model_selection

    def missing(**_kw):
        raise AuthError("No Anthropic credentials found")

    monkeypatch.setattr(selection, "_acquire", missing)
    with pytest.raises(HTTPException) as exc:
        _validated_main_model_selection(
            {"model": {"provider": "openrouter"}}, "anthropic", "claude-sonnet-5",
        )
    assert exc.value.status_code == 400
    assert "credentials" in exc.value.detail.lower()


def test_unscoped_secret_failure_is_not_mislabeled_as_bad_user_selection(monkeypatch):
    from agent.secret_scope import UnscopedSecretError

    def unscoped(**_kw):
        raise UnscopedSecretError("profile scope unavailable")

    monkeypatch.setattr(selection, "_acquire", unscoped)
    with pytest.raises(UnscopedSecretError):
        selection.select_dashboard_main_model(
            config={}, provider="anthropic", model="claude-sonnet-5",
        )


def test_explicit_alias_endpoint_keeps_its_host_without_borrowing_alias_key(monkeypatch):
    seen = []

    def acquire(**kw):
        seen.append(kw)
        return {"provider": "openrouter", "base_url": "https://stale.example/v1",
                "api_key": "openrouter-credential"}

    monkeypatch.setattr(selection, "_acquire", acquire)
    cfg = {"model_aliases": {"relay": {
        "model": "private-model", "provider": "custom",
        "base_url": "https://relay.example/v1",
        "api_key": "other-tenant-secret",
    }}}
    result = selection.select_dashboard_main_model(
        config=cfg, provider="openrouter", model="relay",
    )
    assert result.target_provider == "openrouter"
    assert result.new_model == "private-model"
    assert result.base_url == "https://relay.example/v1"
    assert seen[0]["explicit_base_url"] == "https://relay.example/v1"
    assert seen[0]["explicit_api_key"] is None
