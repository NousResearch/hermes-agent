"""Tests for the explicit-provider gate's handling of Hermes-owned OAuth pool sources.

Regression guard: a MiniMax OAuth login (pool source ``oauth``) is a deliberate
Hermes-initiated sign-in, so the explicit-only picker must keep the
``minimax-oauth`` row visible. It was filtered out because the gate only
recognised ``device_code`` / ``loopback_pkce`` / ``hermes_pkce`` / ``manual``
as explicit sources, so ``minimax-oauth`` vanished from Desktop Settings →
Model and from the chat model picker.
"""

import json

import pytest


def _write_config(tmp_path, config: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    import hermes_yaml as yaml
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config))


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def test_minimax_oauth_pool_row_counts_as_explicit(tmp_path, monkeypatch):
    """A MiniMax OAuth pool row is a Hermes-initiated sign-in → explicit."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"model": {"provider": "llamacpp", "default": "Qwen3.8-27B-Q4_0"}})
    _write_auth_store(tmp_path, {
        "version": 1,
        "providers": {
            "minimax-oauth": {
                "provider": "minimax-oauth",
                "access_token": "sk-cp-token",
                "refresh_token": "dfrt-token",
            },
        },
        "active_provider": "nous",
        "credential_pool": {
            "minimax-oauth": [{
                "id": "4a7798",
                "source": "oauth",
                "auth_type": "oauth",
                "access_token": "sk-cp-token",
            }],
        },
    })

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("minimax-oauth") is True


def test_minimax_oauth_row_survives_the_explicit_only_picker(tmp_path, monkeypatch):
    """The whole point: the row must still be listed by an explicit-only payload.

    A provider the user signed into through Hermes must not vanish from the
    picker — that reads as "my login was dropped".
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"model": {"provider": "llamacpp", "default": "Qwen3.8-27B-Q4_0"}})
    _write_auth_store(tmp_path, {
        "version": 1,
        "providers": {
            "minimax-oauth": {
                "provider": "minimax-oauth",
                "access_token": "sk-cp-token",
                "refresh_token": "dfrt-token",
            },
        },
        "active_provider": "nous",
        "credential_pool": {
            "minimax-oauth": [{
                "id": "4a7798",
                "source": "oauth",
                "auth_type": "oauth",
                "access_token": "sk-cp-token",
            }],
        },
    })

    from hermes_cli.inventory import build_models_payload, load_picker_context
    payload = build_models_payload(load_picker_context(), explicit_only=True)
    slugs = [row.get("slug") for row in payload.get("providers") or []]
    assert "minimax-oauth" in slugs


def test_ambient_oauth_source_for_an_unrelated_provider_is_not_explicit(tmp_path, monkeypatch):
    """A generic ``oauth`` source on a provider Hermes never owns stays hidden.

    The fix is provider-scoped: only the sources Hermes itself writes for that
    provider count, so an ambient row cannot smuggle a provider into the picker.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"model": {"provider": "llamacpp", "default": "Qwen3.8-27B-Q4_0"}})
    _write_auth_store(tmp_path, {
        "version": 1,
        "providers": {},
        "active_provider": "nous",
        "credential_pool": {
            "some-provider": [{
                "id": "abc123",
                "source": "oauth",
                "auth_type": "oauth",
                "access_token": "sk-token",
            }],
        },
    })

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("some-provider") is False
