"""Credential tiebreak between the openai-api and openai-codex catalogs.

The whole ``gpt-5.6-*`` family lives in both static catalogs, and the
first-catalog-match loop prefers ``openai-api`` purely by dict order. When
``OPENAI_API_KEY`` is missing or a placeholder but a Codex OAuth grant is
stored, detection must route to ``openai-codex`` instead of a guaranteed
auth failure. Every other combination keeps the historical routing.
"""

import pytest

from hermes_cli import auth as _auth_mod
from hermes_cli.models import detect_provider_for_model, detect_static_provider_for_model
import hermes_cli.models as _models_mod


def _patch_store(monkeypatch, store):
    monkeypatch.setattr(_auth_mod, "_load_auth_store", lambda *a, **k: store)


def _with_codex_oauth():
    return {
        "providers": {
            "openai-codex": {
                "tokens": {"access_token": "at-token", "refresh_token": "rt-token"}
            }
        }
    }


class TestCodexTiebreak:
    def test_unkeyed_api_with_codex_oauth_routes_to_codex(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _patch_store(monkeypatch, _with_codex_oauth())
        assert detect_provider_for_model("gpt-5.6-luna", "deepseek") == (
            "openai-codex",
            "gpt-5.6-luna",
        )

    def test_placeholder_api_key_with_codex_oauth_routes_to_codex(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
        _patch_store(monkeypatch, _with_codex_oauth())
        assert detect_provider_for_model("gpt-5.6-terra", "deepseek") == (
            "openai-codex",
            "gpt-5.6-terra",
        )

    def test_usable_api_key_keeps_openai_api_routing(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-123")
        _patch_store(monkeypatch, _with_codex_oauth())
        assert detect_provider_for_model("gpt-5.6-luna", "deepseek") == (
            "openai-api",
            "gpt-5.6-luna",
        )

    def test_no_credentials_keeps_historical_openai_api(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _patch_store(monkeypatch, {})
        assert detect_provider_for_model("gpt-5.6-luna", "deepseek") == (
            "openai-api",
            "gpt-5.6-luna",
        )

    def test_model_outside_codex_catalog_untouched(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _patch_store(monkeypatch, _with_codex_oauth())
        assert detect_provider_for_model("gpt-4.1", "deepseek") == ("openai", "gpt-4.1")

    def test_pool_only_codex_credentials_route_to_codex(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _patch_store(
            monkeypatch,
            {"credential_pool": {"openai-codex": [{"access_token": "pool-token"}]}},
        )
        assert detect_provider_for_model("gpt-5.6-sol", "deepseek") == (
            "openai-codex",
            "gpt-5.6-sol",
        )

    def test_current_provider_codex_short_circuits_before_tiebreak(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _patch_store(monkeypatch, _with_codex_oauth())
        # detect_static_provider_for_model returns None when the model is in
        # the current provider's catalog (step before the tiebreak loop).
        assert detect_static_provider_for_model("gpt-5.6-luna", "openai-codex") is None

    def test_broken_auth_store_does_not_break_detection(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        def _boom(*a, **k):
            raise RuntimeError("store unreadable")

        monkeypatch.setattr(_auth_mod, "_load_auth_store", _boom)
        assert detect_provider_for_model("gpt-5.6-luna", "deepseek") == (
            "openai-api",
            "gpt-5.6-luna",
        )

    def test_short_alias_path_unaffected(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _patch_store(monkeypatch, _with_codex_oauth())
        result = detect_provider_for_model("sonnet", "auto")
        assert result is not None
        assert result[0] == "anthropic"
