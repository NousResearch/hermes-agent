"""Regression tests for ``custom_providers[].max_tokens`` propagation.

Covers the fix for #20004 — a per-provider output cap defined in
``custom_providers`` (or the new-style ``providers`` dict) must reach the
gateway's resolved runtime so AIAgent honours it instead of falling back to
the provider transport default.
"""
from __future__ import annotations

from hermes_cli import runtime_provider as rp
from hermes_cli.config import _normalize_custom_provider_entry


class TestNormalizeCustomProviderMaxTokens:
    def test_preserves_positive_int(self):
        normalized = _normalize_custom_provider_entry(
            {
                "name": "ark",
                "base_url": "https://example.invalid/v1",
                "max_tokens": 131072,
            }
        )
        assert normalized is not None
        assert normalized["max_tokens"] == 131072

    def test_drops_zero_or_negative(self):
        for bad in (0, -1, -100):
            normalized = _normalize_custom_provider_entry(
                {
                    "name": "ark",
                    "base_url": "https://example.invalid/v1",
                    "max_tokens": bad,
                }
            )
            assert normalized is not None
            assert "max_tokens" not in normalized, f"value {bad!r} should be rejected"

    def test_drops_non_int(self):
        for bad in ("131072", 1.5, None, [1]):
            normalized = _normalize_custom_provider_entry(
                {
                    "name": "ark",
                    "base_url": "https://example.invalid/v1",
                    "max_tokens": bad,
                }
            )
            assert normalized is not None
            assert "max_tokens" not in normalized, f"value {bad!r} should be rejected"

    def test_does_not_warn_about_max_tokens_as_unknown_key(self, caplog):
        """``max_tokens`` is now in ``_KNOWN_KEYS``; no 'unknown config keys' warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            _normalize_custom_provider_entry(
                {
                    "name": "ark",
                    "base_url": "https://example.invalid/v1",
                    "max_tokens": 8192,
                }
            )
        for record in caplog.records:
            assert "max_tokens" not in record.message or "unknown" not in record.message


class TestRuntimeResolutionPropagatesMaxTokens:
    def test_named_custom_provider_legacy_list_propagates_max_tokens(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setattr(
            rp,
            "load_config",
            lambda: {
                "custom_providers": [
                    {
                        "name": "ark",
                        "base_url": "https://ark.example.invalid/v1",
                        "api_key": "ark-key",
                        "max_tokens": 131072,
                    }
                ]
            },
        )
        monkeypatch.setattr(
            rp,
            "resolve_provider",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("named custom provider should short-circuit resolve_provider")
            ),
        )

        resolved = rp.resolve_runtime_provider(requested="ark")

        assert resolved["provider"] == "custom"
        assert resolved["max_tokens"] == 131072

    def test_named_custom_provider_v12_dict_propagates_max_tokens(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setattr(
            rp,
            "load_config",
            lambda: {
                "providers": {
                    "ark": {
                        "api": "https://ark.example.invalid/v1",
                        "api_key": "ark-key",
                        "default_model": "ark-pro",
                        "max_tokens": 131072,
                    }
                }
            },
        )
        monkeypatch.setattr(
            rp,
            "resolve_provider",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("named custom provider should short-circuit resolve_provider")
            ),
        )

        resolved = rp.resolve_runtime_provider(requested="ark")

        assert resolved["provider"] == "custom"
        assert resolved["max_tokens"] == 131072

    def test_omitted_max_tokens_does_not_appear_in_runtime(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setattr(
            rp,
            "load_config",
            lambda: {
                "custom_providers": [
                    {
                        "name": "ark",
                        "base_url": "https://ark.example.invalid/v1",
                        "api_key": "ark-key",
                    }
                ]
            },
        )
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")

        resolved = rp.resolve_runtime_provider(requested="ark")

        assert resolved["provider"] == "custom"
        assert "max_tokens" not in resolved


class TestGatewayResolveRuntimeAgentKwargsMaxTokens:
    def _isolate_module_state(self, monkeypatch):
        from gateway import run as gateway_run

        return gateway_run

    def test_runtime_max_tokens_wins_over_model_max_tokens(self, monkeypatch):
        gateway_run = self._isolate_module_state(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kwargs: {
                "provider": "custom",
                "api_mode": "chat_completions",
                "base_url": "https://ark.example.invalid/v1",
                "api_key": "ark-key",
                "max_tokens": 131072,
            },
        )
        monkeypatch.setattr(
            gateway_run,
            "_load_gateway_config",
            lambda: {"model": {"max_tokens": 8192}},
        )

        result = gateway_run._resolve_runtime_agent_kwargs()

        assert result["max_tokens"] == 131072

    def test_falls_back_to_model_max_tokens_when_runtime_unset(self, monkeypatch):
        gateway_run = self._isolate_module_state(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kwargs: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "or-key",
            },
        )
        monkeypatch.setattr(
            gateway_run,
            "_load_gateway_config",
            lambda: {"model": {"max_tokens": 8192}},
        )

        result = gateway_run._resolve_runtime_agent_kwargs()

        assert result["max_tokens"] == 8192

    def test_returns_none_when_neither_set(self, monkeypatch):
        gateway_run = self._isolate_module_state(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kwargs: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "or-key",
            },
        )
        monkeypatch.setattr(
            gateway_run,
            "_load_gateway_config",
            lambda: {"model": {}},
        )

        result = gateway_run._resolve_runtime_agent_kwargs()

        assert result["max_tokens"] is None
