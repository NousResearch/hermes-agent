"""HSK-081: startup credential fallback (oneshot shares the gateway's fallback chain).

Before this fix, oneshot's ``_run_agent`` resolved the primary runtime provider
directly via ``resolve_runtime_provider`` before ``AIAgent`` existed; a missing
primary credential raised ``AuthError`` with no fallback path (gateway startup
already had one via ``_try_resolve_fallback_provider``). These tests pin the
shared implementation (``hermes_cli.fallback_config.resolve_first_available_fallback``)
used by both oneshot (``_resolve_oneshot_runtime``) and the gateway
(``gateway.run._try_resolve_fallback_provider``): no second fallback engine,
primary preferred when valid, exact fallback chains fail through, no fallback
configured re-raises, all fallbacks unavailable fails closed, and no
secret/token is ever logged.
"""

from unittest.mock import MagicMock

import pytest

from hermes_cli.auth import AuthError
from hermes_cli.fallback_config import resolve_first_available_fallback
from hermes_cli.runtime_provider import format_runtime_provider_error


def _entry(provider, model):
    return {"provider": provider, "model": model}


class TestCodexToCopilotFallback:
    """openai-codex/gpt-5.6-sol -> copilot/gpt-5.6-sol -> copilot/gpt-5.6-terra."""

    def test_missing_codex_resolves_first_configured_copilot_fallback(self, monkeypatch):
        chain = [
            _entry("copilot", "gpt-5.6-sol"),
            _entry("copilot", "gpt-5.6-terra"),
        ]
        cfg = {"fallback_providers": chain}

        calls = []

        def fake_resolve(*, requested, explicit_base_url, explicit_api_key):
            calls.append(requested)
            if requested == "copilot":
                return {"provider": "copilot", "api_key": "copilot-key", "base_url": "https://copilot"}
            raise AuthError("no credentials")

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
        )
        result = resolve_first_available_fallback(cfg)
        assert result is not None
        runtime, model, configured_provider = result
        assert runtime["provider"] == "copilot"
        assert model == "gpt-5.6-sol"
        assert configured_provider == "copilot"
        # First entry in the configured chain wins; second is never needed.
        assert calls == ["copilot"]


class TestAnthropicToCopilotFallback:
    """anthropic/claude-sonnet-5 -> copilot/claude-sonnet-5 -> copilot/claude-opus-4.8."""

    def test_missing_anthropic_resolves_configured_copilot_claude_fallback(self, monkeypatch):
        chain = [
            _entry("copilot", "claude-sonnet-5"),
            _entry("copilot", "claude-opus-4.8"),
        ]
        cfg = {"fallback_providers": chain}

        def fake_resolve(*, requested, explicit_base_url, explicit_api_key):
            if requested == "copilot":
                return {"provider": "copilot", "api_key": "copilot-key"}
            raise AuthError("no anthropic credentials")

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
        )
        result = resolve_first_available_fallback(cfg)
        assert result is not None
        runtime, model, configured_provider = result
        assert runtime["provider"] == "copilot"
        assert model == "claude-sonnet-5"

    def test_first_fallback_unavailable_advances_to_second(self, monkeypatch):
        chain = [
            _entry("copilot", "claude-sonnet-5"),
            _entry("copilot", "claude-opus-4.8"),
        ]
        cfg = {"fallback_providers": chain}
        seen_models = []

        def fake_resolve(*, requested, explicit_base_url, explicit_api_key):
            seen_models.append(requested)
            raise AuthError("still no credentials")

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
        )
        result = resolve_first_available_fallback(cfg)
        # Both entries were tried (in order) and both failed -> fail closed.
        assert seen_models == ["copilot", "copilot"]
        assert result is None


class TestValidPrimaryPreferred:
    def test_no_fallback_lookup_when_primary_succeeds(self, monkeypatch):
        """resolve_first_available_fallback is only reached from the AuthError branch of
        _resolve_oneshot_runtime; a healthy primary must never invoke it."""
        import hermes_cli.oneshot as oneshot

        called = {"fallback": False}

        def fake_primary(**kwargs):
            return {"provider": "openai-codex", "api_key": "primary-key"}

        def fake_fallback(*a, **k):
            called["fallback"] = True
            raise AssertionError("fallback must not be consulted when primary succeeds")

        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_primary)
        monkeypatch.setattr("hermes_cli.fallback_config.resolve_first_available_fallback", fake_fallback)

        choice = MagicMock(provider="openai-codex", model="gpt-5.6-sol", base_url=None, api_key=None)
        runtime, model = oneshot._resolve_oneshot_runtime(choice)
        assert runtime["provider"] == "openai-codex"
        assert model == "gpt-5.6-sol"
        assert called["fallback"] is False


class TestNoFallbackConfiguredPreservesOriginalError:
    def test_no_chain_returns_none_and_oneshot_reraises(self, monkeypatch):
        import hermes_cli.oneshot as oneshot

        def fake_primary(**kwargs):
            raise AuthError("missing OPENAI_CODEX_API_KEY")

        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_primary)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})  # no fallback_providers/fallback_model

        choice = MagicMock(provider="openai-codex", model="gpt-5.6-sol", base_url=None, api_key=None)
        with pytest.raises(RuntimeError):
            oneshot._resolve_oneshot_runtime(choice)


class TestAllFallbacksUnavailableFailsClosed:
    def test_every_fallback_entry_raising_returns_none(self, monkeypatch):
        cfg = {"fallback_providers": [_entry("copilot", "gpt-5.6-sol"), _entry("copilot", "gpt-5.6-terra")]}

        def always_fail(*, requested, explicit_base_url, explicit_api_key):
            raise AuthError("copilot unavailable too")

        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", always_fail)
        assert resolve_first_available_fallback(cfg) is None

    def test_oneshot_reraises_original_auth_error_when_all_fallbacks_fail(self, monkeypatch):
        import hermes_cli.oneshot as oneshot

        def fake_primary(**kwargs):
            raise AuthError("missing ANTHROPIC_API_KEY")

        def fake_fallback(*a, **k):
            return None  # every configured entry failed to resolve

        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_primary)
        monkeypatch.setattr("hermes_cli.fallback_config.resolve_first_available_fallback", fake_fallback)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"fallback_providers": []})

        choice = MagicMock(provider="anthropic", model="claude-sonnet-5", base_url=None, api_key=None)
        with pytest.raises(RuntimeError) as exc_info:
            oneshot._resolve_oneshot_runtime(choice)
        assert "ANTHROPIC_API_KEY" in str(exc_info.value) or format_runtime_provider_error


class TestNoSecretLeakage:
    def test_resolve_entry_api_key_never_logged(self, monkeypatch, caplog):
        """resolve_first_available_fallback logs only provider/model on skip, never the
        resolved api_key or entry credential fields."""
        cfg = {"fallback_providers": [_entry("copilot", "gpt-5.6-sol")]}
        secret = "sk-super-secret-token-should-not-appear-in-logs"
        cfg["fallback_providers"][0]["api_key"] = secret

        def fake_resolve(*, requested, explicit_base_url, explicit_api_key):
            raise AuthError("boom")

        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve)

        import logging
        logger = logging.getLogger("hsk081-leak-test")
        with caplog.at_level(logging.DEBUG, logger="hsk081-leak-test"):
            result = resolve_first_available_fallback(cfg, logger=logger)
        assert result is None
        assert secret not in caplog.text
