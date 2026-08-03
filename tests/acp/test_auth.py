"""Tests for acp_adapter.auth — provider detection."""

from acp_adapter.auth import (
    TERMINAL_SETUP_AUTH_METHOD_ID,
    build_auth_methods,
    has_provider,
    detect_provider,
)


class TestHasProvider:
    def test_has_provider_with_resolved_runtime(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda: {"provider": "openrouter", "api_key": "sk-or-test"},
        )
        assert has_provider() is True

    def test_has_no_provider_when_runtime_has_no_key(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda: {"provider": "openrouter", "api_key": ""},
        )
        assert has_provider() is False

    def test_has_no_provider_when_runtime_resolution_fails(self, monkeypatch):
        def _boom():
            raise RuntimeError("no provider")

        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _boom)
        assert has_provider() is False


class TestDetectProvider:
    def test_detect_openrouter(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda: {"provider": "openrouter", "api_key": "sk-or-test"},
        )
        assert detect_provider() == "openrouter"

    def test_detect_anthropic(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda: {"provider": "anthropic", "api_key": "sk-ant-test"},
        )
        assert detect_provider() == "anthropic"

    def test_detect_none_when_no_key(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda: {"provider": "kimi-coding", "api_key": ""},
        )
        assert detect_provider() is None

    def test_detect_none_on_resolution_error(self, monkeypatch):
        def _boom():
            raise RuntimeError("broken")

        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _boom)
        assert detect_provider() is None

    def test_detect_provider_strips_and_lowercases_provider(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda: {"provider": " OpenRouter ", "api_key": " sk-or-test "},
        )
        assert detect_provider() == "openrouter"


class TestBuildAuthMethods:
    def test_build_auth_methods_returns_provider_and_terminal_when_configured(self, monkeypatch):
        monkeypatch.setattr("acp_adapter.auth.detect_provider", lambda: "openrouter")

        methods = build_auth_methods()
        payloads = [method.model_dump(by_alias=True, exclude_none=True) for method in methods]

        assert payloads[0]["id"] == "openrouter"
        assert payloads[0]["name"] == "openrouter runtime credentials"
        assert any(payload["id"] == TERMINAL_SETUP_AUTH_METHOD_ID for payload in payloads)
        terminal = next(payload for payload in payloads if payload["id"] == TERMINAL_SETUP_AUTH_METHOD_ID)
        assert terminal["type"] == "terminal"
        assert terminal["args"] == ["--setup"]

    def test_build_auth_methods_returns_terminal_setup_when_unconfigured(self, monkeypatch):
        monkeypatch.setattr("acp_adapter.auth.detect_provider", lambda: None)

        methods = build_auth_methods()
        payloads = [method.model_dump(by_alias=True, exclude_none=True) for method in methods]

        assert payloads == [
            {
                "args": ["--setup"],
                "description": (
                    "Open Hermes' interactive model/provider setup in a terminal. "
                    "Use this when Hermes has not been configured on this machine yet."
                ),
                "id": TERMINAL_SETUP_AUTH_METHOD_ID,
                "name": "Configure Hermes provider",
                "type": "terminal",
            }
        ]
