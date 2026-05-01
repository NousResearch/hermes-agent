"""Tests for acp_adapter.auth — provider detection."""

from acp_adapter.auth import has_provider, detect_provider


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
