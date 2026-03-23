"""Tests for custom provider preservation (fixes #2562 and #2281)."""
import pytest
import hermes_cli.runtime_provider as rp


class TestResolveProviderCustom:
    """auth.resolve_provider must return 'custom' for custom, not 'openrouter'."""

    def test_resolve_provider_custom_returns_custom(self):
        from hermes_cli.auth import resolve_provider
        result = resolve_provider("custom")
        assert result == "custom", f"Expected 'custom', got '{result}'"

    def test_resolve_provider_openrouter_still_returns_openrouter(self):
        from hermes_cli.auth import resolve_provider
        result = resolve_provider("openrouter")
        assert result == "openrouter"

    def test_resolve_provider_auto_does_not_crash(self):
        from hermes_cli.auth import resolve_provider
        # auto may return various things depending on env, just should not raise
        try:
            resolve_provider("auto")
        except Exception as e:
            if "No authentication" in str(e):
                pass  # expected when no keys configured
            else:
                raise


class TestRuntimeProviderCustom:
    """Runtime resolution must preserve custom provider and base_url."""

    def test_named_custom_provider_returns_custom_not_openrouter(self, monkeypatch):
        """Named custom provider resolved via load_config must have provider='custom'."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setattr(
            rp,
            "load_config",
            lambda: {
                "custom_providers": [
                    {
                        "name": "MyLocal",
                        "base_url": "http://localhost:8080/v1",
                        "api_key": "test-key",
                    }
                ]
            },
        )
        monkeypatch.setattr(
            rp,
            "resolve_provider",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError(
                    "resolve_provider should not be called for named custom providers"
                )
            ),
        )

        result = rp._resolve_named_custom_runtime(
            requested_provider="mylocal",
            explicit_api_key=None,
            explicit_base_url=None,
        )
        assert result is not None, "Expected named custom provider to resolve"
        assert result["provider"] == "custom", \
            f"Named custom provider should return 'custom', got '{result['provider']}'"
        assert result["base_url"] == "http://localhost:8080/v1"
        assert result["api_key"] == "test-key"

    def test_custom_base_url_not_replaced_with_openrouter(self):
        """End-to-end: custom provider with explicit base_url must not route to openrouter."""
        from hermes_cli.auth import resolve_provider
        provider = resolve_provider("custom")
        assert provider == "custom", \
            f"resolve_provider('custom') returned '{provider}' -- base_url would be lost"

    def test_openrouter_runtime_preserves_custom_provider(self, monkeypatch):
        """_resolve_openrouter_runtime must return 'custom' when requested_provider is 'custom'."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
        monkeypatch.setattr(
            rp,
            "load_config",
            lambda: {
                "model": {
                    "provider": "custom",
                    "base_url": "http://my-server:8080/v1",
                    "api_key": "my-key",
                }
            },
        )

        result = rp._resolve_openrouter_runtime(
            requested_provider="custom",
            explicit_api_key=None,
            explicit_base_url=None,
        )
        assert result["provider"] == "custom", \
            f"Expected provider='custom', got '{result['provider']}'"
        assert result["base_url"] == "http://my-server:8080/v1"
