"""Env/config api_key runtimes carry the provider credential pool (PR #106833)."""

from types import SimpleNamespace

from hermes_cli import runtime_provider as rp


class TestApiKeyProviderPoolAttachment:
    """Env/config api_key runtimes must carry the provider pool when the resolved key is a
    pool member. Sessions that start while every pool entry is in exhaustion cooldown fall
    through ``_resolve_from_pool`` (``pool.select()`` returns None) onto the bare env key;
    without the pool attached, a 429 can never rotate and the turn dies after max_retries."""

    @staticmethod
    def _pool_with(entries):
        class _Pool:
            def has_credentials(self):
                return True

            provider = "zai"

            def select(self, model=None):
                return None  # every entry in cooldown: forces the env fallback path

            def entries(self):
                return list(entries)

        return _Pool()

    def _resolve_zai(self, monkeypatch, pool):
        monkeypatch.setattr(
            rp,
            "_get_model_config",
            lambda: {"provider": "zai", "default": "glm-5.3"},
        )
        monkeypatch.setattr(rp, "load_pool", lambda _provider: pool)
        monkeypatch.setattr(
            rp,
            "resolve_api_key_provider_credentials",
            lambda _provider: {
                "provider": "zai",
                "api_key": "env-zai-key",
                "base_url": "https://api.z.ai/api/coding/paas/v4",
                "source": "env",
            },
        )
        return rp.resolve_runtime_provider(requested="zai")

    def test_pool_attached_when_env_key_is_pool_member(self, monkeypatch):
        pool = self._pool_with([SimpleNamespace(runtime_api_key="env-zai-key")])
        resolved = self._resolve_zai(monkeypatch, pool)
        assert resolved["provider"] == "zai"
        assert resolved["api_key"] == "env-zai-key"
        assert resolved["credential_pool"] is pool

    def test_foreign_provider_pool_not_attached_despite_matching_env_key(self, monkeypatch):
        pool = self._pool_with([SimpleNamespace(runtime_api_key="env-zai-key")])
        pool.provider = "ollama-cloud"
        resolved = self._resolve_zai(monkeypatch, pool)
        assert resolved["provider"] == "zai"
        assert resolved["api_key"] == "env-zai-key"
        assert resolved["credential_pool"] is None

    def test_pool_not_attached_when_env_key_is_not_pool_member(self, monkeypatch):
        pool = self._pool_with([SimpleNamespace(runtime_api_key="some-other-key")])
        resolved = self._resolve_zai(monkeypatch, pool)
        assert resolved["api_key"] == "env-zai-key"
        assert resolved["credential_pool"] is None

    def test_pool_attach_survives_load_pool_failure(self, monkeypatch):
        monkeypatch.setattr(
            rp,
            "_get_model_config",
            lambda: {"provider": "zai", "default": "glm-5.3"},
        )

        def _boom(_provider):
            raise RuntimeError("auth store unavailable")

        monkeypatch.setattr(rp, "load_pool", _boom)
        monkeypatch.setattr(
            rp,
            "resolve_api_key_provider_credentials",
            lambda _provider: {
                "provider": "zai",
                "api_key": "env-zai-key",
                "base_url": "https://api.z.ai/api/coding/paas/v4",
                "source": "env",
            },
        )
        resolved = rp.resolve_runtime_provider(requested="zai")
        assert resolved["api_key"] == "env-zai-key"
        assert resolved["credential_pool"] is None
