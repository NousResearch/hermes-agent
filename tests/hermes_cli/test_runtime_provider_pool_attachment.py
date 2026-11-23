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

    def test_providers_entry_url_never_gets_the_builtin_pool(self, monkeypatch):
        pool = self._pool_with([SimpleNamespace(runtime_api_key="entry-key")])
        monkeypatch.setattr(rp, "_owned_url_credential", lambda *_a: (True, "entry-key"))
        resolved = self._resolve_zai(monkeypatch, pool)
        assert resolved["source"] == "config:providers"
        assert resolved["api_key"] == "entry-key"
        assert resolved["credential_pool"] is None  # rotating would send another account's keys to that URL


class _FencedPool:
    """Every entry in cooldown: ``select()`` is None, so the ladder lands on the env-key rung."""

    def __init__(self, provider, keys):
        self.provider = provider
        self._entries = [SimpleNamespace(runtime_api_key=k) for k in keys]

    def has_credentials(self):
        return True

    def select(self, model=None):
        return None

    def entries(self):
        return list(self._entries)


class TestAnthropicEnvPoolAttachment:
    def _resolve(self, monkeypatch, pool):
        monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "anthropic", "default": "claude-opus-4-7"})
        monkeypatch.setattr(rp, "load_pool", lambda _provider: pool)
        monkeypatch.setattr(rp, "_anthropic_token_or_raise", lambda **_kw: "env-anthropic-key")
        return rp.resolve_runtime_provider(requested="anthropic")

    def test_pool_attached_when_env_key_is_pool_member(self, monkeypatch):
        pool = _FencedPool("anthropic", ["env-anthropic-key"])
        resolved = self._resolve(monkeypatch, pool)
        assert resolved["api_key"] == "env-anthropic-key"
        assert resolved["credential_pool"] is pool

    def test_pool_not_attached_when_key_is_not_pool_member(self, monkeypatch):
        resolved = self._resolve(monkeypatch, _FencedPool("anthropic", ["other"]))
        assert resolved["credential_pool"] is None

    def test_foreign_provider_pool_not_attached(self, monkeypatch):
        resolved = self._resolve(monkeypatch, _FencedPool("zai", ["env-anthropic-key"]))
        assert resolved["credential_pool"] is None

    def test_load_pool_failure_keeps_bare_runtime(self, monkeypatch):
        def _boom(_provider):
            raise RuntimeError("auth store unavailable")

        monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "anthropic", "default": "claude-opus-4-7"})
        monkeypatch.setattr(rp, "load_pool", _boom)
        monkeypatch.setattr(rp, "_anthropic_token_or_raise", lambda **_kw: "env-anthropic-key")
        resolved = rp.resolve_runtime_provider(requested="anthropic")
        assert resolved["api_key"] == "env-anthropic-key"
        assert resolved["credential_pool"] is None


class TestOpenRouterEnvPoolAttachment:
    def _resolve(self, monkeypatch, pool, *, cfg=None, explicit_api_key=None):
        monkeypatch.setattr(rp, "_get_model_config", lambda: cfg or {"provider": "openrouter", "default": "x/y"})
        monkeypatch.setattr(rp, "load_pool", lambda _provider: pool)
        for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENROUTER_BASE_URL", "CUSTOM_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")
        return rp.resolve_runtime_provider(requested="openrouter", explicit_api_key=explicit_api_key)

    def test_pool_attached_when_env_key_is_pool_member(self, monkeypatch):
        pool = _FencedPool("openrouter", ["sk-or-env-key"])
        resolved = self._resolve(monkeypatch, pool)
        assert resolved["provider"] == "openrouter"
        assert resolved["api_key"] == "sk-or-env-key"
        assert resolved["credential_pool"] is pool

    def test_pool_not_attached_when_key_is_not_pool_member(self, monkeypatch):
        resolved = self._resolve(monkeypatch, _FencedPool("openrouter", ["other"]))
        assert resolved["credential_pool"] is None

    def test_pool_not_attached_for_a_mirror_base_url(self, monkeypatch):
        cfg = {"provider": "openrouter", "default": "x/y", "base_url": "https://mirror.example/v1"}
        resolved = self._resolve(monkeypatch, _FencedPool("openrouter", ["sk-or-env-key"]), cfg=cfg)
        assert resolved["api_key"] == "sk-or-env-key"
        assert resolved["credential_pool"] is None

    def test_explicit_key_never_gets_the_pool(self, monkeypatch):
        resolved = self._resolve(monkeypatch, _FencedPool("openrouter", ["sk-or-explicit"]), explicit_api_key="sk-or-explicit")
        assert resolved["credential_pool"] is None
