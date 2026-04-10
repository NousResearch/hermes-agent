class TestAuthenticatedProviderPickerCatalog:
    def test_openai_codex_picker_uses_provider_catalog(self, monkeypatch):
        from hermes_cli import model_switch as ms

        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {})
        monkeypatch.setattr(
            "hermes_cli.models.provider_model_ids",
            lambda provider: ["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"] if provider == "openai-codex" else [],
        )
        monkeypatch.setattr(
            "hermes_cli.auth._load_auth_store",
            lambda: {"providers": {"openai-codex": {"tokens": {"access_token": "***"}}}},
        )

        providers = ms.list_authenticated_providers(
            current_provider="openai-codex",
            user_providers=None,
            max_models=2,
        )

        codex = next(p for p in providers if p["slug"] == "openai-codex")
        assert codex["is_current"] is True
        assert codex["models"] == ["gpt-5.4", "gpt-5.4-mini"]
        assert codex["total_models"] == 3

    def test_mapped_provider_picker_uses_provider_catalog_before_curated_fallback(self, monkeypatch):
        from hermes_cli import model_switch as ms

        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {"zai": {"env": ["GLM_API_KEY"]}})
        monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {"zai": "zai"})
        monkeypatch.setattr(
            "agent.models_dev.get_provider_info",
            lambda provider_id: type("P", (), {"name": "Z.AI / GLM"})(),
        )
        monkeypatch.setenv("GLM_API_KEY", "***")
        monkeypatch.setattr(
            "hermes_cli.models.provider_model_ids",
            lambda provider: ["glm-5", "glm-4.7"] if provider == "zai" else [],
        )

        providers = ms.list_authenticated_providers(
            current_provider="zai",
            user_providers=None,
            max_models=5,
        )

        zai = next(p for p in providers if p["slug"] == "zai")
        assert zai["models"] == ["glm-5", "glm-4.7"]
        assert zai["total_models"] == 2
        assert zai["is_current"] is True
