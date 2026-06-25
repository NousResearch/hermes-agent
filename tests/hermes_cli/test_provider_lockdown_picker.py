from hermes_cli import model_switch


def test_list_picker_providers_lockdown_keeps_replabs_current_and_user_owned(monkeypatch):
    providers = [
        {"slug": "openrouter", "source": "built-in", "models": ["a"], "total_models": 1},
        {"slug": "replabs", "source": "user-config", "models": ["z-ai/glm-5.2"], "total_models": 1},
        {"slug": "openai-codex", "source": "hermes", "models": ["gpt-5.5"], "total_models": 1},
        {"slug": "custom-local", "source": "user-config", "models": ["local"], "total_models": 1},
        {"slug": "anthropic", "source": "built-in", "models": ["claude"], "total_models": 1},
    ]

    monkeypatch.setattr(model_switch, "list_authenticated_providers", lambda **_: providers)
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models", lambda: [("a", "A")])

    out = model_switch.list_picker_providers(
        current_provider="openai-codex",
        current_model="gpt-5.5",
        provider_lockdown=True,
    )

    assert [p["slug"] for p in out] == ["replabs", "openai-codex", "custom-local"]


def test_list_picker_providers_without_lockdown_keeps_catalog(monkeypatch):
    providers = [
        {"slug": "openrouter", "source": "built-in", "models": ["a"], "total_models": 1},
        {"slug": "anthropic", "source": "built-in", "models": ["claude"], "total_models": 1},
    ]

    monkeypatch.setattr(model_switch, "list_authenticated_providers", lambda **_: providers)
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models", lambda: [("a", "A")])

    out = model_switch.list_picker_providers(provider_lockdown=False)

    assert [p["slug"] for p in out] == ["openrouter", "anthropic"]
