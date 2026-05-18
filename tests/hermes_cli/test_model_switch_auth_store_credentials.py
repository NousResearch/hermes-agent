from types import SimpleNamespace

from hermes_cli.model_switch import list_authenticated_providers
import hermes_cli.providers as providers_mod


class _EmptyPool:
    def has_credentials(self):
        return False


def test_list_authenticated_providers_ignores_empty_builtin_credential_pool_entries(monkeypatch):
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: {"minimax-cn": {"env": ["MINIMAX_CN_API_KEY"]}},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._load_auth_store",
        lambda: {"credential_pool": {"minimax-cn": []}, "providers": {}},
    )
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda _slug: _EmptyPool())
    monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_CN_BASE_URL", raising=False)

    providers = list_authenticated_providers(
        current_provider="",
        user_providers={},
        custom_providers=[],
        max_models=50,
    )

    assert all(p["slug"] != "minimax-cn" for p in providers)


def test_list_authenticated_providers_keeps_nonempty_builtin_credential_pool_entries(monkeypatch):
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: {"minimax-cn": {"env": ["MINIMAX_CN_API_KEY"]}},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._load_auth_store",
        lambda: {"credential_pool": {"minimax-cn": [{"api_key": "sk-test"}]}, "providers": {}},
    )
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda _slug: _EmptyPool())
    monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_CN_BASE_URL", raising=False)

    providers = list_authenticated_providers(
        current_provider="",
        user_providers={},
        custom_providers=[],
        max_models=50,
    )

    assert any(p["slug"] == "minimax-cn" for p in providers)


def test_list_authenticated_providers_ignores_empty_canonical_credential_pool_entries(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr(
        "hermes_cli.models.CANONICAL_PROVIDERS",
        [SimpleNamespace(slug="minimax-cn", label="MiniMax CN")],
    )
    monkeypatch.setattr(
        "hermes_cli.auth._load_auth_store",
        lambda: {"credential_pool": {"minimax-cn": []}, "providers": {}},
    )
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda _slug: _EmptyPool())
    monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_CN_BASE_URL", raising=False)

    providers = list_authenticated_providers(
        current_provider="",
        user_providers={},
        custom_providers=[],
        max_models=50,
    )

    assert all(p["slug"] != "minimax-cn" for p in providers)
