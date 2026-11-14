"""Focused regressions for model-scoped auxiliary fallback identity and endpoints."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _entry(token, *, base_url="https://pool.example/v1", cooldowns=None):
    return SimpleNamespace(
        runtime_api_key=token,
        access_token=token,
        provider="anthropic",
        base_url=base_url,
        model_cooldowns=cooldowns or {},
    )


def _pool(entries, selected=None):
    selected = selected or entries[0]

    class Pool:
        def has_credentials(self):
            return True

        def select(self, *, model=None):
            return selected

        def token_is_blocked(self, token, *, model=None):
            return any(
                entry.runtime_api_key == token
                and bool(entry.model_cooldowns.get(model))
                for entry in entries
            )

    return Pool()


def test_cooled_explicit_auxiliary_credential_is_rejected(monkeypatch):
    from agent import auxiliary_client as aux

    cooled = _entry("synthetic-cooled", cooldowns={"model-a": 9999999999})
    fallback = _entry("synthetic-fallback")
    pool = _pool([cooled, fallback], selected=fallback)
    monkeypatch.setattr(aux, "_get_aux_model_for_provider", lambda provider: "model-a")
    monkeypatch.setattr(aux, "_load_pool_with_credentials", lambda provider, note="": pool)
    with patch("agent.anthropic_adapter.build_anthropic_client") as build:
        client, model = aux._try_anthropic(explicit_api_key="synthetic-cooled")
    assert client is None and model is None
    build.assert_not_called()


def test_same_credential_remains_eligible_for_another_model(monkeypatch):
    from agent import auxiliary_client as aux

    credential = _entry("synthetic-token", cooldowns={"model-a": 9999999999})
    pool = _pool([credential], selected=credential)
    monkeypatch.setattr(aux, "_get_aux_model_for_provider", lambda provider: "model-b")
    monkeypatch.setattr(aux, "_load_pool_with_credentials", lambda provider, note="": pool)
    with patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()):
        client, model = aux._try_anthropic(explicit_api_key="synthetic-token")
    assert client is not None
    assert model == "model-b"


def test_distinct_eligible_fallback_is_selected(monkeypatch):
    from agent import auxiliary_client as aux

    fallback = SimpleNamespace(
        runtime_api_key="synthetic-fallback",
        access_token="synthetic-fallback",
        provider="openrouter",
        base_url="https://fallback.example/v1",
        model_cooldowns={},
    )
    pool = _pool([fallback], selected=fallback)
    monkeypatch.setattr(aux, "_load_pool_with_credentials", lambda provider, note="": pool)
    monkeypatch.setattr(aux, "_aux_openrouter_settings", lambda: (False, "model-b"))
    monkeypatch.setattr(aux, "_is_free_model", lambda model: True)
    with patch("agent.auxiliary_client._create_openai_client", return_value="client") as create:
        client, model = aux._try_openrouter()
    assert client == "client" and model == "model-b"
    assert create.call_args.kwargs["api_key"] == "synthetic-fallback"


def test_fingerprint_backed_model_cooldown_stays_model_specific():
    from agent import auxiliary_client as aux

    credential = _entry("synthetic-fingerprint", cooldowns={"model-a": 9999999999})
    pool = _pool([credential])
    assert not aux._pool_credential_is_model_eligible("anthropic", "synthetic-fingerprint", "model-a", pool)
    assert aux._pool_credential_is_model_eligible("anthropic", "synthetic-fingerprint", "model-b", pool)


def test_unrelated_runtime_credential_does_not_inherit_pool_endpoint():
    from agent import auxiliary_client as aux

    row = _entry("synthetic-row", base_url="https://owned.example/v1")
    assert aux._pool_runtime_base_url_for_key(row, "synthetic-other", "https://default.example/v1") == "https://default.example/v1"


def test_matching_runtime_credential_keeps_pool_endpoint():
    from agent import auxiliary_client as aux

    row = _entry("synthetic-row", base_url="https://owned.example/v1")
    assert aux._pool_runtime_base_url_for_key(row, "synthetic-row", "https://default.example/v1") == "https://owned.example/v1"


def test_explicit_caller_endpoint_precedes_pool_endpoint(monkeypatch):
    from agent import auxiliary_client as aux

    row = SimpleNamespace(
        runtime_api_key="synthetic-row",
        access_token="synthetic-row",
        provider="openrouter",
        base_url="https://owned.example/v1",
        model_cooldowns={},
    )
    pool = _pool([row], selected=row)
    monkeypatch.setattr(aux, "_load_pool_with_credentials", lambda provider, note="": pool)
    monkeypatch.setattr(aux, "_aux_openrouter_settings", lambda: (False, "model-b"))
    monkeypatch.setattr(aux, "_is_free_model", lambda model: True)
    with patch("agent.auxiliary_client._create_openai_client", return_value="client") as create:
        aux._try_openrouter(explicit_base_url="https://caller.example/v1")
    assert create.call_args.kwargs["base_url"] == "https://caller.example/v1"
