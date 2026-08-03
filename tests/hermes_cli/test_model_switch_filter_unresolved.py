"""Picker rows must resolve to a runtime provider (#57503)."""

from unittest.mock import patch

import pytest

from hermes_cli.auth import (
    AuthError,
    is_runtime_provider_routable,
    resolve_provider,
)
from hermes_cli.model_switch import list_authenticated_providers


def _rows_with_env(monkeypatch, env_name: str, provider: str) -> list[dict]:
    monkeypatch.setenv(env_name, "test-key")
    with (
        patch(
            "agent.models_dev.fetch_models_dev",
            return_value={provider: {"env": [env_name], "name": provider.title()}},
        ),
        patch(
            "agent.models_dev.PROVIDER_TO_MODELS_DEV",
            {provider: provider},
        ),
        patch("hermes_cli.models.cached_provider_model_ids", return_value=["model-a"]),
        patch("hermes_cli.providers.HERMES_OVERLAYS", {}),
    ):
        return list_authenticated_providers(max_models=5)


def test_models_dev_only_provider_is_not_selectable(monkeypatch):
    rows = _rows_with_env(monkeypatch, "MISTRAL_API_KEY", "mistral")

    assert all(row["slug"] != "mistral" for row in rows)
    assert not is_runtime_provider_routable("mistral")


def test_special_runtime_provider_does_not_require_registry_membership():
    assert is_runtime_provider_routable("openrouter")
    assert is_runtime_provider_routable("custom:local-lab")


def test_disabled_canonical_identity_is_not_revived_by_active_alias(
    monkeypatch,
):
    """Exact canonical activation wins over the generic alias/name union."""
    managed = {"deepseek", "anthropic", "claude"}
    monkeypatch.setattr(
        "providers.is_plugin_managed_provider_id",
        lambda provider_id: provider_id in managed,
    )
    monkeypatch.setattr(
        "providers.get_provider_identity_provenance",
        lambda provider_id: (
            "canonical"
            if provider_id in {"deepseek", "anthropic"}
            else "alias"
            if provider_id == "claude"
            else None
        ),
    )
    monkeypatch.setattr("providers.is_provider_plugin_active", lambda _id: True)
    monkeypatch.setattr(
        "providers.is_provider_canonical_identity_active",
        lambda provider_id: provider_id != "deepseek",
    )

    with pytest.raises(AuthError, match="disabled by plugin configuration"):
        resolve_provider("deepseek")
    assert not is_runtime_provider_routable("deepseek")
    assert all(
        row["slug"] != "deepseek"
        for row in _rows_with_env(
            monkeypatch,
            "DEEPSEEK_API_KEY",
            "deepseek",
        )
    )

    # Ordinary aliases retain their generic activation semantics.
    assert resolve_provider("claude") == "anthropic"
    assert is_runtime_provider_routable("claude")


def test_canonical_crosscheck_defensively_respects_routability(monkeypatch):
    """Section 2b must not re-emit an unroutable canonical snapshot row."""
    from types import SimpleNamespace

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    with (
        patch("agent.models_dev.fetch_models_dev", return_value={}),
        patch("agent.models_dev.PROVIDER_TO_MODELS_DEV", {}),
        patch("hermes_cli.providers.HERMES_OVERLAYS", {}),
        patch(
            "hermes_cli.models.CANONICAL_PROVIDERS",
            [SimpleNamespace(slug="deepseek", label="DeepSeek")],
        ),
        patch(
            "hermes_cli.auth.is_runtime_provider_routable",
            side_effect=lambda provider_id: provider_id != "deepseek",
        ),
    ):
        rows = list_authenticated_providers(max_models=5)

    assert all(row["slug"] != "deepseek" for row in rows)
