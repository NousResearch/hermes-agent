"""Configured-only model inventory must prune ambient providers before I/O."""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.inventory import ConfigContext, build_models_payload
from hermes_cli.model_switch import list_authenticated_providers


def _ctx() -> ConfigContext:
    return ConfigContext(
        current_provider="openai-codex",
        current_model="gpt-5.6-sol",
        current_base_url="",
        user_providers={},
        custom_providers=[],
        excluded_providers=[],
    )


def test_payload_forwards_explicit_only_to_discovery() -> None:
    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=[],
    ) as discover:
        build_models_payload(_ctx(), explicit_only=True)

    assert discover.call_args.kwargs["explicit_only"] is True


def test_explicit_only_skips_ambient_catalog_io_before_row_filtering() -> None:
    model_calls: list[str] = []

    def _models(provider: str, **_kwargs) -> list[str]:
        model_calls.append(provider)
        return ["gpt-5.6-sol"]

    with (
        patch("agent.models_dev.fetch_models_dev") as models_dev,
        patch("hermes_cli.models.get_curated_nous_model_ids") as nous_manifest,
        patch("hermes_cli.models.cached_provider_model_ids", side_effect=_models),
        patch(
            "hermes_cli.model_switch._credential_pool_is_usable",
            side_effect=lambda slug, **_kwargs: slug == "openai-codex",
        ),
        patch(
            "hermes_cli.auth.is_provider_explicitly_configured",
            return_value=False,
        ),
    ):
        rows = list_authenticated_providers(
            current_provider="openai-codex",
            current_model="gpt-5.6-sol",
            explicit_only=True,
        )

    assert [row["slug"] for row in rows] == ["openai-codex"]
    assert model_calls == ["openai-codex"]
    models_dev.assert_not_called()
    nous_manifest.assert_not_called()
