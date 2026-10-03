"""Configured models extend — or, with ``discover_models: false``, pin — built-in picker rows."""

from unittest.mock import patch

import pytest

from hermes_cli.model_switch import list_authenticated_providers


def _provider_row(configured_models, *, max_models=None, discover_models=None):
    provider_cfg = {"models": configured_models}
    if discover_models is not None:
        provider_cfg["discover_models"] = discover_models
    with (
        patch(
            "agent.models_dev.fetch_models_dev",
            return_value={"deepseek": {"env": ["DEEPSEEK_API_KEY"], "name": "DeepSeek"}},
        ),
        patch(
            "agent.models_dev.PROVIDER_TO_MODELS_DEV",
            {"deepseek": "deepseek"},
        ),
        patch(
            "hermes_cli.models.cached_provider_model_ids",
            return_value=["live-a", "shared"],
        ),
        patch("hermes_cli.providers.HERMES_OVERLAYS", {}),
        patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}),
    ):
        rows = list_authenticated_providers(
            current_provider="deepseek",
            user_providers={"deepseek": provider_cfg},
            max_models=max_models,
        )
    return next(row for row in rows if row["slug"] == "deepseek")


def test_configured_models_precede_and_deduplicate_discovered_models():
    row = _provider_row({"configured-x": {}, "shared": {}})

    assert row["models"] == ["configured-x", "shared", "live-a"]
    assert row["total_models"] == 3


def test_discover_models_false_pins_the_builtin_catalog():
    """``discover_models: false`` narrows a built-in row to the declared ids.

    The live catalog is otherwise always appended, so a user on a lab with a
    large catalog can reorder the picker but never reduce it. This is the same
    opt-in custom endpoints already have, and the one ``_models_config_is_allowlist``
    points at for pinning a catalog.
    """
    row = _provider_row(["configured-x", "shared"], discover_models=False)

    assert row["models"] == ["configured-x", "shared"], "live ids must not be appended"
    assert row["total_models"] == 2


@pytest.mark.parametrize("falsey", ["false", "False", "no", "0"])
def test_discover_models_string_false_pins_like_the_bool(falsey):
    """YAML/env round-trips hand back strings; ``_discover_flag`` is the shared parser.

    A plain ``.get("discover_models", True)`` reads ``"false"`` as truthy and
    silently ignores the pin.
    """
    row = _provider_row(["configured-x"], discover_models=falsey)

    assert row["models"] == ["configured-x"]


def test_discover_models_false_without_models_keeps_discovery():
    """An empty ``models`` list is not a pin: honouring it would blank the row."""
    row = _provider_row([], discover_models=False)

    assert row["models"] == ["live-a", "shared"]


def test_discovery_stays_default_without_the_flag():
    """Absent ``discover_models``, a list-shaped ``models`` still extends (no silent narrowing)."""
    row = _provider_row(["configured-x"])

    assert row["models"] == ["configured-x", "live-a", "shared"]
