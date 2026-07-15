"""Configured models extend built-in picker rows."""

from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.providers import HermesOverlay


def _provider_row(configured_models, *, max_models=None):
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
            user_providers={"deepseek": {"models": configured_models}},
            max_models=max_models,
        )
    return next(row for row in rows if row["slug"] == "deepseek")


def test_configured_models_precede_and_deduplicate_discovered_models():
    row = _provider_row({"configured-x": {}, "shared": {}})

    assert row["models"] == ["configured-x", "shared", "live-a"]
    assert row["total_models"] == 3


def test_configured_models_are_merged_before_picker_limit():
    row = _provider_row(["configured-x", "configured-y"], max_models=2)

    assert row["models"] == ["configured-x", "configured-y"]
    assert row["total_models"] == 4


def _overlay_row(configured_models, *, max_models=None):
    """Same setup as _provider_row, but for a HERMES_OVERLAYS provider
    (section 2 — e.g. nous, openai-codex, copilot, opencode-go) instead of
    a PROVIDER_TO_MODELS_DEV built-in (section 1)."""
    with (
        patch("agent.models_dev.fetch_models_dev", return_value={}),
        patch("agent.models_dev.PROVIDER_TO_MODELS_DEV", {}),
        patch(
            "hermes_cli.providers.HERMES_OVERLAYS",
            {
                "test-overlay": HermesOverlay(
                    auth_type="api_key",
                    extra_env_vars=("TEST_OVERLAY_API_KEY",),
                ),
            },
        ),
        patch(
            "hermes_cli.models.cached_provider_model_ids",
            return_value=["live-a", "shared"],
        ),
        patch.dict("os.environ", {"TEST_OVERLAY_API_KEY": "test-key"}),
    ):
        rows = list_authenticated_providers(
            current_provider="test-overlay",
            user_providers={"test-overlay": {"models": configured_models}},
            max_models=max_models,
        )
    return next(row for row in rows if row["slug"] == "test-overlay")


def test_configured_models_extend_hermes_overlay_provider_row():
    """Regression: providers.<slug>.models must extend a HERMES_OVERLAYS
    row (nous, openai-codex, copilot, opencode-go, ...) the same way it
    already extends a built-in (section 1) row — otherwise a model the user
    can already type via /model <name> never appears in the picker list."""
    row = _overlay_row({"configured-x": {}, "shared": {}})

    assert row["models"] == ["configured-x", "shared", "live-a"]
    assert row["total_models"] == 3


def test_configured_models_are_merged_before_picker_limit_for_overlay():
    row = _overlay_row(["configured-x", "configured-y"], max_models=2)

    assert row["models"] == ["configured-x", "configured-y"]
    assert row["total_models"] == 4


def test_configured_models_extend_hermes_overlay_provider_row_with_mapped_slug():
    """Mapped-slug regression: a HERMES_OVERLAYS key can be a models.dev ID
    (e.g. "github-copilot") that PROVIDER_TO_MODELS_DEV maps back to a
    different Hermes/config slug ("copilot") — see the
    "Resolve Hermes slug" comment in list_authenticated_providers(). The
    providers.<hermes_slug>.models config lookup (and the picker's slug/
    label) must key off that RESOLVED slug, not the raw overlay/pid key,
    or a providers.copilot.models block would silently not extend the
    github-copilot overlay row it's meant for. Also exercises the
    hermes_slug in {"openai-codex", "copilot", "copilot-acp"} special
    live-discovery branch specifically (the two other overlay tests above
    use a synthetic slug that only hits the generic fallback branch)."""
    with (
        patch("agent.models_dev.fetch_models_dev", return_value={}),
        patch("agent.models_dev.PROVIDER_TO_MODELS_DEV", {"copilot": "github-copilot"}),
        patch(
            "hermes_cli.providers.HERMES_OVERLAYS",
            {
                "github-copilot": HermesOverlay(
                    auth_type="api_key",
                    extra_env_vars=("TEST_OVERLAY_API_KEY",),
                ),
            },
        ),
        patch(
            "hermes_cli.models.cached_provider_model_ids",
            return_value=["live-a", "shared"],
        ),
        patch.dict("os.environ", {"TEST_OVERLAY_API_KEY": "test-key"}),
    ):
        rows = list_authenticated_providers(
            current_provider="copilot",
            user_providers={"copilot": {"models": {"configured-x": {}, "shared": {}}}},
            max_models=None,
        )
    row = next(row for row in rows if row["slug"] == "copilot")

    assert row["models"] == ["configured-x", "shared", "live-a"]
    assert row["total_models"] == 3
