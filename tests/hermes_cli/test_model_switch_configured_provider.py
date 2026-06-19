"""Tests for configured-provider catalog fallback in switch_model (#48731).

When the user's config.yaml specifies ``model.provider: opencode-go`` and
the current session is on a different provider (e.g. minimax), typing
``/model deepseek-v4-pro`` should switch to opencode-go (which has the
model in its live catalog) rather than falling to
``detect_provider_for_model`` which picks native ``deepseek``.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


# Live catalog opencode-go currently returns from /v1/models (snapshot).
_OPENCODE_GO_LIVE = [
    "minimax-m2.7", "minimax-m2.5",
    "kimi-k2.6", "kimi-k2.5",
    "glm-5.1", "glm-5",
    "deepseek-v4-pro", "deepseek-v4-flash",
    "qwen3.6-plus", "qwen3.5-plus",
    "mimo-v2-pro", "mimo-v2-omni", "mimo-v2.5-pro", "mimo-v2.5",
]


def _run_switch_from_minimax(
    raw_input: str,
    configured_provider: str = "opencode-go",
    **extra,
):
    """Call switch_model with minimax as current provider, mocking the live
    catalog and config so the test doesn't hit the network."""
    defaults = dict(
        current_provider="minimax",
        current_model="minimax-m2.7",
        current_base_url="https://api.minimax.com/v1",
        current_api_key="sk-test-minimax",
        is_global=False,
        explicit_provider="",
    )
    defaults.update(extra)

    def fake_list_provider_models(provider: str):
        if provider == "opencode-go":
            return list(_OPENCODE_GO_LIVE)
        if provider == "minimax":
            return ["minimax-m2.7", "minimax-m2.5"]
        return []

    fake_config = {
        "model": {
            "provider": configured_provider,
            "default": "minimax-m2.7",
        }
    }

    with (
        patch(
            "hermes_cli.model_switch.list_provider_models",
            side_effect=fake_list_provider_models,
        ),
        patch(
            "hermes_cli.config.load_config",
            return_value=fake_config,
        ),
    ):
        return switch_model(raw_input=raw_input, **defaults)


def test_configured_provider_fallback_deepseek_v4_pro():
    """#48731: /model deepseek-v4-pro on minimax with configured opencode-go
    should switch to opencode-go, not native deepseek."""
    result = _run_switch_from_minimax("deepseek-v4-pro")
    assert result.target_provider == "opencode-go", (
        f"Expected opencode-go, got {result.target_provider}. "
        f"Configured provider's live catalog was not checked."
    )
    assert result.new_model == "deepseek-v4-pro"


def test_configured_provider_fallback_deepseek_v4_flash():
    """Same bug class — flash variant."""
    result = _run_switch_from_minimax("deepseek-v4-flash")
    assert result.target_provider == "opencode-go"
    assert result.new_model == "deepseek-v4-flash"


def test_configured_provider_same_as_current_no_switch():
    """When configured provider == current provider, no extra check needed."""
    result = _run_switch_from_minimax(
        "deepseek-v4-pro",
        configured_provider="minimax",
    )
    # minimax doesn't have deepseek-v4-pro in its catalog, so
    # detect_provider_for_model kicks in → returns native deepseek
    assert result.target_provider == "deepseek"


def test_configured_provider_not_aggregator_no_switch():
    """When configured provider is not an aggregator, skip the check."""
    result = _run_switch_from_minimax(
        "deepseek-v4-pro",
        configured_provider="minimax",  # minimax is not an aggregator
    )
    assert result.target_provider == "deepseek"


def test_configured_provider_model_not_in_catalog_falls_through():
    """When the configured provider's catalog doesn't have the model,
    fall through to detect_provider_for_model."""
    result = _run_switch_from_minimax(
        "gpt-5.3-codex-spark",  # not in opencode-go's catalog
        configured_provider="opencode-go",
    )
    # Should fall through to static detection
    assert result.target_provider != "opencode-go"


def test_configured_provider_config_unavailable_falls_through():
    """When config loading fails, fall through gracefully."""
    defaults = dict(
        current_provider="minimax",
        current_model="minimax-m2.7",
        current_base_url="https://api.minimax.com/v1",
        current_api_key="sk-test-minimax",
        is_global=False,
    )

    def fake_list_provider_models(provider: str):
        if provider == "minimax":
            return ["minimax-m2.7", "minimax-m2.5"]
        return []

    with (
        patch(
            "hermes_cli.model_switch.list_provider_models",
            side_effect=fake_list_provider_models,
        ),
        patch(
            "hermes_cli.config.load_config",
            side_effect=Exception("config unavailable"),
        ),
    ):
        result = switch_model(raw_input="deepseek-v4-pro", **defaults)

    # Should fall through to static detection → native deepseek
    assert result.target_provider == "deepseek"


def test_explicit_provider_bypasses_configured_check():
    """When --provider is given, the configured provider check is skipped."""
    result = _run_switch_from_minimax(
        "deepseek-v4-pro",
        explicit_provider="minimax",
    )
    # With explicit_provider="minimax", the model goes through Path A,
    # not the configured provider fallback
    assert result.target_provider == "minimax"
