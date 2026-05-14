"""Regression tests for OpenAI Codex model validation when the listing lags behind
actually usable backend model IDs.

The bug: `/model` and `switch_model()` reject `gpt-5.3-codex-spark` because the
OpenAI Codex listing omits it, even though direct runtime calls with
`--provider openai-codex -m gpt-5.3-codex-spark` succeed.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model
from hermes_cli.models import validate_requested_model


def test_openai_codex_unknown_but_plausible_model_is_accepted_with_warning():
    """If the Codex listing is incomplete, `/model` should soft-accept the model
    with a warning instead of hard-rejecting it.
    """
    with patch(
        "hermes_cli.models.provider_model_ids",
        return_value=["gpt-5.5", "gpt-5.4", "gpt-5.3-codex"],
    ):
        result = validate_requested_model("gpt-5.3-codex-spark", "openai-codex")

    assert result["accepted"] is True
    assert result["persist"] is True
    assert result["recognized"] is False
    assert "gpt-5.3-codex-spark" in result["message"]
    assert "OpenAI Codex model listing" in result["message"]
    assert "Similar models" in result["message"]
    assert "gpt-5.3-codex" in result["message"]


def test_switch_model_allows_openai_codex_model_missing_from_listing():
    """switch_model() should succeed for Codex models that the runtime accepts
    even when the listing has not caught up yet.
    """
    with patch(
        "hermes_cli.models.provider_model_ids",
        return_value=["gpt-5.5", "gpt-5.4", "gpt-5.3-codex"],
    ):
        result = switch_model(
            "gpt-5.3-codex-spark",
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="",
            current_api_key="",
            user_providers=None,
        )

    assert result.success is True
    assert result.new_model == "gpt-5.3-codex-spark"
    assert result.target_provider == "openai-codex"
    assert result.warning_message
    assert "OpenAI Codex model listing" in result.warning_message
