"""MiniMax-M3.1-Flash is in the MiniMax catalog on every MiniMax endpoint (#154585).

MiniMax has no /models endpoint, so the static catalog is the whole list the desktop picker can
offer and the list ``/model`` validates against. A model missing from it can only be reached by
typing its id, and then draws a "not found in the MiniMax catalog" warning.
"""

from unittest.mock import patch

import pytest

from hermes_cli.models import curated_models_for_provider
from hermes_cli.models_validate import validate_requested_model


@pytest.fixture(autouse=True)
def _no_live_listing():
    with patch("hermes_cli.models.fetch_api_models", return_value=None), \
            patch("hermes_cli.models.probe_api_models", return_value={"models": None}):
        yield


@pytest.mark.parametrize("provider", ["minimax", "minimax-cn"])
def test_m3_1_flash_validates_as_a_known_minimax_model(provider):
    result = validate_requested_model("MiniMax-M3.1-Flash", provider)
    assert result["accepted"] is True
    assert result["recognized"] is True
    assert result["message"] is None


@pytest.mark.parametrize("provider", ["minimax", "minimax-cn", "minimax-oauth"])
def test_m3_1_flash_is_offered_in_every_minimax_picker_row(provider):
    assert "MiniMax-M3.1-Flash" in [mid for mid, _ in curated_models_for_provider(provider)]
