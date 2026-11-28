"""Regression for #154762: listing MiniMax Flash must not change the no-selection default."""

import pytest

from hermes_cli.models import get_default_model_for_provider, static_catalog


@pytest.mark.parametrize("provider", ("minimax", "minimax-cn", "minimax-oauth"))
def test_minimax_flash_selectable_without_changing_silent_default(provider):
    """New releases enter the picker without silently switching existing sessions."""
    catalog = static_catalog(provider)
    assert "MiniMax-M3.1-Flash" in catalog
    assert get_default_model_for_provider(provider) == "MiniMax-M3"
