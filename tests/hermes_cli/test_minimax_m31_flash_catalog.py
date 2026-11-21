"""MiniMax-M3.1-Flash must be selectable in the MiniMax pickers, without moving
the no-selection default (upstream reports #154585, #152406, #134659).

MiniMax serves no ``/models`` listing on the wire Hermes probes, so the curated
static row IS the list the picker offers: a model released after that row was
written cannot be selected at all. Adding a release must not, however, silently
re-point sessions that never chose a model — the curated row's first id is the
provider default (``get_default_model_for_provider``), which is what an
unselected, non-interactive startup falls back to.

Both halves are asserted here: the release is offered, and the default it used
to inherit is unchanged.
"""

from __future__ import annotations

import pytest

MINIMAX_PROVIDERS = ("minimax", "minimax-cn", "minimax-oauth")


@pytest.mark.parametrize("provider", MINIMAX_PROVIDERS)
def test_minimax_picker_offers_m31_flash(provider):
    """The picker must offer the release: the curated row is the only catalog here."""
    from hermes_cli.models import provider_model_ids

    assert "MiniMax-M3.1-Flash" in provider_model_ids(provider)


@pytest.mark.parametrize("provider", MINIMAX_PROVIDERS)
def test_adding_m31_flash_keeps_the_no_selection_default(provider):
    """Offering a new release must not re-point sessions that never chose a model."""
    from hermes_cli.models import get_default_model_for_provider

    assert get_default_model_for_provider(provider) == "MiniMax-M3"
