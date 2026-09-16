"""``auxiliary.*.provider: auto`` never bills a provider the user did not select.

Regression for the Grok/Nous incident: main provider ``xai-oauth`` with an expired token, Nous
Portal still logged in from an earlier setup, no ``fallback_providers``. Every compression, title
and memory-flush call for the Grok conversation fell through to the built-in discovery chain and
was charged to the Portal balance while the chat visibly stayed on Grok. The discovery chain is
only for installs that have no selected main provider at all.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as aux


@pytest.fixture
def nous_is_the_only_working_provider():
    """Main provider unusable; Nous would win the discovery chain; no configured fallback policy."""
    aux._aux_unhealthy_until.clear()
    with patch.object(aux, "_try_main_provider_route", return_value=None), \
         patch.object(aux, "_try_configured_fallback_chain", return_value=(None, None, "")), \
         patch.object(aux, "_try_main_fallback_chain", return_value=(None, None, "")), \
         patch.object(aux, "_try_openrouter", return_value=(None, None)), \
         patch.object(aux, "_try_nous", return_value=(MagicMock(name="nous"), "nous-model")):
        yield


@pytest.mark.parametrize("persisted_provider", ["xai-oauth", "auto"])
def test_selected_main_provider_down_refuses_to_guess_another_account(
        nous_is_the_only_working_provider, persisted_provider):
    """The SESSION runtime is the selection: a live `/model xai-oauth` over a persisted
    ``provider: auto`` must not re-open discovery through the disk value."""
    runtime = {"provider": "xai-oauth", "model": "grok-4.6", "base_url": "https://api.x.ai/v1", "api_key": "dead"}
    with patch.object(aux, "_read_main_provider", return_value=persisted_provider):
        assert aux._resolve_auto_route(main_runtime=runtime, task="compression") == (None, None, "")
        assert aux._try_payment_fallback("xai-oauth", task="compression", main_runtime=runtime) == (None, None, "")


def test_quarantined_fallback_hands_over_to_the_next_configured_entry():
    """A stale first fallback (401, refresh failed) is quarantined mid-request; the second
    configured entry must still get its turn instead of the request dying on the original error."""
    healthy = MagicMock(name="second-fallback")
    route = aux._LadderRoute(None, "compression", "", False, "", "xai-oauth", None, None, None, None, None,
                             {"provider": "xai-oauth"}, None)
    with patch.object(aux, "_try_configured_fallback_chain", return_value=(healthy, "m2", "fallback_chain[1](nous)")), \
         patch.object(aux, "_try_payment_fallback") as discovery:
        client, model, label = aux._next_fallback_after_quarantine(
            "compression", "auto", True, route, None, None)
    assert client is healthy and label == "fallback_chain[1](nous)"
    discovery.assert_not_called()


def test_no_selected_main_provider_still_discovers(nous_is_the_only_working_provider):
    with patch.object(aux, "_read_main_provider", return_value="auto"):
        client, model, label = aux._resolve_auto_route(main_runtime={"provider": "auto"}, task="compression")
    assert client is not None and (model, label) == ("nous-model", "nous")
