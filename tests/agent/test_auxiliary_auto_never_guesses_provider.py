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


def test_selected_main_provider_down_refuses_to_guess_another_account(nous_is_the_only_working_provider):
    runtime = {"provider": "xai-oauth", "model": "grok-4.6", "base_url": "https://api.x.ai/v1", "api_key": "dead"}
    with patch.object(aux, "_read_main_provider", return_value="xai-oauth"):
        assert aux._resolve_auto_route(main_runtime=runtime, task="compression") == (None, None, "")
        assert aux._try_payment_fallback("xai-oauth", task="compression") == (None, None, "")


def test_no_selected_main_provider_still_discovers(nous_is_the_only_working_provider):
    with patch.object(aux, "_read_main_provider", return_value="auto"):
        client, model, label = aux._resolve_auto_route(main_runtime={"provider": "auto"}, task="compression")
    assert client is not None and (model, label) == ("nous-model", "nous")
