"""A pinned engineering model must not silently move through the auxiliary fallback ladder."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import auxiliary_client as aux


def request(provider="provider-a", model="selected"):
    return SimpleNamespace(
        client=object(),
        kwargs={"model": model, "messages": []},
        request_provider=provider,
        base_info="",
        resolved_api_mode="chat_completions",
        resolved_base_url="",
        final_model=model,
    )


def test_strict_route_refuses_fallback_after_provider_failure():
    error = RuntimeError("provider at capacity")
    with (
        patch.object(aux, "_plan_aux_call", return_value=(request(), {}, {})),
        patch.object(aux, "_relay_sync_completion", side_effect=error),
        patch.object(aux, "_should_retry_same_provider", return_value=False),
        patch.object(aux, "_drive_ladder") as fallback,
    ):
        with pytest.raises(RuntimeError, match="provider at capacity"):
            aux._call_llm_impl(
                provider="provider-a",
                model="selected",
                messages=[],
                strict_route=True,
            )
    fallback.assert_not_called()


def test_strict_route_refuses_resolver_model_change_before_request():
    with (
        patch.object(
            aux, "_plan_aux_call", return_value=(request(model="promoted"), {}, {})
        ),
        patch.object(aux, "_relay_sync_completion") as wire,
    ):
        with pytest.raises(RuntimeError, match="selected route changed"):
            aux._call_llm_impl(
                provider="provider-a",
                model="selected",
                messages=[],
                strict_route=True,
            )
    wire.assert_not_called()
