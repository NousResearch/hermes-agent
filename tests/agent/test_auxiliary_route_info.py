"""Route-info contracts for successful auxiliary provider fallback."""

from __future__ import annotations

import pytest

from agent.auxiliary_client import _LadderRoute, _ladder_provider_fallback


def test_provider_fallback_records_final_destination_and_reason(monkeypatch) -> None:
    route_info: dict[str, str] = {}
    route = _LadderRoute(
        object(), "vision", "", True, "", "primary-provider", "primary/model", None,
        None, None, "primary/model", None, route_info,
    )
    fallback_client = object()
    monkeypatch.setattr(
        "agent.auxiliary_client._try_configured_fallback_chain",
        lambda *args, **kwargs: (fallback_client, "fallback/model", "fallback_chain[0](fallback-provider)"),
    )

    ladder = _ladder_provider_fallback(RuntimeError("payment required"), route)
    step = next(ladder)

    assert step.kind == "fallback"
    assert step.args == (fallback_client, "fallback/model", "fallback_chain[0](fallback-provider)")
    assert route_info == {
        "provider": "fallback-provider",
        "model": "fallback/model",
        "fallback_reason": "payment error",
    }
    with pytest.raises(StopIteration) as completed:
        ladder.send(object())
    assert completed.value.value is not None
