from types import SimpleNamespace

from gateway.config import Platform
from gateway.run_turn_persistence import GatewayTurnPersistenceMixin


def test_local_final_does_not_receive_platform_reasoning_projection(monkeypatch):
    """Local viewers receive reasoning on their event stream, never in the final text."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"display": {"show_reasoning": True}})
    result = {"last_reasoning": "private chain"}
    runner = GatewayTurnPersistenceMixin()

    local_response = runner._hmwa_prepend_reasoning(
        result, "final answer", SimpleNamespace(platform=Platform.LOCAL), False,
    )
    telegram_response = runner._hmwa_prepend_reasoning(
        result, "final answer", SimpleNamespace(platform=Platform.TELEGRAM), False,
    )

    assert local_response == "final answer"
    assert "private chain" in telegram_response
    assert telegram_response.endswith("final answer")
