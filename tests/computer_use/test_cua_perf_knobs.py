"""Behavior contracts for computer_use latency knobs."""


from tools.computer_use import tool as cu_tool


def test_aux_vision_route_caches_per_provider_model(monkeypatch):
    cu_tool._AUX_VISION_ROUTE_CACHE.clear()
    calls = {"n": 0}

    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_provider", lambda: "openai"
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_model", lambda: "gpt-test"
    )

    def fake_load():
        calls["n"] += 1
        return {"auxiliary": {"vision": {}}}

    monkeypatch.setattr("hermes_cli.config.load_config", fake_load)
    monkeypatch.setattr(
        "tools.computer_use.vision_routing.should_route_capture_to_aux_vision",
        lambda *a, **k: True,
    )

    assert cu_tool._should_route_through_aux_vision() is True
    assert cu_tool._should_route_through_aux_vision() is True
    assert calls["n"] == 1
