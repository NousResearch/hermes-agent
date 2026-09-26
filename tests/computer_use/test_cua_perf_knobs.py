"""Behavior contracts for computer_use latency knobs."""


from tools.computer_use import tool as cu_tool


def test_aux_vision_route_caches_per_provider_model_and_config(monkeypatch):
    cu_tool._AUX_VISION_ROUTE_CACHE.clear()
    calls = {"load": 0, "decide": 0}

    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_provider", lambda: "openai"
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_model", lambda: "gpt-test"
    )

    cfg = {"auxiliary": {"vision": {}}}

    def fake_load():
        calls["load"] += 1
        return cfg

    def fake_decision(*args, **kwargs):
        calls["decide"] += 1
        return True

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", fake_load)
    monkeypatch.setattr(
        "tools.computer_use.vision_routing.should_route_capture_to_aux_vision",
        fake_decision,
    )

    assert cu_tool._should_route_through_aux_vision() is True
    assert cu_tool._should_route_through_aux_vision() is True
    assert calls == {"load": 2, "decide": 1}


def test_aux_vision_route_recomputes_after_config_edit(monkeypatch):
    """A model seen before a config change must not keep its old route."""
    cu_tool._AUX_VISION_ROUTE_CACHE.clear()
    configs = [
        {"auxiliary": {"vision": {"provider": "custom"}}},
        {"auxiliary": {"vision": {"provider": "auto"}}},
    ]
    current = {"index": 0}
    decisions = iter((True, False))

    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_provider", lambda: "openai-codex"
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_model", lambda: "gpt-5.6-sol"
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: configs[current["index"]],
    )
    monkeypatch.setattr(
        "tools.computer_use.vision_routing.should_route_capture_to_aux_vision",
        lambda *args, **kwargs: next(decisions),
    )

    assert cu_tool._should_route_through_aux_vision() is True
    current["index"] = 1
    assert cu_tool._should_route_through_aux_vision() is False
