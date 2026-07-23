"""Behavior contracts for computer_use latency knobs."""

from unittest.mock import patch

from tools.computer_use import cua_backend
from tools.computer_use import tool as cu_tool


def test_max_image_dimension_default():
    with patch("hermes_cli.config.load_config", return_value={}):
        assert cua_backend._computer_use_max_image_dimension() == 1456


def test_max_image_dimension_zero_disables():
    with patch(
        "hermes_cli.config.load_config",
        return_value={"computer_use": {"max_image_dimension": 0}},
    ):
        assert cua_backend._computer_use_max_image_dimension() is None


def test_capture_after_mode_default_som():
    with patch("hermes_cli.config.load_config", return_value={}):
        assert cu_tool._capture_after_mode() == "som"


def test_capture_after_mode_ax_override():
    with patch(
        "hermes_cli.config.load_config",
        return_value={"computer_use": {"capture_after_mode": "ax"}},
    ):
        assert cu_tool._capture_after_mode() == "ax"


def test_capture_after_mode_invalid_falls_back_to_som():
    with patch(
        "hermes_cli.config.load_config",
        return_value={"computer_use": {"capture_after_mode": "bogus"}},
    ):
        assert cu_tool._capture_after_mode() == "som"


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
