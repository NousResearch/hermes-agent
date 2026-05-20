import json
import logging
import subprocess

from agent.burn_router import BurnRouterConfig, BurnRouterResult, get_burn_router_hint
from hermes_cli.config import DEFAULT_CONFIG


def test_default_config_declares_router_disabled_by_default():
    burn_cfg = DEFAULT_CONFIG["routing"]["burn_router"]

    assert burn_cfg["enabled"] is False
    assert burn_cfg["mode"] == "observe"
    assert burn_cfg["confidence_threshold"] == 0.72


def test_config_loader_reads_nested_burn_router_settings():
    cfg = BurnRouterConfig.from_config({
        "routing": {
            "burn_router": {
                "enabled": True,
                "mode": "hint",
                "binary": "/tmp/router",
                "model": "/tmp/model.safetensors",
                "confidence_threshold": 0.9,
                "timeout_seconds": 0.5,
            }
        }
    })

    assert cfg.enabled is True
    assert cfg.mode == "hint"
    assert cfg.binary == "/tmp/router"
    assert cfg.model == "/tmp/model.safetensors"
    assert cfg.confidence_threshold == 0.9
    assert cfg.timeout_seconds == 0.5


def test_disabled_router_returns_none_without_subprocess(monkeypatch):
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess should not run when disabled")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert get_burn_router_hint("search X for coins", BurnRouterConfig(enabled=False)) is None
    assert called is False


def test_observe_mode_returns_prediction_without_narrowing(monkeypatch):
    payload = {
        "category": "x_search",
        "confidence": 0.99,
        "time_us": 640.0,
        "all": {"x_search": 0.99, "web": 0.01},
    }

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    cfg = BurnRouterConfig(
        enabled=True,
        mode="observe",
        binary="/tmp/router",
        model="/tmp/model.safetensors",
        confidence_threshold=0.72,
    )

    result = get_burn_router_hint("search X for coins", cfg)

    assert isinstance(result, BurnRouterResult)
    assert result.category == "x_search"
    assert result.confidence == 0.99
    assert result.enabled_toolsets == []
    assert result.mode == "observe"


def test_hint_mode_maps_high_confidence_category_to_toolsets(monkeypatch):
    payload = {"category": "file", "confidence": 0.91, "time_us": 650.0, "all": {"file": 0.91}}

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    cfg = BurnRouterConfig(enabled=True, mode="hint", binary="/tmp/router", model="/tmp/model.safetensors")

    result = get_burn_router_hint("read /tmp/foo.txt", cfg)

    assert result.category == "file"
    assert result.enabled_toolsets == ["file"]
    assert result.mode == "hint"


def test_hint_mode_falls_back_when_confidence_is_low(monkeypatch):
    payload = {"category": "x_search", "confidence": 0.33, "time_us": 650.0, "all": {"x_search": 0.33}}

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    cfg = BurnRouterConfig(
        enabled=True,
        mode="hint",
        binary="/tmp/router",
        model="/tmp/model.safetensors",
        confidence_threshold=0.72,
    )

    result = get_burn_router_hint("ambiguous thing", cfg)

    assert result.category == "x_search"
    assert result.enabled_toolsets == []
    assert result.mode == "fallback_full_surface"


def test_router_failures_are_safe_fallback(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="router", timeout=0.1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    cfg = BurnRouterConfig(enabled=True, mode="hint", binary="/tmp/router", model="/tmp/model.safetensors")

    assert get_burn_router_hint("search X", cfg) is None


def test_observe_burn_router_turn_logs_prediction(monkeypatch, caplog):
    from agent import burn_router

    monkeypatch.setattr(
        burn_router,
        "get_burn_router_hint",
        lambda message, config: BurnRouterResult(category="x_search", confidence=0.98, mode="observe"),
    )

    with caplog.at_level(logging.INFO, logger="agent.burn_router"):
        burn_router.observe_burn_router_turn("search X", {"routing": {"burn_router": {"enabled": True}}})

    assert "burn_router prediction" in caplog.text
    assert "x_search" in caplog.text
