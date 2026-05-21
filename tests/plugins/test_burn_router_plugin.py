from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from plugins.burn_router import (
    BurnRouterConfig,
    get_burn_router_hint,
    observe_burn_router_turn,
    register,
)
from hermes_cli.plugins import PluginManager


class _Ctx:
    def __init__(self):
        self.hooks = []

    def register_hook(self, name, callback):
        self.hooks.append((name, callback))


def test_registers_pre_llm_hook():
    ctx = _Ctx()
    register(ctx)
    assert len(ctx.hooks) == 1
    assert ctx.hooks[0][0] == "pre_llm_call"


def test_bundled_plugin_loads_when_enabled_by_documented_key(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - burn_router\n"
        "  entries:\n"
        "    burn_router:\n"
        "      enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    manager = PluginManager()
    manager.discover_and_load(force=True)

    burn = [p for p in manager.list_plugins() if p["key"] == "burn_router"]
    assert burn
    assert burn[0]["enabled"] is True
    assert burn[0]["hooks"] == 1


def test_disabled_returns_none():
    cfg = BurnRouterConfig(enabled=False, binary="router", model="model")
    assert get_burn_router_hint("read a file", cfg) is None


def test_missing_binary_or_model_returns_none():
    assert get_burn_router_hint("read a file", BurnRouterConfig(binary=None, model="m")) is None
    assert get_burn_router_hint("read a file", BurnRouterConfig(binary="b", model=None)) is None


def test_success_observe_mode_logs_no_toolsets():
    completed = subprocess.CompletedProcess(
        args=["router"],
        returncode=0,
        stdout=json.dumps({"category": "file", "confidence": 0.91, "time_us": 123, "all": {"file": 0.91}}),
        stderr="",
    )
    cfg = BurnRouterConfig(enabled=True, mode="observe", binary="router", model="model")

    with patch("plugins.burn_router.subprocess.run", return_value=completed) as run:
        result = get_burn_router_hint("open README", cfg)

    assert result is not None
    assert result.category == "file"
    assert result.confidence == 0.91
    assert result.time_us == 123
    assert result.enabled_toolsets == []
    assert result.mode == "observe"
    run.assert_called_once_with(
        ["router", "predict", "open README", "model"],
        check=False,
        capture_output=True,
        text=True,
        timeout=0.25,
    )


def test_hint_mode_returns_toolsets_only_above_threshold():
    completed = subprocess.CompletedProcess(
        args=["router"],
        returncode=0,
        stdout=json.dumps({"category": "terminal", "confidence": 0.93}),
        stderr="",
    )
    cfg = BurnRouterConfig(enabled=True, mode="hint", binary="router", model="model", confidence_threshold=0.9)

    with patch("plugins.burn_router.subprocess.run", return_value=completed):
        result = get_burn_router_hint("run tests", cfg)

    assert result is not None
    assert result.mode == "hint"
    assert result.enabled_toolsets == ["terminal", "code_execution"]


def test_hint_mode_below_threshold_falls_back_to_full_surface():
    completed = subprocess.CompletedProcess(
        args=["router"],
        returncode=0,
        stdout=json.dumps({"category": "terminal", "confidence": 0.4}),
        stderr="",
    )
    cfg = BurnRouterConfig(enabled=True, mode="hint", binary="router", model="model", confidence_threshold=0.9)

    with patch("plugins.burn_router.subprocess.run", return_value=completed):
        result = get_burn_router_hint("ambiguous", cfg)

    assert result is not None
    assert result.mode == "fallback_full_surface"
    assert result.enabled_toolsets == []


def test_subprocess_failure_is_safe_fallback():
    cfg = BurnRouterConfig(enabled=True, binary="router", model="model")
    with patch("plugins.burn_router.subprocess.run", side_effect=TimeoutError("slow")):
        assert get_burn_router_hint("anything", cfg) is None


def test_malformed_json_is_safe_fallback():
    completed = subprocess.CompletedProcess(args=["router"], returncode=0, stdout="not json", stderr="")
    cfg = BurnRouterConfig(enabled=True, binary="router", model="model")
    with patch("plugins.burn_router.subprocess.run", return_value=completed):
        assert get_burn_router_hint("anything", cfg) is None


def test_pre_llm_observer_returns_none_and_never_injects_context():
    ctx = _Ctx()
    register(ctx)
    _, callback = ctx.hooks[0]
    with patch("plugins.burn_router.observe_burn_router_turn") as observe:
        assert callback(user_message="hello", session_id="s") is None
    observe.assert_called_once()


def test_plugin_config_loads_from_plugins_entries(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "plugins:\n"
        "  enabled: [burn_router]\n"
        "  entries:\n"
        "    burn_router:\n"
        "      binary: /tmp/router\n"
        "      model: /tmp/model\n"
        "      confidence_threshold: 0.81\n"
        "      timeout_seconds: 0.12\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    cfg = BurnRouterConfig.load()

    assert cfg.binary == "/tmp/router"
    assert cfg.model == "/tmp/model"
    assert cfg.confidence_threshold == 0.81
    assert cfg.timeout_seconds == 0.12


@pytest.mark.integration
def test_real_burn_router_sidecar_contract_when_configured():
    """Optional smoke test for reviewers with the real Rust/Burn sidecar.

    Normal CI does not need Rust/Burn artifacts. Set both env vars to verify that
    the Hermes plugin can execute the real sidecar and parse its JSON contract.
    """

    binary = os.getenv("HERMES_BURN_ROUTER_TEST_BINARY")
    model = os.getenv("HERMES_BURN_ROUTER_TEST_MODEL")
    if not binary or not model:
        pytest.skip("set HERMES_BURN_ROUTER_TEST_BINARY and HERMES_BURN_ROUTER_TEST_MODEL to run real sidecar smoke test")

    cases = {
        "search X for trending Base coins": "x_search",
        "read /tmp/foo.txt": "file",
        "generate an image of a skull rocket": "media_generation",
        "schedule this every 2 hours": "cron",
    }
    cfg = BurnRouterConfig(enabled=True, mode="observe", binary=binary, model=model, timeout_seconds=0.25)

    for message, expected_category in cases.items():
        result = get_burn_router_hint(message, cfg)
        assert result is not None, message
        assert result.category == expected_category
        assert result.confidence >= 0.72
        assert result.mode == "observe"
        assert result.enabled_toolsets == []
        assert result.time_us is None or result.time_us >= 0
