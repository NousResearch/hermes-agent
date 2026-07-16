"""Gateway agent executor worker-count configuration tests."""

import logging
import threading
from pathlib import Path

import yaml

from gateway.config import GatewayConfig, load_gateway_config
from gateway.run import GatewayRunner


def test_agent_executor_workers_defaults_to_ten():
    assert GatewayConfig().agent_executor_workers == 10


def test_agent_executor_workers_accepts_positive_integer_string():
    config = GatewayConfig.from_dict({"agent_executor_workers": "14"})

    assert config.agent_executor_workers == 14
    assert config.to_dict()["agent_executor_workers"] == 14


def test_agent_executor_workers_accepts_large_positive_integer():
    config = GatewayConfig.from_dict({"agent_executor_workers": 10_000})

    assert config.agent_executor_workers == 10_000
    assert config.to_dict()["agent_executor_workers"] == 10_000


def test_agent_executor_workers_invalid_values_fall_back_to_ten(caplog):
    caplog.set_level(logging.WARNING, logger="gateway.config")

    for invalid in (0, -1, True, 1.5, "many"):
        assert GatewayConfig.from_dict(
            {"agent_executor_workers": invalid}
        ).agent_executor_workers == 10

    assert any(
        "Ignoring invalid agent_executor_workers" in record.message
        for record in caplog.records
    )


def test_load_gateway_config_bridges_nested_agent_executor_workers(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"agent_executor_workers": 6}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    config = load_gateway_config()

    assert config.agent_executor_workers == 6


def test_gateway_executor_uses_configured_worker_count():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(agent_executor_workers=3)
    runner._executor_lock = threading.Lock()
    runner._executor = None
    runner._executor_closing = False

    executor = runner._get_executor()
    try:
        assert executor._max_workers == 3
    finally:
        runner._shutdown_executor()
