"""Exercise the real YAML loader and executor; no live Gateway or model calls."""
import json
import queue
import threading

import pytest
import yaml

from gateway.config import GatewayConfig, load_gateway_config
from gateway.run import GatewayRunner


def load_config(tmp_path, monkeypatch, document, legacy=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    if legacy is not None:
        (tmp_path / "gateway.json").write_text(json.dumps(legacy), encoding="utf-8")
    return load_gateway_config()


def bare_runner(config):
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner._executor = None
    runner._executor_lock = threading.Lock()
    runner._executor_closing = False
    return runner


@pytest.mark.parametrize("nested", [False, True], ids=["top-level", "nested"])
@pytest.mark.parametrize("value", [3, 16, "24"])
def test_yaml_value_reaches_executor(tmp_path, monkeypatch, nested, value):
    document = {"agent_executor_workers": value}
    if nested:
        document = {"gateway": document}
    config = load_config(tmp_path, monkeypatch, document)
    runner = bare_runner(config)
    try:
        assert config.agent_executor_workers == int(value)
        assert runner._get_executor()._max_workers == int(value)
    finally:
        runner._shutdown_executor()


@pytest.mark.parametrize("value, expected", [(4, 4), (None, 10), (0, 10), (-2, 10), (True, 10), ("invalid", 10)])
def test_present_top_level_wins_even_when_invalid(tmp_path, monkeypatch, value, expected):
    config = load_config(tmp_path, monkeypatch, {
        "agent_executor_workers": value,
        "gateway": {"agent_executor_workers": 16},
    }, legacy={"agent_executor_workers": 6})
    assert config.agent_executor_workers == expected


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("value, expected", [(16, 16), (None, 10), (0, 10), (False, 10), ([], 10), ({}, 10)])
def test_yaml_overrides_legacy_even_when_invalid(tmp_path, monkeypatch, nested, value, expected):
    document = {"agent_executor_workers": value}
    if nested:
        document = {"gateway": document}
    config = load_config(tmp_path, monkeypatch, document, legacy={"agent_executor_workers": 6})
    assert config.agent_executor_workers == expected


def test_legacy_value_survives_absent_yaml_key(tmp_path, monkeypatch):
    config = load_config(tmp_path, monkeypatch, {}, legacy={"agent_executor_workers": 6})
    assert config.agent_executor_workers == 6


def test_unset_keeps_historical_default(tmp_path, monkeypatch):
    config = load_config(tmp_path, monkeypatch, {})
    assert config.agent_executor_workers == GatewayConfig().agent_executor_workers


@pytest.mark.parametrize("workers", [3, 12])
def test_loaded_pool_starts_requested_workers_and_queues_one_extra(tmp_path, monkeypatch, workers):
    config = load_config(tmp_path, monkeypatch, {"gateway": {"agent_executor_workers": workers}})
    assert config.agent_executor_workers == workers
    runner = bare_runner(config)
    release = threading.Event()
    started = queue.Queue()
    pool = runner._get_executor()
    futures = []

    def hold_slot(index):
        started.put(index)
        if not release.wait(timeout=10):
            raise AssertionError("test cleanup did not release blocked task")
        return index

    try:
        futures = [pool.submit(hold_slot, i) for i in range(workers + 1)]
        active = {started.get(timeout=5) for _ in range(workers)}
        assert active == set(range(workers))
        assert not futures[-1].running()
        assert not futures[-1].done()
        release.set()
        assert [future.result(timeout=5) for future in futures] == list(range(workers + 1))
    finally:
        release.set()
        pool.shutdown(wait=True, cancel_futures=True)
        runner._shutdown_executor()
