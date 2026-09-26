"""RED first: live tokens/s + context-window telemetry for the managed llama.cpp runtime.

Issue #123678: Hermes shows no engine throughput (prompt/gen tokens/s) or
context-window usage for the local llama-server — both are only visible by
tailing llama-server.log today.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import web_server

    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return test_client


def test_rates_from_llamacpp_timings():
    from hermes_cli.local_runtime import telemetry

    prompt_tps, gen_tps = telemetry.rates_from_timings({
        "prompt_n": 1800, "prompt_ms": 10000.0,
        "predicted_n": 103, "predicted_ms": 10000.0,
    })
    assert prompt_tps == 180.0
    assert gen_tps == 10.3


def test_rates_missing_or_zero_timings_is_none():
    from hermes_cli.local_runtime import telemetry

    assert telemetry.rates_from_timings({}) == (None, None)
    assert telemetry.rates_from_timings(None) == (None, None)
    assert telemetry.rates_from_timings({"prompt_n": 5, "prompt_ms": 0}) == (None, None)


def test_context_percent_math():
    from hermes_cli.local_runtime import telemetry

    assert telemetry.context_percent(131072, 262144) == 50
    assert telemetry.context_percent(0, 262144) == 0
    assert telemetry.context_percent(999, 0) is None
    assert telemetry.context_percent(300000, 262144) == 100


def test_slots_usage_parses_busy_slot():
    from hermes_cli.local_runtime import telemetry

    used, n_ctx = telemetry.slots_usage([
        {"id": 0, "n_ctx": 262144, "is_processing": False,
         "n_prompt_tokens_processed": 0, "n_tokens_predicted": 0},
        {"id": 1, "n_ctx": 262144, "is_processing": True,
         "n_prompt_tokens_processed": 12000, "n_tokens_predicted": 340},
    ])
    assert n_ctx == 262144
    assert used == 12340


def test_last_turn_store_roundtrip():
    from hermes_cli.local_runtime import telemetry

    assert telemetry.last_turn("no-such-model-xyz") is None
    telemetry.record_last_turn("no-such-model-xyz", 180.0, 10.3, 12000, 340)
    stored = telemetry.last_turn("no-such-model-xyz")
    assert stored is not None
    assert stored["prompt_tps"] == 180.0 and stored["gen_tps"] == 10.3


def test_get_runtime_stats_none_without_managed_server(monkeypatch):
    from hermes_cli.local_runtime import telemetry

    monkeypatch.setattr(telemetry, "managed_root", lambda: None)
    assert telemetry.get_runtime_stats("anything", _no_cache=True) is None


def test_get_runtime_stats_reports_percent_and_rates(monkeypatch):
    from hermes_cli.local_runtime import telemetry

    monkeypatch.setattr(telemetry, "managed_root", lambda: ("http://127.0.0.1:9", "k"))

    def _fake_get(base, key, route, timeout_s):
        if route.startswith("/props"):
            return {"default_generation_settings": {"n_ctx": 262144}}
        if route.startswith("/slots"):
            return [{"id": 0, "n_ctx": 262144, "is_processing": True,
                     "n_prompt_tokens_processed": 131072, "n_tokens_predicted": 0}]
        raise AssertionError(route)

    monkeypatch.setattr(telemetry, "managed_get_json", _fake_get)
    telemetry.record_last_turn("m", 180.0, 10.3, 131072, 0)
    stats = telemetry.get_runtime_stats("m", _no_cache=True)
    assert stats is not None
    assert stats["n_ctx"] == 262144
    assert stats["used_tokens"] == 131072
    assert stats["context_percent"] == 50
    assert stats["prompt_tps"] == 180.0 and stats["gen_tps"] == 10.3
    assert "tok/s" in stats["throughput_label"]
    assert "%" in stats["context_label"]


def test_status_includes_runtime_stats_key(client, monkeypatch):
    """Desktop pane contract: /api/local-models/status always carries runtime_stats."""
    from hermes_cli.web_routers import local_models

    monkeypatch.setattr(local_models, "_state_endpoint", lambda: None)
    data = client.get("/api/local-models/status").json()
    assert data["runtime_stats"] == {}


def test_status_reports_live_engine_stats_for_loaded_model(client, monkeypatch):
    from hermes_cli.local_runtime import telemetry
    from hermes_cli.web_routers import local_models

    model_id = "live-engine-model"
    monkeypatch.setattr(local_models, "_state_endpoint",
                        lambda: {"base_url": "http://127.0.0.1:9/v1", "api_key": "k"})
    monkeypatch.setattr(local_models, "_loaded_models", lambda running: ({model_id: "loaded"}, {}))

    def _fake_get(base, key, route, timeout_s):
        if route.startswith("/props"):
            return {"default_generation_settings": {"n_ctx": 262144}}
        return [{"id": 0, "n_ctx": 262144, "is_processing": True,
                 "n_prompt_tokens_processed": 65536, "n_tokens_predicted": 0}]

    monkeypatch.setattr(telemetry, "managed_root", lambda: ("http://127.0.0.1:9", "k"))
    monkeypatch.setattr(telemetry, "managed_get_json", _fake_get)
    data = client.get("/api/local-models/status").json()
    stats = data["runtime_stats"][model_id]
    assert stats["n_ctx"] == 262144
    assert stats["context_percent"] == 25


def test_extract_timings_from_response_shapes():
    from hermes_cli.local_runtime import telemetry
    from types import SimpleNamespace

    assert telemetry.extract_timings(None) is None
    assert telemetry.extract_timings(object()) is None
    timings = {"prompt_n": 10, "prompt_ms": 100.0}
    assert telemetry.extract_timings(SimpleNamespace(timings=timings)) == timings
    assert telemetry.extract_timings({"timings": timings}) == timings
