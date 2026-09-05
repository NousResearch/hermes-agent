"""Tests for hermes chat -q --usage-file."""

import json
from types import SimpleNamespace

from hermes_cli.usage_report import (
    result_from_cli,
    write_usage_file,
    write_usage_file_from_cli,
)


def test_result_from_cli_prefers_agent_then_session():
    agent = SimpleNamespace(
        session_id="sess-agent",
        session_estimated_cost_usd=1.25,
        session_cost_status="estimated",
        session_cost_source="provider_models_api",
        session_input_tokens=100,
        session_output_tokens=20,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=5,
        session_total_tokens=125,
        api_call_count=3,
        model="moonshotai/kimi-k3",
        provider="openrouter",
        request_overrides={},
    )
    db = SimpleNamespace(
        get_session=lambda _sid: {
            "id": "sess-agent",
            "estimated_cost_usd": 9.99,
            "actual_cost_usd": 0.0,
            "input_tokens": 1,
        }
    )
    cli = SimpleNamespace(agent=agent, session_id="sess-agent", _session_db=db, usage_file=None)
    report = result_from_cli(cli)
    assert report["estimated_cost_usd"] == 1.25
    assert report["actual_cost_usd"] == 0.0
    assert report["input_tokens"] == 100
    assert report["session_id"] == "sess-agent"
    assert report["model"] == "moonshotai/kimi-k3"


def test_result_from_cli_falls_back_to_session_when_agent_empty():
    db = SimpleNamespace(
        get_session=lambda _sid: {
            "id": "sess-db",
            "estimated_cost_usd": 4.5,
            "actual_cost_usd": None,
            "input_tokens": 80,
            "output_tokens": 10,
            "model": "deepseek/deepseek-v4-pro",
        }
    )
    cli = SimpleNamespace(
        agent=None,
        session_id="sess-db",
        _session_db=db,
        usage_file=None,
    )
    report = result_from_cli(cli)
    assert report["estimated_cost_usd"] == 4.5
    assert report["input_tokens"] == 80
    assert report["model"] == "deepseek/deepseek-v4-pro"


def test_write_usage_file_includes_actual_cost(tmp_path):
    path = tmp_path / "usage.json"
    write_usage_file(
        str(path),
        {
            "estimated_cost_usd": 1.5,
            "actual_cost_usd": 1.4,
            "input_tokens": 10,
            "output_tokens": 2,
            "model": "x",
            "provider": "openrouter",
            "session_id": "abc",
            "completed": True,
            "failed": False,
        },
    )
    report = json.loads(path.read_text())
    assert report["estimated_cost_usd"] == 1.5
    assert report["actual_cost_usd"] == 1.4
    assert report["session_id"] == "abc"


def test_write_usage_file_from_cli_noop_without_flag(tmp_path):
    cli = SimpleNamespace(
        usage_file=None,
        agent=SimpleNamespace(session_estimated_cost_usd=1.0, session_id="s"),
        session_id="s",
        _session_db=None,
    )
    write_usage_file_from_cli(cli)
    assert not (tmp_path / "usage.json").exists()


def test_write_usage_file_from_cli_writes_when_flag_set(tmp_path):
    path = tmp_path / "nested" / "usage.json"
    agent = SimpleNamespace(
        session_id="sess-1",
        session_estimated_cost_usd=0.42,
        session_cost_status="estimated",
        session_cost_source="provider_models_api",
        session_input_tokens=50,
        session_output_tokens=8,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_total_tokens=58,
        api_call_count=1,
        model="moonshotai/kimi-k3",
        provider="openrouter",
        request_overrides={},
    )
    cli = SimpleNamespace(
        usage_file=str(path),
        agent=agent,
        session_id="sess-1",
        _session_db=None,
    )
    write_usage_file_from_cli(cli)
    report = json.loads(path.read_text())
    assert report["estimated_cost_usd"] == 0.42
    assert report["failed"] is False
