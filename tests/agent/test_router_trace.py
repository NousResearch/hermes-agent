"""Router trace persistence tests: record shape, off-by-default no-write."""

import json

import agent.router_trace as router_trace
from agent.router_trace import save_router_turn
from agent.usage_pricing import CanonicalUsage


def _turn_kwargs():
    return dict(
        session_id="sess-1",
        preset_name="default",
        platform="whatsapp",
        classifier={
            "provider": "openai-codex",
            "model": "gpt-5.5",
            "input_messages": [{"role": "user", "content": "hoi"}],
            "raw_output": "simple",
            "verdict": "simple",
            "failed": False,
            "error": None,
            "usage": CanonicalUsage(input_tokens=100, output_tokens=1),
            "cost_usd": 0.0001,
        },
        route={"tier": "simple", "provider": "lmstudio", "model": "google/gemma-4-e4b"},
        fallback_events=[
            {"from": "lmstudio:google/gemma-4-e4b", "to": "lmstudio:qwen/qwen3-4b-thinking-2507",
             "error_class": "connection", "error": "down"},
        ],
        acting_label="lmstudio:qwen/qwen3-4b-thinking-2507",
        acting_model="qwen/qwen3-4b-thinking-2507",
        acting_provider="lmstudio",
        acting_output="hoi!",
        acting_streamed=False,
    )


def test_save_router_turn_writes_record(tmp_path, monkeypatch):
    monkeypatch.setattr(router_trace, "_traces_enabled_and_dir", lambda: tmp_path)
    save_router_turn(**_turn_kwargs())

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    assert files[0].name == "sess-1.jsonl"
    record = json.loads(files[0].read_text().strip())
    assert record["preset"] == "default"
    assert record["platform"] == "whatsapp"
    assert record["classifier"]["verdict"] == "simple"
    assert record["classifier"]["usage"]["input_tokens"] == 100
    assert record["route"]["tier"] == "simple"
    assert len(record["fallbacks"]) == 1
    assert record["acting"]["model"] == "qwen/qwen3-4b-thinking-2507"
    assert record["acting"]["output_location"] == "inline"


def test_save_router_turn_streamed_output_locations(tmp_path, monkeypatch):
    monkeypatch.setattr(router_trace, "_traces_enabled_and_dir", lambda: tmp_path)
    kwargs = _turn_kwargs()
    kwargs["acting_streamed"] = True
    save_router_turn(**kwargs)
    kwargs["acting_output"] = None
    save_router_turn(**kwargs)

    lines = (tmp_path / "sess-1.jsonl").read_text().strip().splitlines()
    first, second = (json.loads(line) for line in lines)
    assert first["acting"]["output_location"] == "inline_from_stream"
    assert second["acting"]["output_location"] == "assistant_message_in_session_db"


def test_save_router_turn_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(router_trace, "_traces_enabled_and_dir", lambda: None)
    save_router_turn(**_turn_kwargs())
    # No trace JSONL written anywhere under tmp_path (the repo conftest may
    # create unrelated scaffolding dirs there, so scope the check to *.jsonl).
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_traces_enabled_reads_router_config(tmp_path, monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: {"router": {"save_traces": False}})
    assert router_trace._traces_enabled_and_dir() is None

    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {"router": {"save_traces": True, "trace_dir": str(tmp_path / "custom")}},
    )
    assert router_trace._traces_enabled_and_dir() == tmp_path / "custom"
