"""Tests for the bundled operational LCM context engine."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from agent.context_engine import ContextEngine
from plugins.context_engine import discover_context_engines, load_context_engine

REQUIRED_TOOLS = {
    "operational_context_search",
    "operational_context_get_task",
    "operational_context_recent_decisions",
    "operational_context_artifacts",
    "operational_context_status",
}


def _make_home(tmp_path: Path) -> Path:
    home = tmp_path / "hermes-home"
    home.mkdir()
    os.environ["HERMES_HOME"] = str(home)
    return home


def _seed_kanban(home: Path) -> None:
    conn = sqlite3.connect(home / "kanban.db")
    conn.execute("create table tasks (id text primary key, title text, body text, status text, assignee text, result text, created_at text, completed_at text)")
    conn.execute("create table task_runs (id integer primary key autoincrement, task_id text, profile text, status text, outcome text, summary text, metadata text, error text, started_at integer, ended_at integer)")
    conn.execute("create table task_comments (id integer primary key autoincrement, task_id text, author text, body text, created_at integer)")
    conn.execute("create table task_events (id integer primary key autoincrement, task_id text, run_id integer, kind text, payload text, created_at integer)")
    conn.execute(
        "insert into tasks values (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "t_demo",
            "Implement operational context engine",
            "Decision: keep context local-only. Artifact /tmp/artifact.md sha256=abc123.",
            "done",
            "coder",
            "Result references report /tmp/report.md sha256=def456.",
            "1",
            "2",
        ),
    )
    conn.execute(
        "insert into task_runs(task_id, profile, status, outcome, summary, metadata, error, started_at, ended_at) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "t_demo",
            "coder",
            "done",
            "completed",
            "Run summary with decision record and artifact evidence.",
            json.dumps({
                "artifacts": ["/tmp/artifact.md"],
                "shas": {"report": "def456"},
                "decisions": ["approve local-only LCM"],
                "token": "fake-value-that-must-not-appear",
            }),
            None,
            1,
            2,
        ),
    )
    conn.execute(
        "insert into task_comments(task_id, author, body, created_at) values (?, ?, ?, ?)",
        ("t_demo", "reviewer", "Comment decision: preserve artifact path and hash abc123", 3),
    )
    conn.execute(
        "insert into task_events(task_id, run_id, kind, payload, created_at) values (?, ?, ?, ?, ?)",
        ("t_demo", 1, "completed", json.dumps({"summary": "event artifact sha256:feedbeef"}), 4),
    )
    conn.commit()
    conn.close()


def _seed_sessions(home: Path) -> None:
    conn = sqlite3.connect(home / "state.db")
    conn.execute("create table sessions (id text primary key, source text, user_id text, model text, model_config text, system_prompt text, parent_session_id text, started_at real, ended_at real, end_reason text, message_count integer, tool_call_count integer, input_tokens integer, output_tokens integer, title text, api_call_count integer)")
    unsafe_title = "gh" + "p_" + ("Z" * 36)
    conn.execute(
        "insert into sessions values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("s_demo", "cli", "u", "gpt-test", "{}", "SYSTEM_PROMPT_MUST_NOT_APPEAR", None, 1, 2, "complete", 5, 2, 100, 50, "Session summary: decision to use structured context", 1),
    )
    conn.execute(
        "insert into sessions values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("s_bad", "cli", "u", "gpt-test", "{}", "not indexed", None, 3, 4, "complete", 5, 2, 100, 50, unsafe_title, 1),
    )
    conn.commit()
    conn.close()
    handoffs = home / "memories" / "handoffs"
    handoffs.mkdir(parents=True)
    (handoffs / "handoff-demo.md").write_text("# Handoff\nDecision: keep local-only structured context. Artifact /tmp/artifact.md sha256:def456\n")
    daily = home / "memories" / "daily"
    daily.mkdir(parents=True)
    (daily / "2026-05-08.md").write_text("## Decision\nUse operational LCM for structured context retrieval.\n")
    return None


def _engine(tmp_path: Path):
    home = _make_home(tmp_path)
    _seed_kanban(home)
    _seed_sessions(home)
    engine = load_context_engine("operational_lcm")
    assert engine is not None
    engine.on_session_start("s1", hermes_home=str(home), platform="test", model="test")
    return engine


def test_plugin_loads_as_context_engine_and_declares_required_tools():
    discovered = {name for name, _desc, available in discover_context_engines() if available}
    assert "operational_lcm" in discovered

    engine = load_context_engine("operational_lcm")

    assert isinstance(engine, ContextEngine)
    assert engine.name == "operational_lcm"
    assert {schema["name"] for schema in engine.get_tool_schemas()} == REQUIRED_TOOLS


def test_structured_ingestion_search_task_decisions_artifacts_and_status(tmp_path: Path):
    engine = _engine(tmp_path)
    engine.compress([
        {"role": "system", "content": "system"},
        {"role": "user", "content": "please finish t_demo and remember artifact sha256:feedbeef"},
        {"role": "assistant", "content": "Decision: structured operational memory is required. Artifact /tmp/final.md sha256:feedbeef"},
    ], current_tokens=999999)

    search = json.loads(engine.handle_tool_call("operational_context_search", {"query": "artifact"}))
    assert search["ok"] is True
    assert any("artifact" in item["text"].lower() for item in search["results"])

    task = json.loads(engine.handle_tool_call("operational_context_get_task", {"task_id": "t_demo"}))
    assert task["ok"] is True
    assert task["task"]["task_id"] == "t_demo"
    assert "abc123" in task["task"]["text"]

    decisions = json.loads(engine.handle_tool_call("operational_context_recent_decisions", {"limit": 5}))
    assert decisions["ok"] is True
    assert any("structured" in item["text"].lower() or "local-only" in item["text"].lower() for item in decisions["decisions"])

    artifacts = json.loads(engine.handle_tool_call("operational_context_artifacts", {"query": "feedbeef"}))
    assert artifacts["ok"] is True
    assert any("feedbeef" in item["text"] for item in artifacts["artifacts"])

    status = json.loads(engine.handle_tool_call("operational_context_status", {}))
    assert status["ok"] is True
    assert status["status"]["engine"] == "operational_lcm"
    assert status["status"]["local_only"] is True
    assert set(status["status"]["tools"]) == REQUIRED_TOOLS


def test_sanitization_skips_secret_shaped_values_and_raw_system_prompts(tmp_path: Path):
    engine = _engine(tmp_path)
    pat = "gh" + "p_" + ("A" * 36)
    fine = "github" + "_pat_" + ("B" * 30)
    cloud = "AK" + "IA" + ("C" * 16)
    jwt = "eyJ" + ("D" * 16) + "." + ("E" * 16) + "." + ("F" * 16)
    harmless_hash = "a" * 64
    engine.compress([
        {"role": "assistant", "content": "Decision: harmless artifact sha256:" + harmless_hash},
        {"role": "assistant", "content": "Decision: standalone marker " + pat},
        {"role": "assistant", "content": "Decision: fine marker " + fine},
        {"role": "assistant", "content": "Decision: cloud marker " + cloud},
        {"role": "assistant", "content": "Decision: jwt marker " + jwt},
    ], current_tokens=999999)

    for query in (pat, fine, cloud, jwt, "SYSTEM_PROMPT_MUST_NOT_APPEAR"):
        hit = json.loads(engine.handle_tool_call("operational_context_search", {"query": query}))
        assert hit["ok"] is True
        assert hit["results"] == []

    kept = json.loads(engine.handle_tool_call("operational_context_search", {"query": harmless_hash[:12]}))
    assert kept["ok"] is True
    assert any(harmless_hash[:12] in item["text"] for item in kept["results"])


def test_live_message_tool_context_and_compress_return_valid_messages(tmp_path: Path):
    engine = _engine(tmp_path)
    result = engine.compress([
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Secret-like value must not be indexed. Artifact /tmp/x sha256:abc123"},
        {"role": "assistant", "content": "Decision: keep hashes visible."},
    ], current_tokens=999999)

    assert isinstance(result, list)
    assert all("role" in message and "content" in message for message in result)
    assert any("Operational structured context" in str(message.get("content", "")) for message in result)
    assert engine.compression_count == 1

    live = json.loads(engine.handle_tool_call("operational_context_search", {"query": "live decision"}, messages=[
        {"role": "assistant", "content": "Decision: live decision should be indexed by tool-call message context."}
    ]))
    assert live["ok"] is True
    assert any("live decision" in item["text"].lower() for item in live["results"])
