import json
import sqlite3
from pathlib import Path

from tools.agent_trace_feedback import (
    EvalResult,
    evaluate_traces,
    extract_traces,
    main,
    render_feedback_report,
    run_pipeline,
    write_jsonl,
)


def _make_state_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            user_id TEXT,
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            billing_provider TEXT,
            billing_base_url TEXT,
            billing_mode TEXT,
            estimated_cost_usd REAL,
            actual_cost_usd REAL,
            cost_status TEXT,
            cost_source TEXT,
            pricing_version TEXT,
            title TEXT,
            api_call_count INTEGER DEFAULT 0,
            handoff_state TEXT,
            handoff_platform TEXT,
            handoff_error TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            token_count INTEGER,
            finish_reason TEXT,
            reasoning TEXT,
            reasoning_content TEXT,
            reasoning_details TEXT,
            codex_reasoning_items TEXT,
            codex_message_items TEXT,
            platform_message_id TEXT,
            observed INTEGER DEFAULT 0
        );
        """
    )
    con.execute(
        "INSERT INTO sessions (id, source, user_id, model, model_config, started_at, tool_call_count, billing_provider, title) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "s1",
            "telegram",
            "u1",
            "gpt-5.5",
            json.dumps({"provider": "openai-codex"}),
            100.0,
            2,
            "openai-codex",
            "tool-required prompt",
        ),
    )
    messages = [
        ("s1", "user", "What time is it?", None, None, 101.0),
        (
            "s1",
            "assistant",
            "Checking now.",
            json.dumps(
                [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "terminal",
                            "arguments": json.dumps({"command": "date"}),
                        },
                    },
                    {
                        "id": "call_2",
                        "function": {
                            "name": "mcp_gbrain_put_page",
                            "arguments": json.dumps({"slug": "gbrain/hermes/reports/ok"}),
                        },
                    },
                ]
            ),
            None,
            102.0,
        ),
        ("s1", "tool", "Mon May 25 12:00:00 EDT 2026", None, "terminal", 103.0),
        ("s1", "assistant", "It is 12:00 EDT.", None, None, 104.0),
    ]
    con.executemany(
        "INSERT INTO messages (session_id, role, content, tool_calls, tool_name, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        messages,
    )
    con.commit()
    con.close()


def test_extract_traces_from_state_db_includes_tool_calls_and_paths(tmp_path):
    db_path = tmp_path / "state.db"
    _make_state_db(db_path)

    traces = extract_traces(db_path)

    assert len(traces) == 1
    trace = traces[0]
    assert trace["trace_id"] == "s1"
    assert trace["source"] == "telegram"
    assert trace["model"] == "gpt-5.5"
    assert trace["model_provider"] == "openai-codex"
    assert trace["user_prompt_summary"] == "What time is it?"
    assert trace["tool_call_count"] == 2
    assert trace["tool_calls"] == ["terminal", "mcp_gbrain_put_page"]
    assert trace["paths_written"] == ["gbrain/hermes/reports/ok"]
    assert trace["final_response_summary"] == "It is 12:00 EDT."
    assert trace["selected_skills"] == []
    assert trace["retry_count"] == 0
    assert trace["retry_tools"] == []


def test_extract_traces_includes_selected_skills_corrections_and_retries(tmp_path):
    db_path = tmp_path / "state.db"
    _make_state_db(db_path)
    con = sqlite3.connect(db_path)
    skill_call = json.dumps([
        {"function": {"name": "skill_view", "arguments": json.dumps({"name": "writing-plans"})}},
        {"function": {"name": "terminal", "arguments": json.dumps({"command": "false"})}},
        {"function": {"name": "terminal", "arguments": json.dumps({"command": "false"})}},
    ])
    con.execute(
        "UPDATE messages SET tool_calls = ? WHERE session_id = 's1' AND role = 'assistant' AND timestamp = 102.0",
        (skill_call,),
    )
    con.execute(
        "INSERT INTO messages (session_id, role, content, tool_calls, tool_name, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        ("s1", "tool", "ERROR: command failed", None, "terminal", 103.5),
    )
    con.execute(
        "INSERT INTO messages (session_id, role, content, tool_calls, tool_name, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        ("s1", "user", "You missed the actual current timezone.", None, None, 105.0),
    )
    con.commit()
    con.close()

    trace = extract_traces(db_path)[0]

    assert trace["selected_skills"] == ["writing-plans"]
    assert trace["retry_count"] == 1
    assert trace["retry_tools"] == ["terminal"]
    assert trace["user_followup_or_correction"] == "You missed the actual current timezone."


def test_evaluate_traces_passes_tool_required_prompt_when_tool_was_used(tmp_path):
    db_path = tmp_path / "state.db"
    _make_state_db(db_path)
    traces = extract_traces(db_path)

    results = evaluate_traces(traces)

    result = next(r for r in results if r.rule_id == "tool_required_question_used_tools")
    assert result.status == "pass"
    assert result.trace_id == "s1"


def test_evaluate_traces_flags_canonical_gbrain_write():
    traces = [
        {
            "trace_id": "bad1",
            "session_id": "bad1",
            "user_prompt_summary": "write report",
            "tool_calls": ["mcp_gbrain_put_page"],
            "tool_call_details": [{"name": "mcp_gbrain_put_page", "arguments": {"slug": "entities/projects/openclaw-work-queue"}}],
            "paths_written": ["entities/projects/openclaw-work-queue"],
            "errors": [],
        }
    ]

    results = evaluate_traces(traces)

    result = next(r for r in results if r.rule_id == "no_canonical_gbrain_writes_by_hermes")
    assert result.status == "fail"
    assert "entities/projects/openclaw-work-queue" in result.detail


def test_evaluate_traces_runs_issue_45_minimal_suite():
    trace = {
        "trace_id": "bad2",
        "session_id": "bad2",
        "source": "telegram",
        "user_prompt_summary": "What should I do about my family calendar and health?",
        "final_response_summary": "Here is a plan | bad | table | with no source citation.",
        "tool_calls": ["read_file", "write_file"],
        "tool_call_details": [
            {"name": "read_file", "arguments": {"path": "/Users/openclaw/kb/entities/projects/openclaw-work-queue.md"}},
            {"name": "write_file", "arguments": {"path": "/Users/openclaw/.hermes/profiles/sherlock/skills/foo/SKILL.md", "cross_profile": True}},
        ],
        "paths_read": ["/Users/openclaw/kb/entities/projects/openclaw-work-queue.md"],
        "paths_written": ["/Users/openclaw/.hermes/profiles/sherlock/skills/foo/SKILL.md"],
        "errors": [],
    }

    results = evaluate_traces([trace])
    by_rule = {r.rule_id: r for r in results}

    assert by_rule["citation_quality_for_gbrain_claims"].status == "fail"
    assert by_rule["gbrain_vs_filesystem_routing"].status == "fail"
    assert by_rule["telegram_formatting"].status == "fail"
    assert by_rule["domain_handoff_to_dobby"].status == "fail"
    assert by_rule["cross_profile_write_guard"].status == "fail"


def test_run_pipeline_writes_trace_eval_and_report_files(tmp_path):
    db_path = tmp_path / "state.db"
    workspace = tmp_path / "workspace"
    _make_state_db(db_path)

    outputs = run_pipeline(db_path=db_path, workspace_dir=workspace, date="2026-05-25")

    assert outputs["trace_path"].exists()
    assert outputs["eval_path"].exists()
    assert outputs["report_path"].exists()
    assert outputs["report_path"].read_text(encoding="utf-8").startswith("# Feedback-to-Plan Report — 2026-05-25")


def test_main_runs_pipeline_from_cli_args(tmp_path):
    db_path = tmp_path / "state.db"
    workspace = tmp_path / "workspace"
    _make_state_db(db_path)

    exit_code = main(["--db", str(db_path), "--workspace", str(workspace), "--date", "2026-05-25"])

    assert exit_code == 0
    assert (workspace / "traces" / "2026-05-25.jsonl").exists()
    assert (workspace / "evals" / "wq-021" / "results" / "2026-05-25.jsonl").exists()
    assert (workspace / "reports" / "feedback-to-plan" / "2026-05-25.md").exists()


def test_write_jsonl_and_render_feedback_report(tmp_path):
    results = [
        EvalResult(
            rule_id="no_canonical_gbrain_writes_by_hermes",
            trace_id="bad1",
            status="fail",
            detail="canonical write: entities/foo",
            suggested_delta="Route writes to gbrain/hermes/* or Dobby.",
        ),
        EvalResult(
            rule_id="no_canonical_gbrain_writes_by_hermes",
            trace_id="bad2",
            status="fail",
            detail="canonical write: entities/bar",
            suggested_delta="Route writes to gbrain/hermes/* or Dobby.",
        ),
    ]
    output_path = tmp_path / "results.jsonl"

    write_jsonl(output_path, [r.to_dict() for r in results])
    report = render_feedback_report(results, date="2026-05-25")

    assert json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])["status"] == "fail"
    assert "# Feedback-to-Plan Report — 2026-05-25" in report
    assert "no_canonical_gbrain_writes_by_hermes" in report
    assert "Route writes to gbrain/hermes/* or Dobby." in report
    assert "Count: 2" in report
    assert "Traces: `bad1`, `bad2`" in report
