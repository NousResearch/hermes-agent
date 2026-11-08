"""The post-mortem forensics run end-to-end on a synthetic state.db and report the run population correctly.

Guards the harness itself (it lives in evals/, outside the normal import graph): the root is discovered,
a compression-rollover child is excluded, pricing is fitted, and each lane writes its JSON without error.
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

EVALS = Path(__file__).resolve().parents[2]


def _mk_db(path: Path) -> None:
    c = sqlite3.connect(path)
    c.executescript("""
    CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_session_id TEXT, source TEXT, started_at REAL, ended_at REAL,
        api_call_count INTEGER, input_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
        output_tokens INTEGER, estimated_cost_usd REAL, system_prompt_hash TEXT);
    CREATE TABLE system_prompts(hash TEXT PRIMARY KEY, prompt TEXT);
    CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, tool_calls TEXT,
        tool_name TEXT, reasoning TEXT, timestamp REAL);
    CREATE TABLE state_meta(key TEXT PRIMARY KEY, value TEXT);
    """)
    t0 = time.time() - 4000
    c.execute("INSERT INTO system_prompts VALUES ('h1', ?)", ("x" * 35_000,))

    def sess(sid, parent, source, start, end, calls, cr, cw, out):
        cost = cr * 0.2e-6 + cw * 10e-6 + out * 40e-6
        c.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (sid, parent, source, start, end, calls, 0, cr, cw, out, cost, "h1"))

    sess("root", None, "cli", t0, t0 + 3600, 40, 2_000_000, 300_000, 40_000)
    for i in range(12):  # children at depth 1
        sess(f"c{i}", "root", "subagent", t0 + 10 + i, t0 + 600 + 60 * i, 20, 1_000_000, 200_000, 20_000)
    sess("g0", "c0", "subagent", t0 + 20, t0 + 500, 10, 500_000, 100_000, 10_000)  # depth 2
    sess("rollover", "root", "cli", t0 + 3601, t0 + 7200, 30, 9_000_000, 900_000, 90_000)  # excluded
    sess("unrelated", None, "telegram", t0, t0 + 100, 1, 1000, 100, 10)
    # messages: a delegate_task timeout in c0, a truncated batch block in root, a hardline block, a nudge
    c.execute("INSERT INTO messages(session_id,role,content,tool_name,timestamp) VALUES ('c0','tool',\"Error executing tool 'delegate_task': timed out after 420.0s\",'delegate_task',?)", (t0 + 100,))
    c.execute("INSERT INTO messages(session_id,role,content,tool_calls,timestamp) VALUES ('c0','assistant','','[{\"function\":{\"name\":\"terminal\",\"arguments\":\"{\\\\\"command\\\\\": \\\\\"sleep 600\\\\\"}\"}}]',?)", (t0 + 200,))
    c.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES ('root','user','[ASYNC DELEGATION BATCH COMPLETE — d]\\n--- ✓ TASK 1/2 ...\\n[SUMMARY TRUNCATED]\\n--- ✓ TASK 2/2 ...',?)", (t0 + 700,))
    c.execute("INSERT INTO messages(session_id,role,content,tool_name,timestamp) VALUES ('c1','tool','BLOCKED (hardline): command parser limit or malformed executable payload','terminal',?)", (t0 + 300,))
    c.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES ('root','assistant','Waiting on 3 batches; nothing to dispatch.',?)", (t0 + 800,))
    c.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES ('root','user','[Continuing toward your standing goal]\\nGoal: x',?)", (t0 + 810,))
    c.execute("INSERT INTO state_meta VALUES ('goal:root', ?)", (json.dumps({"status": "active", "waiting_on_session": "proc_x", "waiting_since": t0 + 900, "last_verdict": "wait", "turns_used": 3}),))
    c.commit(); c.close()


@pytest.fixture
def harness_path(monkeypatch):
    monkeypatch.syspath_prepend(str(EVALS.parent))
    return EVALS


def test_run_population_excludes_rollover_and_unrelated_sessions(tmp_path, harness_path):
    from evals.postmortem.forensics.common import Run
    db = tmp_path / "state.db"; _mk_db(db)
    run = Run.open(str(db), out=str(tmp_path / "out"))
    assert run.root == "root"
    assert set(run.in_run) == {"root", *{f"c{i}" for i in range(12)}, "g0"}
    assert "rollover" not in run.in_run and "unrelated" not in run.in_run
    s = run.summary()
    assert s["by_depth"] == {0: 1, 1: 12, 2: 1}
    assert abs(s["fitted_price_per_million"]["cache_write_tokens"] - 10.0) < 0.01
    assert abs(s["cost_usd"] - sum(run.cost(x) for x in run.in_run)) < 1e-6


def test_every_lane_runs_and_writes_its_report(tmp_path, harness_path):
    from evals.postmortem.forensics import delegation, goal_loop, tokens, tools
    db = tmp_path / "state.db"; _mk_db(db)
    out = tmp_path / "out"
    for lane in (tokens, delegation, tools, goal_loop):
        assert lane.main(["--db", str(db), "--out", str(out)]) == 0
    assert json.loads((out / "delegation.json").read_text(encoding="utf-8"))["observed"]["delegate_task_timeouts"] == 1
    assert json.loads((out / "tools.json").read_text(encoding="utf-8"))["observed"]["hardline_blocks_malformed_class"] == 1
    g = json.loads((out / "goal_loop.json").read_text(encoding="utf-8"))["observed"]
    assert g["nudges"] == 1 and g["nudges_within_180s_of_a_waiting_turn"] == 1
    assert (out / "tokens.json").exists()


def test_runner_lists_a_probe_per_pr(harness_path):
    from evals.postmortem import run as runner
    prs = {p[4] for p in runner.PROBES}
    assert {"#103492", "#103513", "#103549", "#103476", "#103551", "#103526", "#103534"} <= prs
    assert sys.version_info >= (3, 10)
