"""Process-level regression coverage for exact CLI usage persistence."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

from hermes_state import SessionDB


EXPECTED_USAGE = {
    "input_tokens": 19,
    "output_tokens": 9,
    "cache_read_tokens": 5,
    "cache_write_tokens": 2,
    "reasoning_tokens": 3,
    "api_call_count": 1,
    "model": "gpt-5.6-sol",
    "billing_provider": "openai-codex",
}


def _create_v22_db_with_legacy_usage_key(db_path: Path) -> None:
    """Model a v22 DB whose task column was added without rebuilding its PK."""
    db = SessionDB(db_path=db_path)
    db.close()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            DROP TABLE session_model_usage;
            CREATE TABLE session_model_usage (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                billing_provider TEXT NOT NULL DEFAULT '',
                billing_base_url TEXT NOT NULL DEFAULT '',
                billing_mode TEXT NOT NULL DEFAULT '',
                api_call_count INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd REAL NOT NULL DEFAULT 0,
                actual_cost_usd REAL NOT NULL DEFAULT 0,
                cost_status TEXT,
                cost_source TEXT,
                first_seen REAL,
                last_seen REAL,
                task TEXT DEFAULT '',
                PRIMARY KEY (
                    session_id, model, billing_provider,
                    billing_base_url, billing_mode
                )
            );
            CREATE INDEX idx_session_model_usage_session
                ON session_model_usage(session_id);
            CREATE INDEX idx_session_model_usage_model
                ON session_model_usage(model);
            UPDATE schema_version SET version = 22;
            """
        )
        conn.commit()
    finally:
        conn.close()


def _run_single_query_child(hermes_home: Path, usage_expression: str) -> subprocess.CompletedProcess:
    child = textwrap.dedent(
        f"""
        from types import SimpleNamespace

        import cli
        import run_agent

        cli._run_state_db_auto_maintenance = lambda _db: None
        cli._run_checkpoint_auto_maintenance = lambda: None
        run_agent.get_tool_definitions = lambda **_kwargs: []
        run_agent.check_toolset_requirements = lambda: {{}}

        real_agent_factory = cli.AIAgent

        def deterministic_agent(*args, **kwargs):
            agent = real_agent_factory(*args, **kwargs)
            agent._cleanup_task_resources = lambda _task_id: None
            agent._save_trajectory = lambda *_args, **_kwargs: None
            response = SimpleNamespace(
                output=[SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="OK")],
                )],
                usage={usage_expression},
                status="completed",
                model="gpt-5.6-sol",
            )
            agent._interruptible_api_call = lambda _api_kwargs: response
            return agent

        cli.AIAgent = deterministic_agent
        cli.main(
            query="Say OK",
            quiet=True,
            model="gpt-5.6-sol",
            provider="openai-codex",
            api_key="test-token",
            base_url="https://chatgpt.com/backend-api/codex",
            toolsets="terminal",
            max_turns=1,
            ignore_rules=True,
        )
        """
    )
    env = os.environ.copy()
    env.update(
        HERMES_HOME=str(hermes_home),
        HERMES_EXIT_WATCHDOG_S="0",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    env.pop("HERMES_KANBAN_TASK", None)
    env.pop("HERMES_KANBAN_GOAL_MODE", None)
    return subprocess.run(
        [sys.executable, "-c", child],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_single_query_exact_usage_survives_process_exit_on_legacy_v22_key(tmp_path):
    """Run the real one-shot CLI turn/finalizer, then reopen and read state.db."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    db_path = hermes_home / "state.db"
    _create_v22_db_with_legacy_usage_key(db_path)

    proc = _run_single_query_child(
        hermes_home,
        """SimpleNamespace(
            input_tokens=26,
            input_tokens_details=SimpleNamespace(
                cached_tokens=5,
                cache_write_tokens=2,
            ),
            output_tokens=9,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
            total_tokens=35,
        )""",
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        session = conn.execute(
            """SELECT input_tokens, output_tokens, cache_read_tokens,
                      cache_write_tokens, reasoning_tokens, api_call_count,
                      model, billing_provider
               FROM sessions ORDER BY started_at DESC LIMIT 1"""
        ).fetchone()
        assert session is not None
        session_id = conn.execute(
            "SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()["id"]
        model_usage = conn.execute(
            """SELECT input_tokens, output_tokens, cache_read_tokens,
                      cache_write_tokens, reasoning_tokens, api_call_count,
                      model, billing_provider
               FROM session_model_usage WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()

    assert dict(session) == EXPECTED_USAGE
    assert model_usage is not None
    assert dict(model_usage) == EXPECTED_USAGE


def test_single_query_without_usage_persists_api_call_across_process_exit(tmp_path):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()

    proc = _run_single_query_child(hermes_home, "None")
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout

    reopened = SessionDB(db_path=hermes_home / "state.db")
    try:
        session = reopened._conn.execute(
            "SELECT id, api_call_count, input_tokens, output_tokens, "
            "cache_read_tokens, cache_write_tokens, reasoning_tokens "
            "FROM sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        assert session is not None
        usage = reopened._conn.execute(
            "SELECT api_call_count, input_tokens, output_tokens, "
            "cache_read_tokens, cache_write_tokens, reasoning_tokens "
            "FROM session_model_usage WHERE session_id = ?",
            (session["id"],),
        ).fetchone()
    finally:
        reopened.close()

    assert tuple(session)[1:] == (1, 0, 0, 0, 0, 0)
    assert usage is not None
    assert tuple(usage) == (1, 0, 0, 0, 0, 0)
