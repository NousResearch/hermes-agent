from __future__ import annotations

import importlib.util
import sqlite3
import time
from pathlib import Path

from agent.self_improvement_audit import build_self_improvement_audit_context


def test_build_self_improvement_audit_context_reads_local_artifacts(tmp_path):
    home = tmp_path
    (home / "config.yaml").write_text(
        """
memory:
  provider: holographic
  nudge_interval: 3
  flush_min_turns: 2
self_improvement:
  enabled: true
  deterministic_triggers: true
  recent_learning_overlay: true
  audit_lookback_hours: 24
  audit_stale_skill_days: 1
skills:
  creation_nudge_interval: 5
""".strip()
    )

    now = time.time()
    conn = sqlite3.connect(home / "state.db")
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            started_at REAL,
            message_count INTEGER,
            tool_call_count INTEGER
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL
        );
        """
    )
    conn.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
        ("s1", "discord", now - 60, 4, 2),
    )
    conn.execute(
        "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("s1", "tool", "Traceback: sample failure", now - 30),
    )
    conn.commit()
    conn.close()

    logs = home / "logs"
    logs.mkdir()
    (logs / "errors.log").write_text("Traceback\nModuleNotFoundError: No module named 'x'\n")

    skill = home / "skills" / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: demo\n---\n")
    old = now - 3 * 86400
    skill.touch()
    import os

    os.utime(skill, (old, old))

    context = build_self_improvement_audit_context(home)

    assert "# Self-improvement audit context" in context
    assert "memory.provider: holographic" in context
    assert "sessions: 1" in context
    assert "messages: 4" in context
    assert "tool_calls: 2" in context
    assert "error_like_messages: 1" in context
    assert "ModuleNotFoundError" in context
    assert "demo:" in context


def test_cron_script_imports_audit_module():
    script = Path.home() / ".hermes" / "scripts" / "self_improvement_audit_context.py"
    assert script.exists(), "self-improvement cron pre-run script is missing"
    spec = importlib.util.spec_from_file_location("self_improvement_audit_context", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
