from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def doctor_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    skills_root = hermes_home / "skills"
    memories.mkdir(parents=True)
    skills_root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Small configured limits so the fixture can exercise warning thresholds easily.
    (hermes_home / "config.yaml").write_text(
        "memory:\n  memory_char_limit: 100\n  user_char_limit: 100\n",
        encoding="utf-8",
    )

    (memories / "MEMORY.md").write_text("memory note\n", encoding="utf-8")
    (memories / "USER.md").write_text("u" * 86, encoding="utf-8")

    for skill_name in ("alpha", "beta"):
        skill_dir = skills_root / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"---\nname: {skill_name}\n---\n", encoding="utf-8")

    def ts(days_ago: int) -> str:
        return (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")

    memory_db = hermes_home / "memory_store.db"
    conn = sqlite3.connect(memory_db)
    conn.executescript(
        """
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hrr_vector BLOB
        );
        CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            entity_type TEXT DEFAULT 'unknown',
            aliases TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE fact_entities (
            fact_id INTEGER,
            entity_id INTEGER,
            PRIMARY KEY (fact_id, entity_id)
        );
        """
    )
    conn.executemany(
        "INSERT INTO facts(content, category, tags, trust_score, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("ordinary preference about documentation", "general", "", 0.9, ts(0), ts(0)),
            ("api token sk-abcdefghijklmnopqrstuvwxyz1234", "general", "", 0.5, ts(0), ts(0)),
            ("low confidence note", "general", "", 0.2, ts(0), ts(0)),
            ("operational note", "general", "current,path,enabled", 0.6, ts(10), ts(10)),
            ("current service endpoint version", "general", "", 0.7, ts(40), ts(40)),
            ("legacy provider config note", "general", "provider,config,active", 0.8, ts(100), ts(100)),
        ],
    )
    conn.executemany(
        "INSERT INTO entities(name, entity_type, aliases) VALUES (?, ?, ?)",
        [("DocEntity", "project", ""), ("UserEntity", "person", "")],
    )
    conn.executemany(
        "INSERT INTO fact_entities(fact_id, entity_id) VALUES (?, ?)",
        [(1, 1), (2, 1), (3, 1), (4, 2), (5, 2)],
    )
    conn.commit()
    conn.close()

    state_db = hermes_home / "state.db"
    conn = sqlite3.connect(state_db)
    conn.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT
        );
        CREATE TABLE messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO sessions(session_id, source, title) VALUES (?, ?, ?)",
        [("s1", "cli", "one"), ("s2", "cli", "two")],
    )
    conn.executemany(
        "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
        [
            ("s1", "user", "hello"),
            ("s1", "assistant", "hi"),
            ("s2", "user", "ping"),
            ("s2", "assistant", "pong"),
            ("s2", "user", "more"),
        ],
    )
    conn.commit()
    conn.close()

    return hermes_home


def test_memory_doctor_builds_read_only_inventory(doctor_env):
    from hermes_cli.memory_doctor import build_memory_doctor_report

    report = build_memory_doctor_report()

    assert report["mode"] == "dry-run"
    assert report["memory"]["path"].endswith("/memories/MEMORY.md")
    assert report["memory"]["size_bytes"] == len("memory note\n".encode("utf-8"))
    assert report["memory"]["status"] == "ok"
    assert report["user"]["status"] == "warning"
    assert report["fact_store"]["facts"] == 6
    assert report["fact_store"]["entities"] == 2
    assert report["fact_store"]["facts_without_entity"] == 1
    assert report["fact_store"]["low_trust_facts"]["count"] == 1
    assert report["fact_store"]["sensitive_facts"]
    assert report["fact_store"]["top_entities"]["items"][0]["entity"] == "DocEntity"
    assert report["fact_store"]["top_entities"]["items"][0]["fact_count"] == 3
    buckets = {item["bucket"]: item for item in report["fact_store"]["stale_mutable_facts"]["buckets"]}
    assert buckets["7-29d"]["count"] == 1
    assert buckets["30-89d"]["count"] == 1
    assert buckets["90+d"]["count"] == 1
    assert buckets["7-29d"]["items"][0]["keywords"]
    assert buckets["30-89d"]["items"][0]["age_days"] >= 30
    assert buckets["90+d"]["items"][0]["age_days"] >= 90
    assert report["session_search"]["sessions"] == 2
    assert report["session_search"]["messages"] == 5
    assert report["skills"]["skill_docs"] == 2


def test_memory_doctor_human_output_redacts_sensitive_content(doctor_env, capsys):
    from hermes_cli.memory_doctor import cmd_memory_doctor

    report = cmd_memory_doctor(argparse.Namespace(dry_run=True, json=False))
    out = capsys.readouterr().out

    assert report["overall_status"] in {"warning", "critical"}
    assert "sk-abcdefghijklmnopqrstuvwxyz1234" not in out
    assert "api token" not in out.lower()
    assert "fact_id=" in out
    assert "possible_sensitive_facts" in out
    assert "top_entities" in out
    assert "stale_mutable_facts" in out
    assert "current service endpoint version" not in out


def test_memory_doctor_main_dispatch_json(doctor_env, monkeypatch, capsys):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(sys, "argv", ["hermes", "memory", "doctor", "--dry-run", "--json"])
    main_mod.main()
    out = capsys.readouterr().out

    payload = json.loads(out)
    assert payload["mode"] == "dry-run"
    assert payload["memory"]["limit_bytes"] == 100
    assert payload["user"]["limit_bytes"] == 100
    assert payload["session_search"]["sessions"] == 2
    assert payload["skills"]["skill_docs"] == 2
    assert payload["fact_store"]["top_entities"]["items"][0]["entity"] == "DocEntity"
    assert payload["fact_store"]["stale_mutable_facts"]["buckets"][0]["count"] == 1
