"""Tests for the bundled tool-audit skill (metadata + workflow).

Metadata: the skill must be discoverable by the real bundled-skill scanner
(SKILL.md, not DESCRIPTION.md) and honor the skill-authoring hardlines.

Workflow: scripts/tool_audit.py must correlate tool calls with results by
tool_call_id against a real (fixture) state.db, count only provable errors,
open the store read-only, and never report latency.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "autonomous-ai-agents" / "tool-audit"
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPT_PATH = SKILL_DIR / "scripts" / "tool_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("tool_audit_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frontmatter() -> dict:
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "frontmatter must open the file"
    body = text.split("---", 2)
    assert len(body) == 3, "frontmatter must be closed"
    fields = {}
    for line in body[1].splitlines():
        m = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
        if m:
            fields[m.group(1)] = m.group(2).strip().strip("\"'")
    return fields


# ── Metadata ────────────────────────────────────────────────────────────────


def test_skill_md_exists_where_discovery_scans():
    assert SKILL_MD.is_file(), "skill must ship a SKILL.md (DESCRIPTION.md is not discoverable)"
    assert not (SKILL_DIR / "DESCRIPTION.md").exists()


def test_bundled_discovery_finds_tool_audit():
    from tools.skills_sync import _discover_bundled_skills

    found = {name: path for name, path in _discover_bundled_skills(REPO_ROOT / "skills")}
    assert "tool-audit" in found
    assert found["tool-audit"] == SKILL_DIR


def test_frontmatter_hardlines():
    fields = _frontmatter()
    assert fields["name"] == "tool-audit"
    description = fields["description"]
    assert description, "description is required"
    assert len(description) <= 60, f"description must be <=60 chars, got {len(description)}"
    assert description.endswith("."), "description must end with a period"
    assert "tool-audit" not in description.lower(), "description must not repeat the skill name"


def test_body_uses_modern_section_order():
    text = SKILL_MD.read_text(encoding="utf-8")
    sections = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [text.index(section) for section in sections]
    assert positions == sorted(positions), "sections must appear in the modern order"


def test_no_legacy_jsonl_or_latency_promises():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "agents/*/sessions" not in text, "legacy JSONL session path must not be referenced"
    assert "state.db" in text, "audit must be grounded in the canonical store"
    assert "Avg Latency" not in text and "average latency" not in text.lower()
    assert "Success Rate" not in text


def test_script_is_referenced_and_stdlib_only():
    assert SCRIPT_PATH.is_file()
    assert "scripts/tool_audit.py" in SKILL_MD.read_text(encoding="utf-8")
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for banned in ("requests", "httpx", "aiohttp"):
        assert f"import {banned}" not in source


# ── Workflow fixtures ───────────────────────────────────────────────────────


def _make_store(tmp_path: Path) -> Path:
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT,
            started_at REAL NOT NULL,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0
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
            active INTEGER NOT NULL DEFAULT 1
        );
        """
    )

    def call(call_id: str, name: str) -> dict:
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}

    conn.execute(
        "INSERT INTO sessions (id, source, title, started_at, message_count, tool_call_count)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        ("sess-aaa111", "cli", "audit me", 1000.0, 8, 4),
    )
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
        ("sess-bbb222", "cli", 2000.0),
    )

    rows = [
        # (role, content, tool_call_id, tool_calls, tool_name, ts, active)
        ("user", "please work", None, None, None, 1001.0, 1),
        (
            "assistant",
            None,
            None,
            json.dumps([call("call_1", "terminal"), call("call_2", "read_file"), call("call_3", "terminal")]),
            None,
            1002.0,
            1,
        ),
        ("tool", json.dumps({"error": "boom exploded"}), "call_1", None, "terminal", 1003.0, 1),
        ("tool", "file contents here", "call_2", None, "read_file", 1004.0, 1),
        # call_3 stays orphaned: no result row.
        # Uncorrelated result — its call was compacted away.
        ("tool", "stale", "call_ghost", None, "terminal", 1005.0, 1),
        # Structured content stored via the SessionDB sentinel encoding.
        (
            "assistant",
            None,
            None,
            json.dumps([call("call_4", "web_search")]),
            None,
            1006.0,
            1,
        ),
        ("tool", "\x00json:" + json.dumps({"error": "quota"}), "call_4", None, "web_search", 1007.0, 1),
        # Plain-text failure prose must NOT count as a provable error.
        (
            "assistant",
            None,
            None,
            json.dumps([call("call_5", "terminal")]),
            None,
            1008.0,
            1,
        ),
        ("tool", "Error: something vague happened", "call_5", None, "terminal", 1009.0, 1),
        # Soft-deleted (rewound) rows are excluded from the audit.
        ("assistant", None, None, json.dumps([call("call_6", "terminal")]), None, 1010.0, 0),
    ]
    conn.executemany(
        "INSERT INTO messages (session_id, role, content, tool_call_id, tool_calls, tool_name, timestamp, active)"
        " VALUES ('sess-aaa111', ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


def _by_name(report: dict) -> dict:
    return {tool["name"]: tool for tool in report["tools"]}


# ── Workflow ────────────────────────────────────────────────────────────────


def test_audit_correlates_calls_and_results_by_id(tmp_path):
    mod = load_module()
    conn = mod.open_db_readonly(_make_store(tmp_path))
    report = mod.audit_messages(mod.fetch_messages(conn, "sess-aaa111"))
    conn.close()

    tools = _by_name(report)
    terminal = tools["terminal"]
    assert terminal["calls"] == 3  # call_6 is inactive, not counted
    assert terminal["results"] == 2  # call_1 + call_5; ghost result not attributed
    assert terminal["orphaned"] == 1  # call_3
    assert terminal["errors"] == 1  # only the JSON {"error": ...} payload
    assert terminal["sample_errors"] == ["boom exploded"]

    assert tools["read_file"] == {
        "name": "read_file", "calls": 1, "results": 1, "orphaned": 0, "errors": 0, "sample_errors": [],
    }
    assert tools["web_search"]["errors"] == 1  # sentinel-encoded structured payload
    assert report["uncorrelated_results"] == 1
    assert report["totals"] == {"calls": 5, "results": 4, "orphaned": 1, "errors": 2}


def test_provable_error_rule_matches_runtime_observer():
    mod = load_module()
    assert mod.result_error(json.dumps({"error": "x"})) == "x"
    assert mod.result_error(json.dumps({"error": ""})) is None  # falsy — same as observer
    assert mod.result_error(json.dumps({"ok": True})) is None
    assert mod.result_error("Error: not json") is None
    assert mod.result_error(None) is None


def test_main_json_output_and_no_latency_fields(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    mod = load_module()
    db_path = _make_store(tmp_path)

    exit_code = mod.main(["--db", str(db_path), "--session", "sess-aaa111", "--json"])
    assert exit_code == 0

    rendered = json.loads(capsys.readouterr().out)
    assert rendered["session"]["id"] == "sess-aaa111"
    assert rendered["session"]["source"] == "cli"
    assert rendered["totals"]["calls"] == 5
    flat = json.dumps(rendered).lower()
    assert "latency" not in flat and "duration_ms" not in flat and "success_rate" not in flat


def test_session_resolution_prefix_env_and_latest(tmp_path, monkeypatch):
    mod = load_module()
    conn = mod.open_db_readonly(_make_store(tmp_path))

    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    assert mod.resolve_session(conn, "sess-aaa") == "sess-aaa111"  # unique prefix
    assert mod.resolve_session(conn, None) == "sess-bbb222"  # latest started_at

    monkeypatch.setenv("HERMES_SESSION_ID", "sess-aaa111")
    assert mod.resolve_session(conn, None) == "sess-aaa111"  # agent's own session wins

    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    try:
        mod.resolve_session(conn, "sess-")  # ambiguous
        raised = False
    except LookupError:
        raised = True
    assert raised
    conn.close()


def test_store_is_opened_readonly_and_missing_db_fails_cleanly(tmp_path, capsys):
    mod = load_module()
    db_path = _make_store(tmp_path)

    real_connect = sqlite3.connect
    seen = {}

    def spy(dsn, *args, **kwargs):
        seen["dsn"], seen["uri"] = dsn, kwargs.get("uri")
        return real_connect(dsn, *args, **kwargs)

    with patch.object(mod.sqlite3, "connect", side_effect=spy):
        assert mod.main(["--db", str(db_path), "--session", "sess-aaa111"]) == 0
    assert seen["uri"] is True and "mode=ro" in seen["dsn"]
    capsys.readouterr()

    assert mod.main(["--db", str(tmp_path / "nope.db")]) == 2
    assert "not found" in capsys.readouterr().err
