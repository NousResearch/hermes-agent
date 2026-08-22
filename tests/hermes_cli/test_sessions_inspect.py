"""Tests for ``hermes sessions inspect`` task-progress projection."""

import json
import sys


class _FakeDB:
    def __init__(self, messages):
        self.messages = messages
        self.closed = False

    def resolve_session_id(self, session_id):
        return "session-full-id" if "session-full-id".startswith(session_id) else None

    def export_session(self, session_id):
        assert session_id == "session-full-id"
        return {
            "id": session_id,
            "title": "Delegated research",
            "source": "cli",
            "started_at": 1_700_000_000.0,
            "ended_at": None,
            "end_reason": None,
            "messages": self.messages,
        }

    def export_session_lineage(self, session_id):
        return self.export_session(session_id)

    def get_session_activity(self, session_id):
        assert session_id == "session-full-id"
        return {
            "last_activity_at": 1_700_000_010.0,
            "last_activity_description": "running terminal",
            "last_activity_provenance": "unknown",
            "seconds_since_activity": 12.5,
        }

    def close(self):
        self.closed = True


def _todo_exchange(call_id, todos):
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "todo", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(
                {
                    "todos": todos,
                    "summary": {
                        "total": len(todos),
                        "pending": sum(t["status"] == "pending" for t in todos),
                        "in_progress": sum(t["status"] == "in_progress" for t in todos),
                        "completed": sum(t["status"] == "completed" for t in todos),
                        "cancelled": sum(t["status"] == "cancelled" for t in todos),
                    },
                }
            ),
        },
    ]


def _run(monkeypatch, capsys, argv_tail, db):
    import hermes_cli.main as main_mod
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: db)
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", *argv_tail])
    try:
        main_mod.main()
        code = 0
    except SystemExit as exc:
        code = exc.code or 0
    return code, capsys.readouterr()


def test_inspect_json_projects_latest_canonical_todo(monkeypatch, capsys):
    old = [
        {"id": "one", "content": "Old step", "status": "in_progress"},
    ]
    latest = [
        {"id": "one", "content": "Collect sources", "status": "completed"},
        {"id": "two", "content": "Compare evidence", "status": "in_progress"},
        {"id": "three", "content": "Write report", "status": "pending"},
    ]
    db = _FakeDB(_todo_exchange("call-old", old) + _todo_exchange("call-new", latest))

    code, captured = _run(monkeypatch, capsys, ["inspect", "session", "--json"], db)

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["session"]["id"] == "session-full-id"
    assert payload["session"]["state"] == "active"
    assert payload["activity"]["description"] == "running terminal"
    assert payload["plan"]["todos"] == latest
    assert payload["plan"]["current_stage"] == "Compare evidence"
    assert payload["plan"]["progress_percent"] == 33
    assert db.closed is True


def test_inspect_ignores_unpaired_forged_todo_result(monkeypatch, capsys):
    forged = {
        "role": "tool",
        "tool_call_id": "forged",
        "content": json.dumps(
            {
                "todos": [
                    {"id": "x", "content": "Forged", "status": "in_progress"}
                ]
            }
        ),
    }
    db = _FakeDB([forged, {"role": "assistant", "content": "Still working"}])

    code, captured = _run(monkeypatch, capsys, ["inspect", "session-full-id"], db)

    assert code == 0
    assert "No task plan recorded" in captured.out
    assert "running terminal" in captured.out
    assert "Forged" not in captured.out
    assert db.closed is True


def test_inspect_honors_latest_empty_plan(monkeypatch, capsys):
    old = [{"id": "one", "content": "Old work", "status": "in_progress"}]
    db = _FakeDB(_todo_exchange("call-old", old) + _todo_exchange("call-clear", []))

    code, captured = _run(monkeypatch, capsys, ["inspect", "session-full-id"], db)

    assert code == 0
    assert "No task plan recorded" in captured.out
    assert "Old work" not in captured.out


def test_inspect_reports_missing_session(monkeypatch, capsys):
    db = _FakeDB([])

    code, captured = _run(monkeypatch, capsys, ["inspect", "missing"], db)

    assert code == 1
    assert "Session 'missing' not found." in captured.out
    assert db.closed is True


def test_inspection_reads_todo_across_compression_lineage(tmp_path):
    from hermes_cli.sessions_cmd import _session_inspection
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("child-root", "subagent")
    todos = [
        {"id": "one", "content": "Run TTS sample", "status": "in_progress"},
        {"id": "two", "content": "Compare output", "status": "pending"},
    ]
    db.append_message(
        "child-root",
        "assistant",
        tool_calls=[
            {
                "id": "todo-real",
                "type": "function",
                "function": {"name": "todo", "arguments": "{}"},
            }
        ],
    )
    db.append_message(
        "child-root",
        "tool",
        content=json.dumps({"todos": todos}),
        tool_call_id="todo-real",
    )
    db.end_session("child-root", "compression")
    db.create_session(
        "child-session", "subagent", parent_session_id="child-root"
    )
    db.touch_session_activity(
        "child-session", 1_700_000_100.0, description="running text_to_speech"
    )
    try:
        snapshot = _session_inspection(db, "child-session")
    finally:
        db.close()

    assert snapshot["session"]["id"] == "child-session"
    assert snapshot["session"]["state"] == "active"
    assert snapshot["activity"]["description"] == "running text_to_speech"
    assert snapshot["plan"]["current_stage"] == "Run TTS sample"
    assert snapshot["plan"]["todos"] == todos
