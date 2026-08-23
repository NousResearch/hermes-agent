"""Tests for hermes_cli.foreign_sessions — Claude Code / Codex CLI import.

Fixture JSONL is synthesized inline (tmp_path); the SessionDB is opened
against a temp path so nothing touches the real HERMES_HOME store.
"""

import json

import pytest

from hermes_cli.foreign_sessions import (
    gather_foreign_sessions,
    import_foreign_session,
    list_claude_sessions,
    list_codex_sessions,
    parse_claude_session,
    parse_codex_session,
)


# ── fixture builders ─────────────────────────────────────────────────────


def _claude_lines():
    def msg(role, content):
        return {
            "type": role,
            "sessionId": "abc-123",
            "cwd": "/home/user/proj",
            "message": {"role": role, "content": content},
        }

    return [
        {"type": "summary", "summary": "Fix the flaky test"},
        msg("user", "Please fix the flaky test in CI."),
        msg(
            "assistant",
            [
                {"type": "text", "text": "Looking into it now."},
                {"type": "tool_use", "name": "Bash", "id": "t1", "input": {}},
            ],
        ),
        # tool_result echoed back as a user message — must NOT become a turn
        msg("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]),
        msg("assistant", [{"type": "text", "text": "Fixed — the sleep was too short."}]),
        msg("user", "Great, thanks!"),
        msg("assistant", [{"type": "text", "text": "Anytime."}]),
    ]


def _write_claude_fixture(tmp_path, extra_lines=None):
    proj = tmp_path / ".claude" / "projects" / "-home-user-proj"
    proj.mkdir(parents=True)
    f = proj / "abc-123.jsonl"
    lines = [json.dumps(entry) for entry in _claude_lines()]
    if extra_lines:
        lines = extra_lines + lines
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


def _codex_lines():
    def item(payload):
        return {"timestamp": "2026-08-15T21:35:28Z", "type": "response_item", "payload": payload}

    def message(role, kind, text):
        return item({"type": "message", "role": role, "content": [{"type": kind, "text": text}]})

    return [
        {
            "type": "session_meta",
            "payload": {"session_id": "0000-1111", "cwd": "/home/user/repo"},
        },
        # developer/system payloads must never be imported
        message("developer", "input_text", "<skills_instructions>secret system stuff"),
        message("user", "input_text", "<recommended_plugins>\ninjected wrapper"),
        message("user", "input_text", "Summarize the transcripts please."),
        message("assistant", "output_text", "Reading them one at a time."),
        item({"type": "custom_tool_call", "name": "shell", "call_id": "c1"}),
        item({"type": "custom_tool_call_output", "call_id": "c1", "output": "big output"}),
        message("assistant", "output_text", "Done — here is the summary."),
        message("user", "input_text", "Now write it to a file."),
        message("assistant", "output_text", "Written to summary.md."),
    ]


def _write_codex_fixture(tmp_path, extra_lines=None):
    day = tmp_path / ".codex" / "sessions" / "2026" / "08" / "15"
    day.mkdir(parents=True)
    f = day / "rollout-2026-08-15T21-35-28-0000-1111.jsonl"
    lines = [json.dumps(entry) for entry in _codex_lines()]
    if extra_lines:
        lines = extra_lines + lines
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


@pytest.fixture
def session_db(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


def _assert_alternating(messages):
    roles = [m["role"] for m in messages]
    assert roles, "no messages"
    assert roles[0] == "user"
    for a, b in zip(roles, roles[1:]):
        assert a != b, f"two consecutive {a} messages"


# ── parsing ──────────────────────────────────────────────────────────────


def test_parse_claude_session(tmp_path):
    f = _write_claude_fixture(tmp_path)
    parsed = parse_claude_session(f)
    turns = parsed["turns"]
    _assert_alternating(turns)
    assert len(turns) == 4  # tool_result-only user line skipped; assistants merge
    assert parsed["cwd"] == "/home/user/proj"
    assert parsed["title_guess"] == "Fix the flaky test"
    assert parsed["session_id"] == "abc-123"
    # tool_use flattened to a bracketed summary, merged with adjacent text
    joined_assistant = "\n".join(t["content"] for t in turns if t["role"] == "assistant")
    assert "[ran tool: Bash]" in joined_assistant
    # no fabricated tool_call structures
    assert all(set(t) == {"role", "content"} for t in turns)


def test_parse_codex_session(tmp_path):
    f = _write_codex_fixture(tmp_path)
    parsed = parse_codex_session(f)
    turns = parsed["turns"]
    _assert_alternating(turns)
    assert len(turns) == 4
    assert parsed["cwd"] == "/home/user/repo"
    assert parsed["session_id"] == "0000-1111"
    assert parsed["title_guess"].startswith("Summarize the transcripts")
    # developer + wrapper user lines excluded
    all_text = "\n".join(t["content"] for t in turns)
    assert "skills_instructions" not in all_text
    assert "recommended_plugins" not in all_text
    # tool call summarized in assistant text
    assert "[ran tool: shell]" in all_text


def test_malformed_lines_are_skipped(tmp_path):
    garbage = [
        "this is not json at all {{{",
        '"just a string"',
        json.dumps({"type": "user", "message": "not-a-dict-payload... wait, string"}),
        json.dumps({"type": "user"}),  # missing message
        "",
    ]
    f = _write_claude_fixture(tmp_path, extra_lines=garbage)
    parsed = parse_claude_session(f)
    assert len(parsed["turns"]) == 4  # good lines still parse

    g = _write_codex_fixture(tmp_path, extra_lines=["not json", json.dumps({"type": "response_item"})])
    parsed_codex = parse_codex_session(g)
    assert len(parsed_codex["turns"]) == 4


# ── discovery ────────────────────────────────────────────────────────────


def test_list_sessions(tmp_path):
    _write_claude_fixture(tmp_path)
    _write_codex_fixture(tmp_path)
    claude = list_claude_sessions(tmp_path / ".claude" / "projects")
    codex = list_codex_sessions(tmp_path / ".codex" / "sessions")
    assert len(claude) == 1 and claude[0].source == "claude"
    assert claude[0].turn_count == 4
    assert len(codex) == 1 and codex[0].source == "codex"
    assert codex[0].turn_count == 4
    both = gather_foreign_sessions(
        claude_root=tmp_path / ".claude" / "projects",
        codex_root=tmp_path / ".codex" / "sessions",
    )
    assert len(both) == 2
    assert both[0].mtime >= both[1].mtime  # newest first


def test_list_sessions_missing_roots(tmp_path):
    assert list_claude_sessions(tmp_path / "nope") == []
    assert list_codex_sessions(tmp_path / "nope") == []


# ── import into SessionDB ────────────────────────────────────────────────


def test_import_claude_session(tmp_path, session_db):
    f = _write_claude_fixture(tmp_path)
    session_id = import_foreign_session("claude", f, db=session_db)
    row = session_db.get_session(session_id)
    assert row is not None
    assert row["source"] == "claude-code"
    assert row["cwd"] == "/home/user/proj"
    assert row["message_count"] == 4
    title = session_db.get_session_title(session_id)
    assert title.startswith("Imported from Claude Code: ")
    assert "Please fix the flaky test" in title
    messages = session_db.get_messages(session_id)
    assert len(messages) == 4
    _assert_alternating(messages)
    origin = json.loads(row["origin_json"])
    assert origin["imported_from"]["tool"] == "claude-code"
    assert origin["imported_from"]["path"] == str(f)


def test_import_codex_session(tmp_path, session_db):
    f = _write_codex_fixture(tmp_path)
    session_id = import_foreign_session("@codex", f, db=session_db)
    row = session_db.get_session(session_id)
    assert row is not None
    assert row["source"] == "codex-cli"
    assert row["message_count"] == 4
    title = session_db.get_session_title(session_id)
    assert title.startswith("Imported from Codex CLI: ")
    messages = session_db.get_messages(session_id)
    _assert_alternating(messages)
    # resumable: resolve_session_id round-trips
    assert session_db.resolve_session_id(session_id) == session_id


def test_import_rejects_bad_input(tmp_path, session_db):
    with pytest.raises(ValueError, match="Unknown foreign session source"):
        import_foreign_session("gemini", tmp_path / "x.jsonl", db=session_db)
    with pytest.raises(ValueError, match="not found"):
        import_foreign_session("claude", tmp_path / "missing.jsonl", db=session_db)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No user/assistant conversation"):
        import_foreign_session("claude", empty, db=session_db)


def test_leading_assistant_gets_single_stub(tmp_path):
    day = tmp_path / ".codex" / "sessions" / "2026" / "01" / "01"
    day.mkdir(parents=True)
    f = day / "rollout-x.jsonl"
    lines = [
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Continuing from before."}],
                },
            }
        )
    ]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    parsed = parse_codex_session(f)
    _assert_alternating(parsed["turns"])
    assert len(parsed["turns"]) == 2
    assert parsed["turns"][0]["role"] == "user"
