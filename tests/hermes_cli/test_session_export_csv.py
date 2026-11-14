"""CSV transcript exports through the real CLI and SessionDB (part of #57233)."""

import csv
import io
import json
import sys
from pathlib import Path

import pytest

from hermes_cli import main
from hermes_state import SessionDB


@pytest.fixture
def store(tmp_path):
    with SessionDB() as db:
        assert Path(db.db_path).resolve().is_relative_to(tmp_path.resolve())
        db.create_session("csv-session", source="cli")
        yield db


def _export(monkeypatch, output, *options):
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "export", str(output),
                                    "--format", "csv", *options])
    main.main()


def test_csv_cli_writes_message_rows_with_tool_details(store, tmp_path, monkeypatch):
    content = '你好, "quoted"\nsecond line'
    calls = [{"id": "call_1", "type": "function",
              "function": {"name": "read_file", "arguments": '{"path":"中文.txt"}'}}]
    store.append_message("csv-session", "user", content, timestamp=1700000000)
    store.append_message("csv-session", "assistant", None, tool_calls=calls, timestamp=1700000001)
    store.append_message("csv-session", "tool", "", tool_name="read_file", tool_call_id="call_1",
                         timestamp=1700000002)
    expected_calls = store.export_session("csv-session")["messages"][1]["tool_calls"]
    output = tmp_path / "transcript.csv"

    _export(monkeypatch, output, "--session-id", "csv-sess")

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["session_id"] for row in rows] == ["csv-session"] * 3
    assert [row["index"] for row in rows] == ["1", "2", "3"]
    assert [row["role"] for row in rows] == ["user", "assistant", "tool"]
    assert [row["content"] for row in rows] == [content, "", ""]
    assert rows[0]["created_at"] == "2023-11-14T22:13:20Z"
    assert json.loads(rows[1]["tool_calls"]) == expected_calls
    assert rows[2]["tool_call_id"] == "call_1"
    assert rows[2]["tool_name"] == "read_file"


@pytest.mark.parametrize("value", ["=1+1", "+SUM(1,2)", "-2+3", "@SUM(1,2)",
                                   "\t=1+1", "\r=1+1", "\n=1+1", " \x00\x1f=1+1",
                                   "\tplain text"])
def test_csv_neutralizes_formula_and_control_prefixes_in_text_cells(value):
    from hermes_cli.session_export import render_sessions_export

    session = {"id": value, "messages": [{"role": value, "content": value,
               "timestamp": value, "tool_call_id": value, "tool_name": value}]}
    row = next(csv.DictReader(io.StringIO(render_sessions_export([session], fmt="csv"))))
    for field in ("session_id", "created_at", "role", "content", "tool_call_id", "tool_name"):
        assert row[field] == "'" + value
    # Quoting a field is not a formula defense; JSONL stays the machine-readable source.
    assert json.loads(render_sessions_export([session]))["messages"][0]["content"] == value


def test_csv_prompt_only_reuses_redaction_and_session_filters(store, monkeypatch, capsys):
    secret = "sk-" + "a" * 48
    store.append_message("csv-session", "user", f"first {secret}")
    store.append_message("csv-session", "assistant", "not a prompt")
    store.append_message("csv-session", "user", "=1+1")
    store.end_session("csv-session", "done")
    store.create_session("other-session", source="telegram")
    store.append_message("other-session", "user", "other source")
    store.end_session("other-session", "done")

    _export(monkeypatch, "-", "--source", "cli", "--only", "user-prompts", "--redact")

    output = capsys.readouterr().out
    rows = list(csv.DictReader(io.StringIO(output)))
    assert [r["role"] for r in rows] == ["user", "user"]
    assert [r["index"] for r in rows] == ["1", "2"]
    assert rows[1]["content"] == "'=1+1"
    assert secret not in output
    assert rows[0]["content"].startswith("first ")
    assert {r["session_id"] for r in rows} == {"csv-session"}
    assert secret in store.export_session("csv-session")["messages"][0]["content"]


def test_csv_transcript_omits_system_and_undone_rows_but_keeps_compacted(store, monkeypatch, capsys):
    store.append_message("csv-session", "system", "private system context")
    store.append_message("csv-session", "user", "archived question")
    store.append_message("csv-session", "assistant", "archived answer")
    store.append_message("csv-session", "user", "live question")
    tail = store.get_messages("csv-session")[-1:]
    store.archive_and_compact("csv-session", [{"role": "user", "content": "summary"}, *tail], tail_count=1)
    removed = store.append_message("csv-session", "user", "undone question")
    store.deactivate_message("csv-session", removed)
    expected = [m["content"] for m in store.export_session("csv-session", include_compacted=True)["messages"]
                if m["role"] != "system"]

    _export(monkeypatch, "-", "--session-id", "csv-session")

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [row["content"] for row in rows] == expected
    assert "archived question" in expected
    assert "undone question" not in expected
    assert all(row["role"] != "system" for row in rows)


def test_csv_prompt_only_does_not_export_tool_metadata_on_user_rows(store, monkeypatch, capsys):
    store.append_message("csv-session", "user", "just my prompt", tool_calls=[
        {"id": "call_x", "type": "function", "function": {"name": "private", "arguments": "{}"}}],
        tool_call_id="call_x", tool_name="private")
    _export(monkeypatch, "-", "--session-id", "csv-session", "--only", "user-prompts")
    row = next(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert row["content"] == "just my prompt"
    assert all(row[field] == "" for field in ("tool_calls", "tool_call_id", "tool_name"))


def test_csv_stdout_preserves_embedded_line_endings(monkeypatch):
    from hermes_cli.sessions_cmd import _write_output
    from hermes_cli.session_export import render_sessions_export

    content = "one\r\ntwo\nthree"
    text = render_sessions_export([{"id": "s", "messages": [{"role": "user", "content": content}]}], fmt="csv")
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8", newline="\r\n")
    monkeypatch.setattr(sys, "stdout", stream)
    _write_output("-", text, "unused", newline="")
    stream.flush()
    rows = list(csv.DictReader(io.StringIO(raw.getvalue().decode("utf-8"), newline="")))
    assert rows[0]["content"] == content


def test_csv_empty_selection_writes_header(store, monkeypatch, capsys):
    _export(monkeypatch, "-", "--source", "nonexistent")
    reader = csv.DictReader(io.StringIO(capsys.readouterr().out))
    assert list(reader) == []
    assert reader.fieldnames == ["session_id", "index", "created_at", "role", "content",
                                 "tool_calls", "tool_call_id", "tool_name"]


def test_csv_file_preserves_cr_lf_without_blank_records(store, tmp_path, monkeypatch):
    content = "one\rtwo\r\nthree\nfour"
    store.append_message("csv-session", "user", content)
    _export(monkeypatch, tmp_path, "--session-id", "csv-session")
    with (tmp_path / "hermes_session_csv-session.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 2
    assert rows[1][4] == content
