"""Native agent-run transcript: one JSONL per (agent, session) under ``HERMES_HOME/sessions``."""

from __future__ import annotations

import json

from agent import run_transcript


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "find the cause"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "x = 1"},
        {"role": "assistant", "content": "the cause is x"},
    ]


def test_each_agent_and_session_gets_its_own_file(tmp_path):
    researcher = run_transcript.path_for("researcher", "s1", home=tmp_path)
    writer = run_transcript.path_for("writer", "s1", home=tmp_path)
    other_session = run_transcript.path_for("researcher", "s2", home=tmp_path)

    assert researcher == tmp_path / "sessions" / "researcher" / "s1.jsonl"
    assert writer.parent.parent == tmp_path / "sessions"
    assert len({researcher, writer, other_session}) == 3


def test_names_cannot_escape_the_sessions_root(tmp_path):
    path = run_transcript.path_for("../../etc", "x/../y", home=tmp_path)

    assert path.parent.parent == tmp_path / "sessions"
    assert path.parent.name not in {"..", "."}
    assert path.suffix == ".jsonl"


def test_record_appends_rows_and_history_reads_them_back(tmp_path):
    info = run_transcript.record("researcher", "s1", _messages(), run_id="r1",
                                 model="m", provider="p", system_message="You research.",
                                 usage={"total_tokens": 12}, home=tmp_path)

    assert info["written"] is True and info["appended"] == 4
    rows = run_transcript.read_events("researcher", "s1", home=tmp_path)
    assert [row["kind"] for row in rows].count("run") == 1
    assert rows[0]["session_id"] == "s1" and rows[0]["model"] == "m"
    assert [row["role"] for row in rows if row.get("kind") == "msg"] == [
        "system", "user", "assistant", "tool", "assistant"]

    history = run_transcript.read_history("researcher", "s1", home=tmp_path)
    assert history[0] == {"role": "user", "content": "find the cause"}
    assert history[1]["tool_calls"][0]["function"] == {"name": "read_file",
                                                       "arguments": '{"path": "a.py"}'}
    assert history[2] == {"role": "tool", "tool_call_id": "c1", "name": "read_file",
                          "content": "x = 1"}
    assert history[3]["content"] == "the cause is x"


def test_a_history_prefix_is_not_written_twice(tmp_path):
    first = run_transcript.record("researcher", "s1", _messages(), run_id="r1",
                                  system_message="You research.", home=tmp_path)
    again = run_transcript.record("researcher", "s1", _messages(), run_id="r2",
                                  system_message="You research.", home=tmp_path)

    assert first["written"] is True
    assert again["written"] is False, "a replayed history prefix is not new activity"

    continued = _messages() + [{"role": "assistant", "content": "and it stayed fixed"}]
    third = run_transcript.record("researcher", "s1", continued, run_id="r3",
                                  system_message="You research.", home=tmp_path)
    assert third["written"] is True and third["appended"] == 1

    rows = run_transcript.read_events("researcher", "s1", home=tmp_path)
    assert [row["text"] for row in rows if row.get("role") == "system"] == ["You research."]
    assert [row["text"] for row in rows if row.get("role") == "assistant"] == [
        "", "the cause is x", "and it stayed fixed"]
    history = run_transcript.read_history("researcher", "s1", home=tmp_path)
    assert [row["role"] for row in history] == ["user", "assistant", "tool", "assistant",
                                                "assistant"]


def test_reasoning_is_its_own_row_and_not_replayed_as_a_turn(tmp_path):
    messages = [{"role": "assistant", "content": "answer", "reasoning_content": "thinking"}]
    run_transcript.record("researcher", "s1", messages, run_id="r1", home=tmp_path)

    rows = run_transcript.read_events("researcher", "s1", home=tmp_path)
    assert [row["role"] for row in rows if row.get("kind") == "msg"] == ["assistant",
                                                                        "reasoning"]
    assert [row["role"] for row in run_transcript.read_history("researcher", "s1",
                                                              home=tmp_path)] == ["assistant"]


def test_a_truncated_final_line_is_skipped(tmp_path):
    run_transcript.record("researcher", "s1", _messages()[:1], run_id="r1", home=tmp_path)
    path = run_transcript.path_for("researcher", "s1", home=tmp_path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"kind": "msg", "role": "assis')

    rows = run_transcript.read_events("researcher", "s1", home=tmp_path)
    assert [row["role"] for row in rows if row.get("kind") == "msg"] == ["user"]


def test_a_missing_stream_reads_as_empty(tmp_path):
    assert run_transcript.read_events("nobody", "nothing", home=tmp_path) == []
    assert run_transcript.read_history("nobody", "nothing", home=tmp_path) == []


def test_parts_of_a_multimodal_message_are_joined(tmp_path):
    content = [{"type": "text", "text": "see this"}, {"type": "image_url",
                                                      "image_url": {"url": "data:..."}}]
    run_transcript.record("researcher", "s1", [{"role": "user", "content": content}],
                          run_id="r1", home=tmp_path)

    text = run_transcript.read_history("researcher", "s1", home=tmp_path)[0]["content"]
    assert text.startswith("see this\n")
    assert json.loads(text.split("\n", 1)[1])["image_url"]["url"] == "data:..."
