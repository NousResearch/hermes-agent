from __future__ import annotations

import json

from claude_selfimprove import scanner
from claude_selfimprove.paths import checkpoints_path


def _cp():
    return scanner.CheckpointStore()


def test_discovers_and_parses_user_and_assistant_text(sandbox):
    sandbox.write_transcript(
        "claude",
        "proj-a",
        "sess-1",
        [
            {"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "Never use --no-verify."}},
            {"type": "assistant", "sessionId": "sess-1", "message": {"role": "assistant", "content": [{"type": "text", "text": "Got it, I will not use --no-verify."}]}},
        ],
    )
    cp = _cp()
    turns = list(scanner.scan_new_turns(cp))
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].text == "Never use --no-verify."
    assert turns[1].role == "assistant"
    assert "not use --no-verify" in turns[1].text


def test_skips_non_text_entry_types(sandbox):
    sandbox.write_transcript(
        "claude",
        "proj-a",
        "sess-1",
        [
            {"type": "queue-operation", "sessionId": "sess-1", "operation": "x"},
            {"type": "attachment", "sessionId": "sess-1", "attachment": {}},
            {"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "Real text"}},
        ],
    )
    turns = list(scanner.scan_new_turns(_cp()))
    assert len(turns) == 1
    assert turns[0].text == "Real text"


def test_skips_tool_result_list_content_on_user_role(sandbox):
    sandbox.write_transcript(
        "claude",
        "proj-a",
        "sess-1",
        [
            {
                "type": "user",
                "sessionId": "sess-1",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "some tool output"}],
                },
            },
        ],
    )
    turns = list(scanner.scan_new_turns(_cp()))
    assert turns == []


def test_skips_malformed_json_lines_without_crashing(sandbox):
    path = sandbox.write_transcript(
        "claude",
        "proj-a",
        "sess-1",
        [{"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "ok line"}}],
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
        fh.write(json.dumps({"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "second ok line"}}) + "\n")

    turns = list(scanner.scan_new_turns(_cp()))
    texts = [t.text for t in turns]
    assert texts == ["ok line", "second ok line"]


def test_unchanged_file_is_skipped_on_second_scan(sandbox):
    sandbox.write_transcript(
        "claude", "proj-a", "sess-1",
        [{"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "first pass"}}],
    )
    cp1 = _cp()
    first = list(scanner.scan_new_turns(cp1))
    cp1.save()
    assert len(first) == 1

    cp2 = _cp()
    second = list(scanner.scan_new_turns(cp2))
    assert second == []


def test_appended_lines_are_picked_up_incrementally(sandbox):
    path = sandbox.write_transcript(
        "claude", "proj-a", "sess-1",
        [{"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "line one"}}],
    )
    cp1 = _cp()
    first = list(scanner.scan_new_turns(cp1))
    cp1.save()
    assert [t.text for t in first] == ["line one"]

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "line two"}}) + "\n")

    cp2 = _cp()
    second = list(scanner.scan_new_turns(cp2))
    assert [t.text for t in second] == ["line two"]


def test_partial_trailing_line_is_not_consumed(sandbox):
    path = sandbox.write_transcript(
        "claude", "proj-a", "sess-1",
        [{"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "complete line"}}],
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "cut off')
        # no trailing newline — simulates a session still writing this line

    cp = _cp()
    turns = list(scanner.scan_new_turns(cp))
    assert [t.text for t in turns] == ["complete line"]
    # The checkpoint offset must sit right after the complete line, not at EOF.
    checkpoint = cp.get(str(path))
    with path.open("rb") as fh:
        content = fh.read()
    complete_line_bytes = content.split(b"\n", 1)[0] + b"\n"
    assert checkpoint["offset"] == len(complete_line_bytes)


def test_truncated_file_restarts_from_zero(sandbox):
    path = sandbox.write_transcript(
        "claude", "proj-a", "sess-1",
        [
            {"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "aaaaaaaaaaaaaaaaaaaa"}},
            {"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "bbbbbbbbbbbbbbbbbbbb"}},
        ],
    )
    cp1 = _cp()
    list(scanner.scan_new_turns(cp1))
    cp1.save()

    # Simulate rotation: file replaced with shorter content.
    path.write_text(
        json.dumps({"type": "user", "sessionId": "sess-2", "message": {"role": "user", "content": "new short content"}}) + "\n",
        encoding="utf-8",
    )

    cp2 = _cp()
    turns = list(scanner.scan_new_turns(cp2))
    assert [t.text for t in turns] == ["new short content"]


def test_scans_across_both_source_roots(sandbox):
    sandbox.write_transcript(
        "claude", "proj-a", "sess-1",
        [{"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "from claude"}}],
    )
    sandbox.write_transcript(
        "claude-hermes", "t3code-worktree", "sess-2",
        [{"type": "user", "sessionId": "sess-2", "message": {"role": "user", "content": "from claude-hermes"}}],
    )
    turns = list(scanner.scan_new_turns(_cp()))
    sources = {t.source for t in turns}
    assert sources == {"claude", "claude-hermes"}


def test_checkpoint_store_survives_corrupt_file(sandbox):
    checkpoints_path().parent.mkdir(parents=True, exist_ok=True)
    checkpoints_path().write_text("{not json", encoding="utf-8")
    cp = _cp()  # must not raise
    assert cp.known_file_count() == 0
