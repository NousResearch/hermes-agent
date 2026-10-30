"""Delta rendering: cursors, wip marker, rewritten-prefix replay, adviser-row filtering."""

from __future__ import annotations

from agent.adviser_delta import WIP_MARKER, DeltaTracker, fingerprint_message, render_delta


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def test_first_capture_renders_the_transcript_tail():
    tracker = DeltaTracker()
    messages = [_user("fix the bug")]

    delta = tracker.capture(messages, wip=False)

    assert delta is not None
    assert "**user**" in delta.text and "fix the bug" in delta.text
    assert delta.text.startswith("### Session update")
    assert WIP_MARKER not in delta.text


def test_second_capture_only_renders_new_messages():
    tracker = DeltaTracker()
    messages = [_user("fix the bug"), _assistant("working")]
    tracker.capture(messages, wip=False)
    messages.append({"role": "tool", "name": "read_file", "content": "line 1"})

    delta = tracker.capture(messages, wip=True)

    assert delta is not None
    assert "read_file" in delta.text
    assert "fix the bug" not in delta.text
    assert delta.text.endswith(WIP_MARKER)


def test_rewritten_prefix_replays_and_flags_reset():
    tracker = DeltaTracker()
    messages = [_user("first"), _assistant("one")]
    tracker.capture(messages, wip=False)
    messages[1] = _assistant("compressed summary")

    delta = tracker.capture(messages, wip=False)

    assert delta is not None and delta.reset is True
    assert "compressed summary" in delta.text
    assert "first" in delta.text


def test_shrunk_transcript_replays():
    tracker = DeltaTracker()
    tracker.capture([_user("a"), _assistant("b")], wip=False)

    delta = tracker.capture([_user("a")], wip=False)

    assert delta is not None and delta.reset is True


def test_adviser_rows_and_system_messages_are_not_rendered():
    text = render_delta([
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "[adviser — advisory only]\n- [nit] stale", "display_kind": "adviser"},
        _assistant("answer"),
    ])

    assert text is not None
    assert "system prompt" not in text
    assert "advisory only" not in text
    assert "answer" in text


def test_fingerprint_ignores_timestamp_and_usage():
    base = {"role": "assistant", "content": "x"}
    assert fingerprint_message({**base, "timestamp": 1}) == fingerprint_message({**base, "usage": {"t": 2}})


def test_tool_calls_render_arguments():
    text = render_delta([{
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}],
    }])

    assert text is not None
    assert "read_file" in text and "a.py" in text
