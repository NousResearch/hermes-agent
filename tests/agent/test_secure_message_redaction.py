from agent.conversation_loop import _redact_secure_markers, _redact_secure_markers_in_messages


def test_redact_secure_markers_removes_payload():
    text = "Password: [[secure]]abc123[[/secure]] after"

    redacted = _redact_secure_markers(text)

    assert "abc123" not in redacted
    assert "[redacted — secure message omitted from transcript]" in redacted
    assert redacted.endswith(" after")


def test_redact_secure_markers_in_messages_only_assistant_content():
    messages = [
        {"role": "user", "content": "[[secure]]user text[[/secure]]"},
        {"role": "assistant", "content": "token [[secure]]abc123[[/secure]]"},
        {"role": "tool", "content": "[[secure]]tool text[[/secure]]"},
    ]

    _redact_secure_markers_in_messages(messages)

    assert messages[0]["content"] == "[[secure]]user text[[/secure]]"
    assert "abc123" not in messages[1]["content"]
    assert messages[2]["content"] == "[[secure]]tool text[[/secure]]"


def test_redact_secure_markers_honors_configured_bounds(monkeypatch):
    import agent.conversation_loop as loop

    monkeypatch.setattr(loop, "_secure_marker_bounds", lambda: ("<<secret>>", "<</secret>>"))

    redacted = _redact_secure_markers("token <<secret>>abc123<</secret>>")

    assert "abc123" not in redacted
    assert "[redacted — secure message omitted from transcript]" in redacted
