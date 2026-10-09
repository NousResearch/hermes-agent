"""Regression tests for ``close_interrupted_tool_sequence`` (#48879 follow-up).

#48879 closed the tool-call sequence on interrupt inside ``finalize_turn``,
but the retry/backoff/error interrupt aborts in ``conversation_loop`` ``return``
early and never reach it — so they persisted a raw ``tool`` tail. The next user
message then lands as ``... tool → user``, the role-alternation violation that
makes strict providers (Gemini, Claude) hallucinate a continuation and ignore
prior context (what the user perceives as "lost context").

The fix routes every interrupt abort through this one shared helper. These tests
pin the helper's contract and prove the post-interrupt + next-user-message
transcript is alternation-safe.
"""

from agent.message_sanitization import close_interrupted_tool_sequence


def _tool_tail():
    return [
        {"role": "user", "content": "edit the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "patch", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok edited"},
    ]


def _assert_no_tool_then_user(messages):
    for i in range(len(messages) - 1):
        if messages[i].get("role") == "tool":
            assert messages[i + 1].get("role") != "user", (
                f"role-alternation violation: tool → user at index {i}"
            )








def test_closing_makes_next_user_message_alternation_safe():
    """The whole point: appending a user turn after the close must not
    produce the ``tool → user`` shape strict providers choke on."""
    messages = _tool_tail()
    close_interrupted_tool_sequence(messages, None)
    follow_on = messages + [{"role": "user", "content": "they do! increase the timing"}]
    _assert_no_tool_then_user(follow_on)


def test_assistant_tail_is_left_untouched():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "partial reply"},
    ]
    before = [dict(m) for m in messages]
    assert close_interrupted_tool_sequence(messages, "interrupted") is False
    assert messages == before


def test_user_tail_is_left_untouched():
    messages = [{"role": "user", "content": "hi"}]
    assert close_interrupted_tool_sequence(messages, None) is False
    assert len(messages) == 1


