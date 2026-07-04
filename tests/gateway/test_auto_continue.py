"""Tests for the auto-continue feature (#4493 / #45232).

When the gateway restarts mid-agent-work, the session transcript can end on a
tool result that the agent never processed.  The auto-continue logic detects
this and prepends an API-only system note to the next user message so the model
does not re-execute stale interrupted tool calls before addressing new input.
"""



def _simulate_auto_continue(agent_history: list, user_message: str) -> str:
    """Reproduce the auto-continue injection logic from _run_agent().

    This mirrors the exact code in gateway/run.py so we can test the
    detection and message transformation without spinning up a full
    gateway runner.
    """
    message = user_message
    if agent_history and agent_history[-1].get("role") == "tool":
        message = (
            "[System note: A new message has arrived. The conversation "
            "history contains pending tool outputs from an interrupted turn. "
            "IGNORE those pending results. Address the user's NEW message "
            "below FIRST. Do NOT re-execute old tool calls from the history.]\n\n"
            + message
        )
    return message


class TestAutoDetection:
    """Test that trailing tool results are correctly detected."""

    def test_trailing_tool_result_triggers_note(self):
        history = [
            {"role": "user", "content": "deploy the app"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "deployed successfully"},
        ]
        result = _simulate_auto_continue(history, "what happened?")
        assert "[System note:" in result
        assert "interrupted" in result
        assert "NEW message" in result
        assert "Do NOT re-execute" in result
        assert "what happened?" in result

    def test_trailing_assistant_message_no_note(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = _simulate_auto_continue(history, "how are you?")
        assert "[System note:" not in result
        assert result == "how are you?"

    def test_empty_history_no_note(self):
        result = _simulate_auto_continue([], "hello")
        assert result == "hello"

    def test_trailing_user_message_no_note(self):
        """Shouldn't happen in practice, but ensure no false positive."""
        history = [
            {"role": "user", "content": "hello"},
        ]
        result = _simulate_auto_continue(history, "hello again")
        assert result == "hello again"

    def test_multiple_tool_results_still_triggers(self):
        """Multiple tool calls in a row — last one is still role=tool."""
        history = [
            {"role": "user", "content": "search and read"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "function": {"name": "search", "arguments": "{}"}},
                {"id": "call_2", "function": {"name": "read", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "found it"},
            {"role": "tool", "tool_call_id": "call_2", "content": "file content here"},
        ]
        result = _simulate_auto_continue(history, "continue")
        assert "[System note:" in result

    def test_original_message_preserved_after_note(self):
        """The user's actual message must appear after the system note."""
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "t", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
        ]
        result = _simulate_auto_continue(history, "now do X")
        # System note comes first, then user's message
        note_end = result.index("]\n\n")
        user_msg_start = result.index("now do X")
        assert user_msg_start > note_end


class TestInterruptedReplayFiltering:
    def test_interrupted_tool_tail_is_removed_from_agent_history(self):
        from gateway.run import _build_gateway_agent_history

        history = [
            {"role": "user", "content": "transcribe this video"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"exit_code": 130, "output": "[Command interrupted]"}',
            },
        ]

        agent_history, observed_context = _build_gateway_agent_history(history)

        assert observed_context is None
        assert agent_history == [{"role": "user", "content": "transcribe this video"}]

    def test_mixed_tail_with_one_interrupted_result_is_removed(self):
        from gateway.run import _build_gateway_agent_history

        history = [
            {"role": "user", "content": "search and transcribe"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
                    {"id": "call_2", "function": {"name": "terminal", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "found URL"},
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": '{"exit_code": 130, "output": "[Command interrupted]"}',
            },
        ]

        agent_history, _observed_context = _build_gateway_agent_history(history)

        assert agent_history == [{"role": "user", "content": "search and transcribe"}]

    def test_successful_tool_tail_is_preserved(self):
        from gateway.run import _build_gateway_agent_history

        history = [
            {"role": "user", "content": "deploy"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "deployed successfully"},
        ]

        agent_history, _observed_context = _build_gateway_agent_history(history)

        assert agent_history[-1]["role"] == "tool"
        assert agent_history[-1]["content"] == "deployed successfully"

    def test_dangling_unanswered_tool_call_tail_is_removed(self):
        """A trailing assistant(tool_calls) with NO tool answers is stripped.

        This is the SIGKILL signature from #49201: the tool itself ran a
        restart/shutdown command and killed the gateway before its result was
        persisted. The transcript tail is an assistant message with tool_calls
        and zero matching tool rows. Without stripping it, the model re-issues
        the unanswered call on resume and loops the restart forever.
        """
        from gateway.run import _build_gateway_agent_history

        history = [
            {"role": "user", "content": "restart the container"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command": "docker restart hermes-agent"}',
                        },
                    },
                ],
            },
        ]

        agent_history, _observed_context = _build_gateway_agent_history(history)

        assert agent_history == [{"role": "user", "content": "restart the container"}]

    def test_dangling_tail_after_completed_pair_is_removed_only_at_tail(self):
        """Only the trailing unanswered tool-call block is stripped.

        An earlier completed assistant→tool pair must survive — we only drop
        the final assistant(tool_calls) that has no answers.
        """
        from gateway.run import _build_gateway_agent_history

        history = [
            {"role": "user", "content": "do two things"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "found it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command": "systemctl restart hermes"}',
                        },
                    },
                ],
            },
        ]

        agent_history, _observed_context = _build_gateway_agent_history(history)

        # The completed call_1 pair survives; the dangling call_2 tail is gone.
        assert agent_history[-1]["role"] == "tool"
        assert agent_history[-1]["content"] == "found it"
        # The surviving assistant(tool_calls) is the completed call_1 (which
        # has a matching tool answer), not the stripped dangling call_2.
        _surviving_calls = [
            tc.get("id")
            for m in agent_history
            if m.get("role") == "assistant" and m.get("tool_calls")
            for tc in m["tool_calls"]
        ]
        assert _surviving_calls == ["call_1"]

    def test_persisted_auto_continue_note_is_not_replayed(self):
        from gateway.run import _build_gateway_agent_history

        history = [
            {"role": "user", "content": "first real question"},
            {
                "role": "user",
                "content": (
                    "[System note: Your previous turn was interrupted before you could "
                    "process the last tool result(s).]\n\nsecond real question"
                ),
            },
            {"role": "assistant", "content": "answer"},
            {
                "role": "user",
                "content": (
                    "[System note: A new message has arrived. The conversation "
                    "history contains pending tool outputs from an interrupted turn.]\n\nthird"
                ),
            },
        ]

        agent_history, _observed_context = _build_gateway_agent_history(history)

        assert agent_history == [
            {"role": "user", "content": "first real question"},
            {"role": "user", "content": "second real question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "third"},
        ]
