"""Tests for the auto-continue feature (#4493 / #45232).

When the gateway restarts mid-agent-work, the session transcript can end on a
tool result that the agent never processed.  The auto-continue logic detects
this and prepends an API-only system note to the next user message so the model
does not re-execute stale interrupted tool calls before addressing new input.
"""


def _simulate_auto_continue(agent_history: list, user_message: str) -> str:
    """Exercise production recovery without starting a gateway or a model."""
    from types import SimpleNamespace
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    runner = SimpleNamespace(
        session_store=SimpleNamespace(_entries={}),
        _adapter_for_source=lambda source: None,
    )
    ctx = TurnContext(message=user_message, history=agent_history, session_key="fixture")
    TurnRunner(runner, ctx)._prepare_turn_message(agent_history)
    return ctx.message


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
        assert "Do NOT repeat successful tool calls" in result
        assert "failed or incomplete result may require a retry after checking" in result
        assert "what happened?" in result


    def test_empty_history_no_note(self):
        result = _simulate_auto_continue([], "hello")
        assert result == "hello"


class TestInterruptedReplayFiltering:
    def test_interrupted_side_effect_is_replayed_as_unknown(self):
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
        assert agent_history[:2] == history[:2]
        assert agent_history[-1]["role"] == "tool"
        assert agent_history[-1]["tool_call_id"] == "call_1"
        assert agent_history[-1]["effect_disposition"] == "unknown"


    def test_dangling_unanswered_side_effect_is_replayed_as_unknown(self):
        """A trailing side-effecting call gets an UNKNOWN result, not a retry.

        This is the SIGKILL signature from #49201: the tool itself ran a
        restart/shutdown command and killed the gateway before its result was
        persisted. The transcript tail is an assistant message with tool_calls
        and zero matching tool rows. A synthetic UNKNOWN result closes the tool
        pair without claiming the restart did not happen or inviting a retry.
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

        assert agent_history[:2] == history
        assert agent_history[-1]["role"] == "tool"
        assert agent_history[-1]["tool_call_id"] == "call_1"
        assert agent_history[-1]["effect_disposition"] == "unknown"


