"""Behavior contract: the verify-on-stop and pre_verify nudge paths in
``run_conversation`` must persist the attempted final answer to the session
DB BEFORE projecting it to the UI — and must not continue the
verification/pre_verify loop on an answer that only exists in this process.

``agent/conversation_loop.py`` already pins this invariant for the
tool-call persistence path (``tests/run_agent/test_tool_call_incremental_persistence.py::
test_failed_assistant_persist_blocks_ui_projection_and_tool_side_effects``);
these tests pin the identical contract for its two nudge-loop siblings.
"""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _mock_response(content="Hello", finish_reason="stop"):
    from types import SimpleNamespace

    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_failed_verify_on_stop_persist_blocks_ui_projection_and_loop_continuation():
    """A verify-on-stop nudge must not surface or loop on an unpersisted answer."""
    agent = _make_agent()
    agent.client.chat.completions.create.return_value = _mock_response(
        content="Here is the final answer.", finish_reason="stop"
    )
    agent._flush_messages_to_session_db = MagicMock(return_value=False)
    agent.interim_assistant_callback = MagicMock()

    with (
        patch("agent.verification_stop.verify_on_stop_enabled", return_value=True),
        patch(
            "agent.verification_stop.build_verify_on_stop_nudge",
            return_value="Please verify your changes before finishing.",
        ),
    ):
        result = agent.run_conversation("do something")

    agent.interim_assistant_callback.assert_not_called()
    assert agent.client.chat.completions.create.call_count == 1, (
        "the verification-stop loop must not continue on unpersisted state"
    )
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["turn_exit_reason"] == "session_persistence_failed"


def test_failed_pre_verify_persist_blocks_ui_projection_and_loop_continuation():
    """A pre_verify-hook nudge must not surface or loop on an unpersisted answer."""
    agent = _make_agent()

    def _respond(*_args, **_kwargs):
        # turn_context resets `_turn_file_mutation_paths` to an empty set
        # during per-turn setup, BEFORE the API call — populate it here
        # (simulating an earlier tool call in the same turn having edited a
        # file) so it is non-empty by the time the pre_verify gate checks it.
        agent._turn_file_mutation_paths = {"src/example.py"}
        return _mock_response(content="I edited the file.", finish_reason="stop")

    agent.client.chat.completions.create.side_effect = _respond
    agent._flush_messages_to_session_db = MagicMock(return_value=False)
    agent.interim_assistant_callback = MagicMock()

    with (
        patch("agent.verification_stop.verify_on_stop_enabled", return_value=False),
        patch("agent.verify_hooks.max_verify_nudges", return_value=3),
        patch("hermes_cli.plugins.has_hook", return_value=True),
        patch(
            "hermes_cli.plugins.get_pre_verify_continue_message",
            return_value="Please run the tests before finishing.",
        ),
    ):
        result = agent.run_conversation("edit the file")

    agent.interim_assistant_callback.assert_not_called()
    assert agent.client.chat.completions.create.call_count == 1, (
        "the pre_verify loop must not continue on unpersisted state"
    )
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["turn_exit_reason"] == "session_persistence_failed"
