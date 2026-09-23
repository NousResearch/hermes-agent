"""Regression tests: max-iterations summary must strip internal display metadata
(display_kind / display_metadata) from the wire request.

Live evidence (codex-router-dev state.db, Session20260915_114653_8be3ba): after the
500-iteration budget ended, the terminal summary request failed with API400
``messages[66].display_kind steer``. The mid-turn steer row (prompt_builder
``steer_user_row()`` persists ``display_kind: "steer"``) is delivered AFTER a tool
result, so ``_drop_thinking_only_and_merge_users`` cannot merge it away; the summary
path's hand-built messages bypass the transport's key sweeper and the row's display
metadata reached the strict Fireworks gateway. The main loop strips these keys at
turn_context (``display_kind, display_metadata, _row_id``); the summary path must
mirror that — key-level only, never deleting from persisted history, never dropping
the steer content itself.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)], model="test/model"
    )


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked OpenAI client and tool loading."""
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


def _live_shaped_messages():
    """The live failure shape: a tool round, then the standalone steer row
    (DB row 6266 followed a tool result — no merge can consume it)."""
    return [
        {"role": "user", "content": "start the work"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "terminal", "arguments": "{\"command\": \"pytest\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "test output"},
        {"role": "user",
         "content": "[OUT-OF-BAND USER MESSAGE]\nchange course\n[/OUT-OF-BAND USER MESSAGE]",
         "display_kind": "steer", "_row_id": 6266,
         "display_metadata": {"thread": 78465}},
    ]


class TestSummaryStripsInternalDisplayMetadata:
    def test_summary_wire_has_no_display_kind_or_metadata(self, agent):
        """Steer rows (display_kind='steer') must not reach the wire. Strict
        Fireworks-class gateways 400:
        'Extra inputs are not permitted, field: messages[N].display_kind'."""
        agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
        agent._cached_system_prompt = "You are helpful."
        messages = _live_shaped_messages()

        result = agent._handle_max_iterations(messages, 60)

        assert result == "Summary"
        sent_msgs = agent.client.chat.completions.create.call_args.kwargs.get("messages", [])
        for m in sent_msgs:
            assert "display_kind" not in m, m
            assert "display_metadata" not in m, m

    def test_summary_keeps_steer_content_intact(self, agent):
        """Sanitization is key-level only: the steer TEXT rides the wire so the
        summary reflects the user's mid-turn correction."""
        agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
        agent._cached_system_prompt = "You are helpful."
        steer_text = "change course"
        messages = _live_shaped_messages()

        agent._handle_max_iterations(messages, 60)

        sent_msgs = agent.client.chat.completions.create.call_args.kwargs.get("messages", [])
        wire_steer = [m for m in sent_msgs
                      if m.get("role") == "user" and steer_text in (m.get("content") or "")]
        assert wire_steer, f"steer content lost from wire: {sent_msgs}"

    def test_summary_does_not_mutate_persisted_history(self, agent):
        """The path copies each message: display_kind/display_metadata survive in
        the caller's list (session renderers and compaction recognizers read them)."""
        agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
        agent._cached_system_prompt = "You are helpful."
        messages = _live_shaped_messages()

        agent._handle_max_iterations(messages, 60)

        assert messages[3]["display_kind"] == "steer"
        assert messages[3]["display_metadata"] == {"thread": 78465}

    def test_summary_request_contract_full_schema_cleanliness(self, agent):
        """Provider-request contract: every summary wire message must be
        Chat-Completions-schema-clean: no display_*, no _row_id, no api_content,
        no tool_name, no timestamp."""
        agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
        agent._cached_system_prompt = "You are helpful."
        messages = _live_shaped_messages()
        messages[3]["api_content"] = "sidecar-bytes"

        agent._handle_max_iterations(messages, 60)

        sent_msgs = agent.client.chat.completions.create.call_args.kwargs.get("messages", [])
        foreign = ("display_kind", "display_metadata", "_row_id", "api_content",
                   "tool_name", "timestamp")
        for m in sent_msgs:
            present = [k for k in foreign if k in m]
            assert not present, f"schema-foreign keys on wire: {present} in {m}"

    def test_summary_failure_path_still_returns_handoff(self, agent):
        """Error path: when the summary call itself fails, the fallback text is
        returned (never a misleading Done) and the relay call is closed as failed."""
        agent.client.chat.completions.create.side_effect = Exception(
            "400 - Extra inputs are not permitted, field: messages[66].display_kind"
        )
        agent._cached_system_prompt = "You are helpful."
        messages = _live_shaped_messages()

        with patch("agent.relay_llm.complete_logical_call") as complete_logical:
            result = agent._handle_max_iterations(messages, 60)

        assert isinstance(result, str) and result
        assert "maximum iterations" in result.lower() or "couldn't" in result.lower()
        complete_logical.assert_called_once()
        assert complete_logical.call_args.kwargs == {"outcome": "failed"}
        # History untouched by the failed attempt.
        assert messages[3]["display_kind"] == "steer"
