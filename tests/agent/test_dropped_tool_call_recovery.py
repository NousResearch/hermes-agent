"""Regression tests for dropped tool-call recovery.

Some providers (observed: claude-opus-4.8 / claude-sonnet-4.5 on GitHub
Copilot, ~2026-07) return ``finish_reason="tool_calls"`` while the parsed
``tool_calls`` array is empty — the model signalled it wanted to act but the
payload shipped no call. Before the fix, the conversation loop took the
no-tool-calls ``else`` branch, treated the turn's narration as the final
answer, and exited with the task unstarted. On an unattended multi-step job
(e.g. a scheduled PR reviewer) this silently did nothing.

The fix keys on the provider contract violation itself
(``finish_reason == "tool_calls"`` with zero ``tool_calls``) and re-prompts,
bounded to 3 consecutive stalls, with the budget resetting after any
successful tool round so it guards each stall rather than the whole run. It
also recovers when a model emits a trailing ``card`` + ``<invoke>`` block as
plain text with ``finish_reason="stop"`` instead of using the provider's
structured tool channel. Genuine text turns are unaffected.
"""

from __future__ import annotations

from contextlib import closing
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_run_agent's fixture)
    so we can stage a dropped-tool-call response + continuation pair on
    ``.chat.completions.create``."""
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
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


def _dropped_tool_call_response(content: str):
    """A response whose finish_reason claims a tool call, but tool_calls is
    empty — the provider contract violation this fix recovers from."""
    from tests.agent.test_run_agent import _mock_assistant_msg
    return SimpleNamespace(
        id="chatcmpl-dropped",
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content=content, tool_calls=None),
            finish_reason="tool_calls",
        )],
        usage=None,
    )


class TestDroppedToolCallRecovery:
    def test_textual_invoke_stop_reprompts_instead_of_exiting(self, loop_agent, tmp_path):
        """A known tool emitted as trailing XML text is an attempted call,
        not a final answer, even when the provider reports a clean stop."""
        from tests.agent.test_run_agent import _mock_response, _mock_tool_call
        from hermes_state import SessionDB

        loop_agent.valid_tool_names = {"web_search"}
        leaked = (
            "I will search now.\n\ncard\n"
            '<invoke name="web_search">\n'
            '<parameter name="query">Hermes</parameter>\n'
            "</invoke>"
        )
        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content=leaked),
            _mock_response(content="", finish_reason="tool_calls", tool_calls=[
                _mock_tool_call("web_search", '{"query":"Hermes"}', call_id="native-call"),
            ]),
            _mock_response(content="Found the result."),
        ]

        with (
            closing(SessionDB(db_path=tmp_path / "state.db")) as db,
            patch.object(loop_agent, "_session_db", db),
            patch.object(loop_agent, "_session_db_created", False),
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
            patch("model_tools.handle_function_call", return_value="search result") as execute,
        ):
            result = loop_agent.run_conversation("search for Hermes")
            assert loop_agent._flush_messages_to_session_db(result["messages"])
            saved = db.get_messages_as_conversation(loop_agent.session_id)

        assert loop_agent.client.chat.completions.create.call_count == 3
        execute.assert_called_once()
        assert execute.call_args.args[:2] == ("web_search", {"query": "Hermes"})
        assert "Found the result." in result["final_response"]
        assert saved[-1]["content"] == "Found the result."
        assert all(leaked != msg.get("content") for msg in saved)
        assert not any("provider-native structured tool call" in (msg.get("content") or "") for msg in saved)
        requests = loop_agent.client.chat.completions.create.call_args_list
        retry_messages = requests[1].kwargs["messages"]
        assert "provider-native structured tool call" in retry_messages[-1]["content"]
        assert retry_messages[0] == requests[0].kwargs["messages"][0]

    def test_dropped_tool_call_reprompts_instead_of_exiting(self, loop_agent):
        """finish_reason=tool_calls with an empty tool_calls array must
        re-prompt the model to emit the call rather than exiting the loop
        with the narration as the final answer."""
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _dropped_tool_call_response("Let me verify the PR and gather evidence."),
            _mock_response(content="All checks pass. Approved.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("review the PR")

        assert loop_agent.client.chat.completions.create.call_count == 2, (
            "A dropped tool call must trigger a re-prompt (second API call), "
            "not exit the loop after one call."
        )

        # The loop must have injected a nudge user-message telling the model to
        # issue the actual tool call.
        second_call = loop_agent.client.chat.completions.create.call_args_list[1]
        msgs = second_call.kwargs.get("messages") or second_call.args[0].get("messages")
        last_user = next(
            (m for m in reversed(msgs) if m.get("role") == "user"), None,
        )
        assert last_user is not None
        assert "tool call" in (last_user.get("content") or "").lower(), (
            "The nudge must explicitly ask the model to issue the tool call."
        )
        assert "All checks pass" in result["final_response"]


    def test_clean_stop_text_turn_is_unaffected(self, loop_agent):
        """A genuine finish_reason=stop text response must exit normally — the
        recovery path must not fire on ordinary final answers."""
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content="Here is your answer.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("hello")

        assert loop_agent.client.chat.completions.create.call_count == 1, (
            "A clean finish_reason=stop turn must not trigger a re-prompt."
        )
        assert "Here is your answer." in result["final_response"]

    @pytest.mark.parametrize(
        "content",
        [
            (
                'For example: <invoke name="memory"></invoke> is transcript '
                "data, not a tool call."
            ),
            (
                "A provider may represent a memory call like this:\n"
                '<invoke name="memory">\n'
                '<parameter name="target">memory</parameter>\n'
                "</invoke>"
            ),
            'card\n<invoke name="not_a_real_tool"></invoke>',
            'card\n<invoke name="memory">incomplete',
            'card\n<invoke name="memory"></invoke>\nThis is an example.',
            '```xml\ncard\n<invoke name="memory"></invoke>\n```',
            'postcard\n<invoke name="memory"></invoke>',
            '<think>card\n<invoke name="memory"></invoke></think>\nHere is your answer.',
        ],
    )
    def test_non_call_invoke_text_is_unaffected(self, loop_agent, content):
        """Explanatory text, including trailing known-tool examples, and
        unknown tool names must not be mistaken for executable tool intent."""
        from tests.agent.test_run_agent import _mock_response

        loop_agent.valid_tool_names = {"memory"}
        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content=content, finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("explain tool-call markup")

        assert loop_agent.client.chat.completions.create.call_count == 1
        assert loop_agent._strip_think_blocks(content).strip() == result["final_response"]

    @pytest.mark.parametrize("textual", [False, True])
    def test_persistent_dropped_tool_calls_are_bounded(self, loop_agent, textual):
        """If the model never emits a call, the recovery must give up after a
        bounded number of consecutive stalls instead of looping forever."""
        from tests.agent.test_run_agent import _mock_response

        # Stage plenty of dropped-tool-call responses followed by a clean stop,
        # so that if the bound is respected the loop exits on its own well
        # before exhausting the staged responses (no StopIteration).
        loop_agent.valid_tool_names = {"memory"}
        stalled = (
            _mock_response(content='card\n<invoke name="memory"></invoke>')
            if textual else _dropped_tool_call_response("Let me check.")
        )
        loop_agent.client.chat.completions.create.side_effect = [stalled] * 9 + [
            _mock_response(content="done", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("review the PR")

        # 1 initial call + at most 3 bounded re-prompts = 4 total before the
        # guard stops firing. It must NOT consume all 9 staged stalls.
        assert loop_agent.client.chat.completions.create.call_count <= 4, (
            "Consecutive dropped tool calls must be bounded (no infinite loop)."
        )
        assert result is not None

    @pytest.mark.parametrize("textual", [False, True])
    def test_nudge_pair_is_ephemeral_scaffolding(self, loop_agent, textual):
        """The re-prompt pair (interim assistant turn + synthetic user nudge)
        must be flagged as ephemeral scaffolding so persistence never writes
        it to the durable transcript — a resumed session must not replay the
        internal "issue the actual tool call now" instruction as user-authored
        context (#69630 review follow-up)."""
        from agent.session_persistence import _is_ephemeral_scaffolding
        from tests.agent.test_run_agent import _mock_response

        loop_agent.valid_tool_names = {"memory"}
        stalled = (
            _mock_response(content='card\n<invoke name="memory"></invoke>')
            if textual else _dropped_tool_call_response("Let me verify the PR.")
        )
        loop_agent.client.chat.completions.create.side_effect = [
            stalled,
            _mock_response(content="All checks pass. Approved.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("review the PR")

        assert result["completed"] is True
        # The finalization pop strips the answered pair from the live list —
        # no flagged scaffolding may survive into the returned transcript.
        leftover = [
            m for m in result["messages"]
            if isinstance(m, dict) and m.get("_dropped_toolcall_nudge")
        ]
        assert not leftover, (
            "The re-prompt pair must be stripped at finalization, not kept "
            "in the returned transcript."
        )
        # And the persistence filter must classify the flag as ephemeral so a
        # mid-turn flush can never write the pair to the durable store either.
        assert _is_ephemeral_scaffolding(
            {"role": "user", "content": "nudge", "_dropped_toolcall_nudge": True}
        ), (
            "_dropped_toolcall_nudge messages must be classified as "
            "ephemeral scaffolding so they are never persisted."
        )

