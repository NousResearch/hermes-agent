"""test_run_agent_reasoning: split from the former tests/agent/test_run_agent.py monolith (#99982)."""
from unittest.mock import patch
import pytest
from tests.agent._run_agent_helpers import _mock_assistant_msg, _mock_response


class TestHasContentAfterThinkBlock:
    def test_none_returns_false(self, agent):
        assert agent._has_content_after_think_block(None) is False

class TestStripThinkBlocks:
    def test_none_returns_empty(self, agent):
        assert agent._strip_think_blocks(None) == ""

    def test_list_content_flattened_no_crash(self, agent):
        """Anthropic-via-OpenRouter returns content as a block list.

        A raw list reaching ``re.sub`` raised ``TypeError: expected string
        or bytes-like object, got 'list'``, which the outer conversation
        loop swallowed and retried forever (infinite "preparing terminal…"
        loop). ``strip_think_blocks`` must flatten list content to visible
        text and drop reasoning blocks.
        """
        result = agent._strip_think_blocks(
            [
                {"type": "text", "text": "visible answer"},
                {"type": "thinking", "thinking": "internal reasoning"},
            ]
        )
        assert isinstance(result, str)
        assert "visible answer" in result
        assert "internal reasoning" not in result


    def test_single_block_removed(self, agent):
        result = agent._strip_think_blocks("<think>reasoning</think> answer")
        assert "reasoning" not in result
        assert "answer" in result


    # ─── Unterminated-block coverage (#8878, #9568, #10408) ──────────────
    # Reasoning models served via NIM / MiniMax M2.7 frequently drop the
    # closing tag, leaking raw reasoning into assistant content. The open
    # tag appears at a block boundary (start of text or after a newline);
    # everything from that tag to end-of-string is stripped.


    def test_mixed_case_closed_pair_stripped(self, agent):
        """Mixed-case variants <THINK>…</THINK>, <Thinking>…</Thinking> are
        handled by case-insensitive closed-pair regex, so the trailing
        content is preserved."""
        result = agent._strip_think_blocks("<THINK>upper</THINK>final")
        assert "upper" not in result
        assert "final" in result
        result = agent._strip_think_blocks("<Thinking>mixed</Thinking>final")
        assert "mixed" not in result
        assert "final" in result

    # ─── Tool-call XML block stripping (openclaw/openclaw#67318) ─────────
    # Some open models (notably Gemma variants via OpenRouter) emit
    # standalone tool-call XML inside assistant content instead of via the
    # structured `tool_calls` field. Left unstripped, raw XML leaks to
    # gateway users (Discord/Telegram/Matrix) and the CLI.


    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "before <tool_call>{x}</function_call> after",
                "before <tool_call>{x}after",
            ),
            (
                "before <function_calls>{x}</tool_calls> after",
                "before <function_calls>{x}after",
            ),
        ],
    )
    def test_mismatched_generic_tool_tags_preserve_opener_and_payload(
        self, agent, text, expected
    ):
        assert agent._strip_think_blocks(text) == expected

class TestExtractReasoning:
    def test_reasoning_field(self, agent):
        msg = _mock_assistant_msg(reasoning="thinking hard")
        assert agent._extract_reasoning(msg) == "thinking hard"

    def test_thinking_block_string_payload_still_extracted(self, agent):
        msg = _mock_assistant_msg(
            content=[{"type": "thinking", "thinking": "  block reasoning  "}]
        )
        assert agent._extract_reasoning(msg) == "block reasoning"

    def test_thinking_block_list_payload_flattened_not_crashed(self, agent):
        # Non-strict OpenAI-compatible backends (Mistral via custom provider) can
        # deliver the thinking value as a JSON array; .strip() on a list crashed
        # the whole API call with AttributeError (#106006). Flatten instead.
        msg = _mock_assistant_msg(
            content=[{"type": "thinking", "thinking": ["list-shaped reasoning", "part two"]}]
        )
        assert agent._extract_reasoning(msg) == "list-shaped reasoningpart two"

class TestReasoningReplayForStrictProviders:
    """Assistant replay must preserve provider-native reasoning fields."""

    def _setup_agent(self, agent):
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False

    def test_kimi_tool_replay_includes_space_reasoning_content(self, agent):
        self._setup_agent(agent)
        agent.base_url = "https://api.kimi.com/coding/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.provider = "kimi-coding"

        prior_assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{\"command\":\"date\"}"},
                }
            ],
        }
        tool_result = {"role": "tool", "tool_call_id": "c1", "content": "Tue Apr 21"}
        final_resp = _mock_response(content="done", finish_reason="stop")
        agent.client.chat.completions.create.return_value = final_resp

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "next step",
                conversation_history=[prior_assistant, tool_result],
            )

        assert result["completed"] is True
        sent_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
        replayed_assistant = next(msg for msg in sent_messages if msg.get("role") == "assistant")
        assert replayed_assistant["role"] == "assistant"
        assert replayed_assistant["tool_calls"][0]["function"]["name"] == "terminal"
        assert "reasoning_content" in replayed_assistant
        assert replayed_assistant["reasoning_content"] == " "

    def test_explicit_reasoning_content_beats_normalized_reasoning_on_replay(self, agent):
        self._setup_agent(agent)
        # Precedence (explicit reasoning_content wins over the 'reasoning'
        # field) only matters on a provider that echoes reasoning_content
        # back — strict providers strip the field entirely. Pin a
        # reasoning provider so the precedence is observable.
        agent.base_url = "https://api.kimi.com/coding/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.provider = "kimi-coding"
        prior_assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{\"q\":\"test\"}"},
                }
            ],
            "reasoning": "summary reasoning",
            "reasoning_content": "provider-native scratchpad",
        }
        tool_result = {"role": "tool", "tool_call_id": "c1", "content": "ok"}
        final_resp = _mock_response(content="done", finish_reason="stop")
        agent.client.chat.completions.create.return_value = final_resp

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "next step",
                conversation_history=[prior_assistant, tool_result],
            )

        assert result["completed"] is True
        sent_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
        replayed_assistant = next(msg for msg in sent_messages if msg.get("role") == "assistant")
        assert replayed_assistant["reasoning_content"] == "provider-native scratchpad"
