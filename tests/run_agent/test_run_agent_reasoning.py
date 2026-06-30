"""Unit tests for run_agent.py (AIAgent) — reasoning extraction, think-block stripping, reasoning replay.

Split out of the former monolithic ``tests/run_agent/test_run_agent.py`` (which
outgrew the per-file CI wall-clock cap). Shared fixtures live in ``conftest.py``;
mock-builders in ``_run_agent_helpers.py``.
"""

from unittest.mock import patch
import pytest
from run_agent import AIAgent

from tests.run_agent._run_agent_helpers import (
    _mock_assistant_msg,
    _mock_response,
)


class TestHasContentAfterThinkBlock:
    def test_none_returns_false(self, agent):
        assert agent._has_content_after_think_block(None) is False

    def test_empty_returns_false(self, agent):
        assert agent._has_content_after_think_block("") is False

    def test_only_think_block_returns_false(self, agent):
        assert agent._has_content_after_think_block("<think>reasoning</think>") is False

    def test_content_after_think_returns_true(self, agent):
        assert (
            agent._has_content_after_think_block("<think>r</think> actual answer")
            is True
        )

    def test_no_think_block_returns_true(self, agent):
        assert agent._has_content_after_think_block("just normal content") is True


class TestStripThinkBlocks:
    def test_none_returns_empty(self, agent):
        assert agent._strip_think_blocks(None) == ""

    def test_no_blocks_unchanged(self, agent):
        assert agent._strip_think_blocks("hello world") == "hello world"

    def test_single_block_removed(self, agent):
        result = agent._strip_think_blocks("<think>reasoning</think> answer")
        assert "reasoning" not in result
        assert "answer" in result

    def test_multiline_block_removed(self, agent):
        text = "<think>\nline1\nline2\n</think>\nvisible"
        result = agent._strip_think_blocks(text)
        assert "line1" not in result
        assert "visible" in result

    def test_orphaned_closing_think_tag(self, agent):
        result = agent._strip_think_blocks("some reasoning</think>actual answer")
        assert "</think>" not in result
        assert "actual answer" in result

    def test_orphaned_closing_thinking_tag(self, agent):
        result = agent._strip_think_blocks("reasoning</thinking>answer")
        assert "</thinking>" not in result
        assert "answer" in result

    def test_orphaned_opening_think_tag(self, agent):
        result = agent._strip_think_blocks("<think>orphaned reasoning without close")
        assert "<think>" not in result

    def test_mixed_orphaned_and_paired_tags(self, agent):
        text = "stray</think><think>paired reasoning</think> visible"
        result = agent._strip_think_blocks(text)
        assert "</think>" not in result
        assert "<think>" not in result
        assert "visible" in result

    def test_thought_block_removed(self, agent):
        """Gemma 4 uses <thought> tags for inline reasoning."""
        result = agent._strip_think_blocks("<thought>internal reasoning</thought> answer")
        assert "internal reasoning" not in result
        assert "<thought>" not in result
        assert "answer" in result

    def test_orphaned_thought_tag(self, agent):
        result = agent._strip_think_blocks("<thought>orphaned reasoning without close")
        assert "<thought>" not in result

    # ─── Unterminated-block coverage (#8878, #9568, #10408) ──────────────
    # Reasoning models served via NIM / MiniMax M2.7 frequently drop the
    # closing tag, leaking raw reasoning into assistant content. The open
    # tag appears at a block boundary (start of text or after a newline);
    # everything from that tag to end-of-string is stripped.

    def test_unterminated_think_block_content_stripped(self, agent):
        """Content after unterminated <think> is fully stripped."""
        result = agent._strip_think_blocks("<think>orphaned reasoning without close")
        assert "orphaned reasoning" not in result
        assert result.strip() == ""

    def test_unterminated_thought_block_content_stripped(self, agent):
        """Gemma-style <thought> with no close is fully stripped."""
        result = agent._strip_think_blocks("<thought>orphaned reasoning without close")
        assert "orphaned reasoning" not in result
        assert result.strip() == ""

    def test_unterminated_multiline_block_stripped(self, agent):
        """Multi-line unterminated blocks are stripped in full."""
        result = agent._strip_think_blocks(
            "<think>\nmulti\nline\nreasoning\nthat never closes"
        )
        assert "multi" not in result
        assert "never closes" not in result

    def test_unterminated_block_after_answer_preserves_prefix(self, agent):
        """Visible answer before a line-starting unterminated tag is kept."""
        result = agent._strip_think_blocks(
            "Answer is 42.\n<think>actually let me reconsider"
        )
        assert "Answer is 42." in result
        assert "reconsider" not in result

    def test_inline_think_mention_in_prose_not_over_stripped(self, agent):
        """Mid-line `<think>` mentioned in prose must not swallow the rest
        of the content (the block-boundary check prevents this)."""
        text = "Use the <think> tag like this in your prose."
        result = agent._strip_think_blocks(text)
        # Block-boundary check prevents unterminated-strip from firing
        assert "prose" in result
        assert "Use the" in result

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

    def test_tool_call_block_stripped(self, agent):
        text = '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call> done'
        result = agent._strip_think_blocks(text)
        assert "<tool_call>" not in result
        assert "read_file" not in result
        assert "done" in result

    def test_function_calls_block_stripped(self, agent):
        text = '<function_calls>[{"name":"x"}]</function_calls>after'
        result = agent._strip_think_blocks(text)
        assert "<function_calls>" not in result
        assert "after" in result

    def test_gemma_function_name_block_stripped(self, agent):
        """Gemma-style: <function name="read"><parameter>...</parameter></function>."""
        text = (
            'Let me check the file.\n'
            '<function name="read_file"><parameter name="path">/tmp/x.md</parameter></function>\n'
            'Here is the result.'
        )
        result = agent._strip_think_blocks(text)
        assert '<function name="read_file">' not in result
        assert "/tmp/x.md" not in result
        assert "Let me check the file." in result
        assert "Here is the result." in result

    def test_gemma_function_multiline_payload_stripped(self, agent):
        text = (
            'Reading now.\n'
            '<function name="read_file">\n'
            '  <parameter name="path">/etc/passwd</parameter>\n'
            '</function>\n'
            'Done.'
        )
        result = agent._strip_think_blocks(text)
        assert "/etc/passwd" not in result
        assert "Reading now." in result
        assert "Done." in result

    def test_function_mention_in_prose_preserved(self, agent):
        """'Use <function> in JavaScript.' — no name attr, not at block boundary
        in a way that suggests tool call. Must survive."""
        text = "In JS you can use <function> declarations for hoisting."
        result = agent._strip_think_blocks(text)
        # Prose mention has no name="..." attribute -> not stripped
        assert "declarations for hoisting" in result

    def test_function_with_attr_in_middle_of_sentence_preserved(self, agent):
        """Docs example: 'Use <function name="x">...</function> in docs.'
        The sentence-middle position without a preceding punctuation block
        boundary means it is NOT stripped. Prose context remains."""
        text = 'You can write <function name="x">y</function> inline.'
        result = agent._strip_think_blocks(text)
        # Without a leading block boundary (no punctuation before), leaves intact
        assert "You can write" in result
        assert "inline" in result

    def test_stray_function_close_tag_removed(self, agent):
        text = "answer</function> trailing"
        result = agent._strip_think_blocks(text)
        assert "</function>" not in result
        assert "answer" in result
        assert "trailing" in result

    def test_dangling_function_open_tag_preserved(self, agent):
        """A streamed-but-truncated <function name="..."> block with no close
        is intentionally NOT stripped (OpenClaw's asymmetry). The tail of a
        streaming reply may still be valuable to the user."""
        text = 'Checking: <function name="read">'
        result = agent._strip_think_blocks(text)
        assert "Checking:" in result

    def test_mixed_reasoning_and_tool_call_both_stripped(self, agent):
        text = '<think>let me plan</think><tool_call>{"name":"x"}</tool_call>final answer'
        result = agent._strip_think_blocks(text)
        assert "let me plan" not in result
        assert "<tool_call>" not in result
        assert "final answer" in result


class TestExtractReasoning:
    def test_reasoning_field(self, agent):
        msg = _mock_assistant_msg(reasoning="thinking hard")
        assert agent._extract_reasoning(msg) == "thinking hard"

    def test_reasoning_content_field(self, agent):
        msg = _mock_assistant_msg(reasoning_content="deep thought")
        assert agent._extract_reasoning(msg) == "deep thought"

    def test_reasoning_details_array(self, agent):
        msg = _mock_assistant_msg(
            reasoning_details=[{"summary": "step-by-step analysis"}],
        )
        assert "step-by-step analysis" in agent._extract_reasoning(msg)

    def test_no_reasoning_returns_none(self, agent):
        msg = _mock_assistant_msg()
        assert agent._extract_reasoning(msg) is None

    def test_combined_reasoning(self, agent):
        msg = _mock_assistant_msg(
            reasoning="part1",
            reasoning_content="part2",
        )
        result = agent._extract_reasoning(msg)
        assert "part1" in result
        assert "part2" in result

    def test_deduplication(self, agent):
        msg = _mock_assistant_msg(
            reasoning="same text",
            reasoning_content="same text",
        )
        result = agent._extract_reasoning(msg)
        assert result == "same text"

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("<think>thinking hard</think>", "thinking hard"),
            ("<thinking>step by step</thinking>", "step by step"),
            (
                "<REASONING_SCRATCHPAD>scratch analysis</REASONING_SCRATCHPAD>",
                "scratch analysis",
            ),
        ],
    )
    def test_inline_reasoning_blocks_fallback(self, agent, content, expected):
        msg = _mock_assistant_msg(content=content)
        assert agent._extract_reasoning(msg) == expected

    def test_content_list_thinking_blocks_extracted(self, agent):
        """DeepSeek V4 Pro returns content as a typed-block list (issue #21944).

        Without this branch thinking text is silently dropped → HTTP 400 on
        the next turn ("thinking must be passed back to the API").
        """
        msg = _mock_assistant_msg(
            content=[
                {"type": "thinking", "thinking": "deep analysis here"},
                {"type": "output", "text": "final answer"},
            ]
        )
        result = agent._extract_reasoning(msg)
        assert result == "deep analysis here"

    def test_content_list_non_thinking_blocks_ignored(self, agent):
        """Non-thinking blocks in a content list must not be treated as reasoning."""
        msg = _mock_assistant_msg(
            content=[
                {"type": "text", "text": "just a regular response"},
            ]
        )
        assert agent._extract_reasoning(msg) is None

    def test_content_list_thinking_prefers_structured_field(self, agent):
        """Structured ``reasoning`` field wins over content-list thinking blocks."""
        msg = _mock_assistant_msg(
            reasoning="from structured field",
            content=[
                {"type": "thinking", "thinking": "from content list"},
            ],
        )
        result = agent._extract_reasoning(msg)
        # structured field was found first → content-list branch skipped
        assert result == "from structured field"


class TestReasoningReplayForStrictProviders:
    """Assistant replay must preserve provider-native reasoning fields."""

    def _setup_agent(self, agent):
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
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
        # Echo-back provider (Kimi): copy_reasoning_content_for_api preserves an
        # explicit reasoning_content verbatim and ignores the normalized
        # `reasoning` field. On non-echo-back providers (default OpenRouter)
        # the field is stripped entirely (refs #45655), which is a separate path.
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


class TestSupportsReasoningExtraBody:
    def _make_agent(self):
        agent = object.__new__(AIAgent)
        agent.provider = "openrouter"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = ""
        return agent

    def test_xiaomi_models_are_treated_as_reasoning_capable(self):
        agent = self._make_agent()
        for model in (
            "xiaomi/mimo-v2.5-pro",
            "xiaomi/mimo-v2.5",
            "xiaomi/mimo-v2-omni",
            "xiaomi/mimo-v2-pro",
            "xiaomi/mimo-v2-flash",
        ):
            agent.model = model
            assert agent._supports_reasoning_extra_body() is True, model
