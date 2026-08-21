"""Cache-aware summarization input assembly (feat(compression): cache-aware compaction).

The auxiliary summarization call must be a GENUINE PREFIX of the last routed
request — the conversation's own system prompt, the tool schemas, then the
protected head + compacted region as real chat messages in order, with the
summarization instruction as the FINAL user message. That lets the provider
reuse its KV cache for the replayed prefix; only the instruction + generated
summary are novel tokens.

Contracts under test:
- message order: system → head → region → instruction (last)
- tools are replayed ahead of the prefix when provided
- region messages stay STRUCTURED (role/content/tool_calls/tool_call_id),
  never flattened into the instruction text
- per-message safety bounds still apply (redaction, think-strip, truncation)
- the aggregate input cap preserves edges and marks the omitted middle
- legacy callers without prefix/tools still work (region + instruction)
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import (
    ContextCompressor,
    SUMMARY_PREFIX,
    HISTORICAL_TASK_HEADING,
)

SECRET = "sk-proj-" + ("a" * 40)


def _compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100000,
    ):
        return ContextCompressor(model="test/model", quiet_mode=True)


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _call_kwargs(c, turns, **kwargs):
    """Run _generate_summary with a mocked call_llm and return its kwargs."""
    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("## Goal\nsafe summary"),
    ) as mock_call:
        result = c._generate_summary(turns, **kwargs)
    assert result is not None
    assert result.startswith(SUMMARY_PREFIX)
    return mock_call.call_args.kwargs


class TestSummarizerWireMessage:
    def test_plain_string_content_preserved(self):
        c = _compressor()
        out = c._summarizer_wire_message({"role": "user", "content": "hello"})
        assert out == {"role": "user", "content": "hello"}

    def test_secrets_redacted(self):
        c = _compressor()
        out = c._summarizer_wire_message(
            {"role": "user", "content": f"deploy with {SECRET}"}
        )
        # _redact_compaction_text masks the secret (keeps a short fragment);
        # the full value must never reach the summarizer.
        assert SECRET not in out["content"]
        assert "sk-proj-" not in out["content"]

    def test_content_list_flattened_with_image_label(self):
        c = _compressor()
        out = c._summarizer_wire_message(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image_url", "image_url": {"url": "https://x.io/i.png"}},
                    {"type": "input_audio", "data": "..."},
                ],
            }
        )
        assert out["content"] == "look at this\n[image: https://x.io/i.png]\n[input_audio]"

    def test_media_directive_replaced(self):
        c = _compressor()
        out = c._summarizer_wire_message(
            {"role": "user", "content": "see MEDIA:C:/tmp/x.png here"}
        )
        assert "[media attachment]" in out["content"]
        assert "MEDIA:" not in out["content"]

    def test_assistant_think_blocks_stripped(self):
        c = _compressor()
        out = c._summarizer_wire_message(
            {
                "role": "assistant",
                "content": "<think>scratch work</think>real answer",
            }
        )
        assert "<think>" not in out["content"]
        assert "real answer" in out["content"]

    def test_tool_calls_preserved_with_redacted_args(self):
        c = _compressor()
        out = c._summarizer_wire_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": f'{{"command": "echo {SECRET}"}}',
                        },
                    }
                ],
            }
        )
        assert out["role"] == "assistant"
        assert out["tool_calls"][0]["id"] == "call-1"
        assert out["tool_calls"][0]["function"]["name"] == "terminal"
        assert SECRET not in out["tool_calls"][0]["function"]["arguments"]

    def test_tool_message_keeps_tool_call_id(self):
        c = _compressor()
        out = c._summarizer_wire_message(
            {"role": "tool", "tool_call_id": "call-1", "content": "result"}
        )
        assert out["tool_call_id"] == "call-1"
        assert out["content"] == "result"

    def test_long_content_truncated(self):
        c = _compressor()
        big = "x" * (c._CONTENT_MAX * 2)
        out = c._summarizer_wire_message({"role": "user", "content": big})
        assert "..." in out["content"]
        assert len(out["content"]) < c._CONTENT_MAX + 100

    def test_input_not_mutated(self):
        c = _compressor()
        msg = {"role": "user", "content": f"secret {SECRET}"}
        c._summarizer_wire_message(msg)
        assert msg["content"] == f"secret {SECRET}"


class TestBoundSummarizerMessages:
    def test_under_cap_unchanged(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        out = ContextCompressor._bound_summarizer_messages(msgs)
        assert out is msgs

    def test_over_cap_keeps_edges_and_marks_middle(self, monkeypatch):
        monkeypatch.setattr(
            ContextCompressor, "_SUMMARY_INPUT_MAX_CHARS", 200
        )
        msgs = [
            {"role": "user", "content": "head " * 30},
            {"role": "assistant", "content": "mid1 " * 30},
            {"role": "assistant", "content": "mid2 " * 30},
            {"role": "user", "content": "tail " * 30},
        ]
        out = ContextCompressor._bound_summarizer_messages(msgs)
        # Edges kept, middle dropped, marker appended to the last pre-gap msg.
        assert out[0]["content"].startswith("head")
        assert out[-1]["content"].startswith("tail")
        assert any("summary input truncated" in m["content"] for m in out)
        total = sum(len(m.get("content") or "") for m in out)
        assert total < 200 + 500  # marker inflates the last kept message

    def test_roles_preserved_through_cap(self, monkeypatch):
        monkeypatch.setattr(
            ContextCompressor, "_SUMMARY_INPUT_MAX_CHARS", 100
        )
        msgs = [
            {"role": "user", "content": "u" * 60},
            {"role": "tool", "tool_call_id": "c1", "content": "t" * 60},
            {"role": "assistant", "content": "a" * 60},
        ]
        out = ContextCompressor._bound_summarizer_messages(msgs)
        assert out[0]["role"] == "user"
        assert out[-1]["role"] == "assistant"
        assert len(out) == 2


class TestGenerateSummaryCacheAwareInput:
    def _head_and_region(self):
        head = [
            {"role": "system", "content": "You are Hermes. Be helpful."},
            {"role": "user", "content": "protected first ask"},
            {"role": "assistant", "content": "protected first answer"},
        ]
        region = [
            {"role": "user", "content": "middle ask"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-9",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"cmd": "ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-9", "content": "file list"},
            {"role": "assistant", "content": "middle answer"},
        ]
        return head, region

    def test_message_order_system_head_region_instruction_last(self):
        c = _compressor()
        head, region = self._head_and_region()
        kwargs = _call_kwargs(c, region, prefix_messages=head)
        messages = kwargs["messages"]

        assert messages[0] == {"role": "system", "content": "You are Hermes. Be helpful."}
        # Head replayed next, in order.
        assert messages[1] == {"role": "user", "content": "protected first ask"}
        assert messages[2] == {"role": "assistant", "content": "protected first answer"}
        # Region follows, structured (tool_calls / tool_call_id preserved).
        assert messages[3] == {"role": "user", "content": "middle ask"}
        assert messages[4]["role"] == "assistant"
        assert messages[4]["tool_calls"][0]["id"] == "call-9"
        assert messages[5] == {
            "role": "tool",
            "tool_call_id": "call-9",
            "content": "file list",
        }
        assert messages[6] == {"role": "assistant", "content": "middle answer"}
        # Instruction is the FINAL message.
        assert messages[-1]["role"] == "user"
        assert messages[-1] is not messages[3]  # separate message, not merged
        instruction = messages[-1]["content"]
        assert "You are a summarization agent" in instruction
        assert HISTORICAL_TASK_HEADING in instruction
        assert "## Goal" in instruction
        # Region content is NOT flattened into the instruction.
        assert "middle ask" not in instruction
        assert "file list" not in instruction
        assert "[ASSISTANT]:" not in instruction
        assert "TURNS TO SUMMARIZE" not in instruction

    def test_tools_replayed_before_prefix(self):
        c = _compressor()
        head, region = self._head_and_region()
        tools = [
            {
                "type": "function",
                "function": {"name": "terminal", "description": "run a command"},
            }
        ]
        kwargs = _call_kwargs(c, region, prefix_messages=head, tools=tools)
        assert kwargs["tools"] == tools
        # Tools don't appear as messages; messages still start with system.
        messages = kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are Hermes. Be helpful."

    def test_no_prefix_legacy_shape(self):
        c = _compressor()
        region = [
            {"role": "user", "content": "ask"},
            {"role": "assistant", "content": "answer"},
        ]
        kwargs = _call_kwargs(c, region)
        messages = kwargs["messages"]
        # No system, no head: region first, instruction last.
        assert messages[0] == {"role": "user", "content": "ask"}
        assert messages[1] == {"role": "assistant", "content": "answer"}
        assert messages[2]["role"] == "user"
        assert messages[2]["content"].startswith("You are a summarization agent")
        assert "tools" not in kwargs

    def test_region_without_system_prompt_in_prefix(self):
        c = _compressor()
        head = [{"role": "user", "content": "head only, no system"}]
        region = [{"role": "user", "content": "region ask"}]
        kwargs = _call_kwargs(c, region, prefix_messages=head)
        messages = kwargs["messages"]
        assert messages[0] == {"role": "user", "content": "head only, no system"}
        assert messages[1] == {"role": "user", "content": "region ask"}
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"].startswith("You are a summarization agent")

    def test_iterative_update_keeps_previous_summary_in_instruction(self):
        c = _compressor()
        c._previous_summary = "PREVIOUS-SUMMARY-BODY unique"
        region = [{"role": "user", "content": "REGION-UNIQUE-TOKEN"}]
        kwargs = _call_kwargs(c, region)
        instruction = kwargs["messages"][-1]["content"]
        assert "PREVIOUS-SUMMARY-BODY unique" in instruction
        assert "You are updating a context compaction summary" in instruction
        assert "PREVIOUS SUMMARY:" in instruction
        # The region is not duplicated into the instruction.
        assert "REGION-UNIQUE-TOKEN" not in instruction

    def test_memory_context_lands_in_instruction(self):
        c = _compressor()
        region = [{"role": "user", "content": "turn"}]
        kwargs = _call_kwargs(c, region, memory_context="memory provider payload")
        instruction = kwargs["messages"][-1]["content"]
        assert "MEMORY PROVIDER CONTEXT:" in instruction
        assert "memory provider payload" in instruction
        # Not injected into the replayed region.
        assert kwargs["messages"][0]["content"] == "turn"

    def test_no_wire_max_tokens_cap(self):
        c = _compressor()
        kwargs = _call_kwargs(c, [{"role": "user", "content": "turn"}])
        assert "max_tokens" not in kwargs


class TestCompressCacheAwarePlumbing:
    def test_compress_forwards_prefix_and_tools(self):
        c = _compressor()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ask 1"},
            {"role": "assistant", "content": "ans 1"},
            {"role": "user", "content": "ask 2"},
            {"role": "assistant", "content": "ans 2"},
            {"role": "user", "content": "ask 3"},
            {"role": "assistant", "content": "ans 3"},
            {"role": "user", "content": "tail ask"},
            {"role": "assistant", "content": "tail ans"},
        ]
        tools = [{"type": "function", "function": {"name": "x"}}]

        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response("## Goal\ncompressed"),
        ) as mock_call:
            with patch(
                "agent.context_compressor.ContextCompressor._derive_auto_focus_topic",
                return_value=None,
            ):
                result = c.compress(messages, tools=tools)

        assert len(result) < len(messages)
        assert mock_call.called
        kwargs = mock_call.call_args.kwargs
        assert kwargs["tools"] == tools
        sent = kwargs["messages"]
        # System first, then the protected head before the region.
        assert sent[0] == {"role": "system", "content": "sys"}
        assert sent[1] == {"role": "user", "content": "ask 1"}
        # Instruction always last.
        assert sent[-1]["role"] == "user"
        assert sent[-1]["content"].startswith("You are a summarization agent")

    def test_compress_without_tools_omits_key(self):
        c = _compressor()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ask 1"},
            {"role": "assistant", "content": "ans 1"},
            {"role": "user", "content": "ask 2"},
            {"role": "assistant", "content": "ans 2"},
            {"role": "user", "content": "ask 3"},
            {"role": "assistant", "content": "ans 3"},
            {"role": "user", "content": "tail ask"},
            {"role": "assistant", "content": "tail ans"},
        ]
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response("## Goal\ncompressed"),
        ) as mock_call:
            with patch(
                "agent.context_compressor.ContextCompressor._derive_auto_focus_topic",
                return_value=None,
            ):
                c.compress(messages)

        assert "tools" not in mock_call.call_args.kwargs
