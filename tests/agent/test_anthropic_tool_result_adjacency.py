"""Anthropic tool_use/tool_result adjacency invariant.

Regression coverage for a bug that permanently bricked live agent sessions:
a text block injected ahead of the tool_result blocks in a user message made
Anthropic reject the request with HTTP 400 ("`tool_use` ids were found
without `tool_result` blocks immediately after"). The error was classified
non-retryable, the malformed history was persisted, and every subsequent
prompt rebuilt the same invalid payload — the agent went silent for good
while context grew on each retry (70k -> 151k tokens over 5 retries).

Two layers are asserted here, matching the two-layer fix:

1. Conversion (``_hoist_tool_results_to_front``) — tool_results are ordered
   first, so the payload is valid on the wire.
2. Classification (``FailoverReason.tool_result_adjacency``) — the 400 is
   retryable, so a repairable payload-shape error can never be terminal.

These are behavior contracts (invariants over the output shape), not
snapshots of a current value.
"""

import pytest

from agent.anthropic_message_convert import (
    _hoist_tool_results_to_front,
    convert_messages_to_anthropic,
)
from agent.error_classifier import FailoverReason, classify_api_error


class _MockAPIError(Exception):
    """Simulates an OpenAI/Anthropic SDK APIStatusError."""

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


def _assistant_with_tool_calls(*ids):
    return {
        "role": "assistant",
        "content": "working on it",
        "tool_calls": [
            {
                "id": tid,
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }
            for tid in ids
        ],
    }


def _block_types(message):
    content = message["content"]
    if not isinstance(content, list):
        return []
    return [b.get("type") for b in content]


def assert_adjacency_invariant(result):
    """Every tool_use turn is followed by a user turn LEADING with tool_result.

    This is the exact condition Anthropic enforces: presence of the results in
    the next message is not enough, they must come first.

    Fails if no tool_use survives conversion — otherwise the assertion is
    vacuous and the test passes on broken code. (Caught by mutation testing:
    an earlier version of these tests passed with the hoist disabled, because
    the orphan-strip pass had removed the tool_use blocks entirely.)
    """
    checked = 0
    for i, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        tool_use_ids = [
            b.get("id") for b in m["content"] if b.get("type") == "tool_use"
        ]
        if not tool_use_ids:
            continue
        checked += 1
        assert i + 1 < len(result), "tool_use turn has no following message"
        nxt = result[i + 1]
        assert nxt["role"] == "user"
        assert isinstance(nxt["content"], list)
        assert nxt["content"][0].get("type") == "tool_result", (
            f"tool_result must lead the user message, got "
            f"{_block_types(nxt)}"
        )
        adjacent = [
            b.get("tool_use_id")
            for b in nxt["content"]
            if b.get("type") == "tool_result"
        ]
        for tid in tool_use_ids:
            assert tid in adjacent, f"{tid} has no adjacent tool_result"
    assert checked > 0, (
        "no tool_use survived conversion — the adjacency assertion would be "
        "vacuous, so this test proves nothing"
    )


class TestHoistToolResultsToFront:
    def test_injected_text_before_tool_results_is_reordered(self):
        """The production bug: turn-time injection prepends a text block."""
        result = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "..."},
                    {"type": "tool_use", "id": "toolu_A", "name": "terminal", "input": {}},
                    {"type": "tool_use", "id": "toolu_B", "name": "terminal", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<injected memory/plugin context>"},
                    {"type": "tool_result", "tool_use_id": "toolu_A", "content": "a"},
                    {"type": "tool_result", "tool_use_id": "toolu_B", "content": "b"},
                ],
            },
        ]
        _hoist_tool_results_to_front(result)

        assert _block_types(result[1]) == ["tool_result", "tool_result", "text"]
        assert_adjacency_invariant(result)

    def test_injected_text_is_preserved_not_dropped(self):
        """Reordering must be lossless — the injected context still ships."""
        result = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "IMPORTANT CONTEXT"},
                    {"type": "tool_result", "tool_use_id": "toolu_A", "content": "a"},
                ],
            },
        ]
        _hoist_tool_results_to_front(result)

        texts = [b["text"] for b in result[0]["content"] if b.get("type") == "text"]
        assert "IMPORTANT CONTEXT" in texts

    def test_already_ordered_message_is_left_byte_identical(self):
        """Prompt caching depends on not rewriting well-formed messages."""
        blocks = [
            {"type": "tool_result", "tool_use_id": "toolu_A", "content": "a"},
            {"type": "text", "text": "trailing"},
        ]
        result = [{"role": "user", "content": blocks}]
        _hoist_tool_results_to_front(result)

        assert result[0]["content"] is blocks

    def test_message_without_tool_results_is_untouched(self):
        blocks = [{"type": "text", "text": "just a question"}]
        result = [{"role": "user", "content": blocks}]
        _hoist_tool_results_to_front(result)

        assert result[0]["content"] is blocks

    def test_assistant_messages_are_not_reordered(self):
        blocks = [
            {"type": "text", "text": "thinking"},
            {"type": "tool_use", "id": "toolu_A", "name": "t", "input": {}},
        ]
        result = [{"role": "assistant", "content": blocks}]
        _hoist_tool_results_to_front(result)

        assert _block_types(result[0]) == ["text", "tool_use"]

    @pytest.mark.parametrize("content", ["plain string", [], None])
    def test_non_list_and_empty_content_do_not_raise(self, content):
        result = [{"role": "user", "content": content}]
        _hoist_tool_results_to_front(result)  # must not raise


class TestConversionPipelineOrdering:
    def test_parallel_tool_calls_satisfy_adjacency(self):
        messages = [
            {"role": "system", "content": "sys"},
            _assistant_with_tool_calls("toolu_A", "toolu_B"),
            {"role": "tool", "tool_call_id": "toolu_A", "content": "a"},
            {"role": "tool", "tool_call_id": "toolu_B", "content": "b"},
        ]
        _system, result = convert_messages_to_anthropic(messages)
        assert_adjacency_invariant(result)

    def test_interleaved_user_message_satisfies_adjacency(self):
        """A user message landing between tool_use and its results."""
        messages = [
            {"role": "system", "content": "sys"},
            _assistant_with_tool_calls("toolu_A"),
            {"role": "user", "content": "wait, also check X"},
            {"role": "tool", "tool_call_id": "toolu_A", "content": "a"},
        ]
        _system, result = convert_messages_to_anthropic(messages)
        # The orphan-strip pass drops the tool_use here rather than repairing
        # it, so adjacency is trivially satisfied. Assert the weaker but
        # non-vacuous property: no tool_use is left stranded without results.
        for i, m in enumerate(result):
            if m.get("role") == "assistant" and isinstance(m.get("content"), list):
                assert not [
                    b for b in m["content"] if b.get("type") == "tool_use"
                ], "stranded tool_use survived conversion"

    def test_turn_time_injection_after_conversion_is_repaired(self):
        """The exact production failure path.

        Ephemeral context (memory prefetch / plugin ``pre_llm_call`` hooks)
        is prepended to the user message *after* ``convert_messages_to_anthropic``
        has already run, so the strip pass never sees it. The hoist is the
        only thing standing between that injection and a hard HTTP 400.
        """
        messages = [
            {"role": "system", "content": "sys"},
            _assistant_with_tool_calls("toolu_A", "toolu_B"),
            {"role": "tool", "tool_call_id": "toolu_A", "content": "a"},
            {"role": "tool", "tool_call_id": "toolu_B", "content": "b"},
        ]
        _system, result = convert_messages_to_anthropic(messages)
        assert_adjacency_invariant(result)  # clean before injection

        # Simulate compose_user_api_content prepending ephemeral context.
        result[1]["content"].insert(
            0, {"type": "text", "text": "<system-reminder>...</system-reminder>"}
        )
        _hoist_tool_results_to_front(result)

        assert_adjacency_invariant(result)

    def test_hoist_is_wired_into_the_conversion_pipeline(self, monkeypatch):
        """The hoist must actually run inside convert_messages_to_anthropic.

        The repair function being correct is worthless if nothing calls it.
        This is the assertion that fails if the call site is removed — the
        shape-level tests above cannot catch that, because the orphan-strip
        pass independently normalizes most inputs (verified by mutation
        testing: deleting the call site left every other test green).
        """
        import agent.anthropic_message_convert as adapter

        calls = []
        real = adapter._hoist_tool_results_to_front

        def spy(result):
            calls.append(len(result))
            return real(result)

        monkeypatch.setattr(adapter, "_hoist_tool_results_to_front", spy)

        messages = [
            {"role": "system", "content": "sys"},
            _assistant_with_tool_calls("toolu_A"),
            {"role": "tool", "tool_call_id": "toolu_A", "content": "a"},
        ]
        adapter.convert_messages_to_anthropic(messages)

        assert calls, (
            "_hoist_tool_results_to_front was never called by "
            "convert_messages_to_anthropic — the adjacency repair is dead code"
        )

    def test_normal_followup_turn_satisfies_adjacency(self):
        messages = [
            {"role": "system", "content": "sys"},
            _assistant_with_tool_calls("toolu_A"),
            {"role": "tool", "tool_call_id": "toolu_A", "content": "a"},
            {"role": "user", "content": "next question"},
        ]
        _system, result = convert_messages_to_anthropic(messages)
        assert_adjacency_invariant(result)


class TestAdjacencyErrorIsRetryable:
    ERROR = (
        "messages.2: `tool_use` ids were found without `tool_result` blocks "
        "immediately after: toolu_0113BvY3H7gBhezMbd6u9r6v. Each `tool_use` "
        "block must have a corresponding `tool_result` block in the next message."
    )

    def _classify(self):
        return classify_api_error(
            _MockAPIError(self.ERROR, status_code=400),
            provider="anthropic",
            model="claude-opus-5",
        )

    def test_classified_as_tool_result_adjacency(self):
        assert self._classify().reason == FailoverReason.tool_result_adjacency

    def test_is_retryable_so_it_can_never_be_terminal(self):
        """The core regression: this 400 must not abort the session.

        Non-retryable classification is what turned one malformed pair into a
        permanently dead agent.
        """
        assert self._classify().retryable is True

    def test_does_not_trigger_compression(self):
        """Compressing is the wrong recovery — the shape is repaired on retry."""
        assert self._classify().should_compress is False

    def test_unrelated_400_is_not_misclassified(self):
        classified = classify_api_error(
            _MockAPIError("messages: invalid role 'foo'", status_code=400),
            provider="anthropic",
            model="claude-opus-5",
        )
        assert classified.reason != FailoverReason.tool_result_adjacency
