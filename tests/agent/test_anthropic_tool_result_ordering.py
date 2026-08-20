"""Regression tests for #79147 — leading text block ahead of tool_result.

Anthropic requires the ``tool_result`` blocks answering an assistant's
``tool_use`` to be FIRST in the following user message. When an injected
``<system-reminder>`` text block lands ahead of them the request is rejected
with a non-retryable HTTP 400 and the session is permanently bricked, because
every retry replays the same malformed prefix.

``_strip_orphaned_tool_blocks`` compares ID *sets* between adjacent messages,
so a present-but-misordered pair looks healthy to it. These tests lock in the
ordering invariant that ``_hoist_tool_results_to_front`` enforces.
"""

import pytest

from agent.anthropic_adapter import (
    _hoist_tool_results_to_front,
    _merge_consecutive_roles,
    _strip_orphaned_tool_blocks,
)


def _tool_result_messages_are_wellformed(messages):
    """Every tool_result must sit in a contiguous run at the START of its message.

    Checking only ``content[0]`` is too weak: ``[tool_result, text, tool_result]``
    leads with a tool_result but still strands the second one behind the text.
    """
    for msg in messages:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        flags = [
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in msg["content"]
        ]
        n_results = sum(flags)
        if n_results and not all(flags[:n_results]):
            return False
    return True


class TestHoistToolResults:
    def test_leading_text_block_is_hoisted_behind_tool_result(self):
        """The exact shape captured from the bricked agent (#79147)."""
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>ctx</system-reminder>"},
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
            ]},
        ]

        _hoist_tool_results_to_front(messages)

        assert [b["type"] for b in messages[1]["content"]] == ["tool_result", "text"]
        assert _tool_result_messages_are_wellformed(messages)

    def test_injected_context_is_preserved_not_dropped(self):
        """Reorder, never discard — the text is real context for the model."""
        reminder = "<system-reminder>important standing instructions</system-reminder>"
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": reminder},
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
            ]},
        ]

        _hoist_tool_results_to_front(messages)

        assert any(
            b.get("type") == "text" and b.get("text") == reminder
            for b in messages[1]["content"]
        )

    def test_parallel_tool_batch_preserves_relative_result_order(self):
        """Multiple tool_results keep their order; only the text moves."""
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "a", "name": "x", "input": {}},
                {"type": "tool_use", "id": "b", "name": "y", "input": {}},
                {"type": "tool_use", "id": "c", "name": "z", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "reminder"},
                {"type": "tool_result", "tool_use_id": "a", "content": "1"},
                {"type": "tool_result", "tool_use_id": "b", "content": "2"},
                {"type": "tool_result", "tool_use_id": "c", "content": "3"},
            ]},
        ]

        _hoist_tool_results_to_front(messages)

        ids = [b["tool_use_id"] for b in messages[1]["content"]
               if b["type"] == "tool_result"]
        assert ids == ["a", "b", "c"]
        assert messages[1]["content"][-1]["type"] == "text"

    def test_interleaved_results_are_compacted_into_leading_run(self):
        """A leading tool_result does not mean the message is valid.

        ``[tool_result, text, tool_result]`` starts at index 0, so a fast path
        keyed on "is the first tool_result at index 0" skips it and strands the
        second result behind the text block (review feedback on #79158).
        """
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "a", "name": "x", "input": {}},
                {"type": "tool_use", "id": "b", "name": "y", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "1"},
                {"type": "text", "text": "reminder"},
                {"type": "tool_result", "tool_use_id": "b", "content": "2"},
            ]},
        ]

        _hoist_tool_results_to_front(messages)

        assert [b["type"] for b in messages[1]["content"]] == [
            "tool_result", "tool_result", "text",
        ]
        assert [b["tool_use_id"] for b in messages[1]["content"]
                if b["type"] == "tool_result"] == ["a", "b"]
        assert _tool_result_messages_are_wellformed(messages)

    def test_trailing_result_after_multiple_texts_is_compacted(self):
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "1"},
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
                {"type": "tool_result", "tool_use_id": "b", "content": "2"},
            ]},
        ]

        _hoist_tool_results_to_front(messages)

        assert [b["type"] for b in messages[0]["content"]] == [
            "tool_result", "tool_result", "text", "text",
        ]
        assert [b["text"] for b in messages[0]["content"]
                if b["type"] == "text"] == ["one", "two"]

    def test_wellformed_helper_rejects_interleaved_shape(self):
        """The helper itself must not call the interleaved shape valid."""
        interleaved = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "1"},
                {"type": "text", "text": "reminder"},
                {"type": "tool_result", "tool_use_id": "b", "content": "2"},
            ]},
        ]

        assert not _tool_result_messages_are_wellformed(interleaved)

    def test_multiple_leading_blocks_all_move_behind_results(self):
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t", "name": "x", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
                {"type": "tool_result", "tool_use_id": "t", "content": "ok"},
            ]},
        ]

        _hoist_tool_results_to_front(messages)

        assert [b["type"] for b in messages[1]["content"]] == [
            "tool_result", "text", "text",
        ]
        assert [b["text"] for b in messages[1]["content"] if b["type"] == "text"] == [
            "one", "two",
        ]


class TestNoOpOnHealthyInput:
    """Prompt caching is sacred: valid messages must not be rewritten."""

    def test_already_wellformed_message_is_left_byte_identical(self):
        content = [
            {"type": "tool_result", "tool_use_id": "t", "content": "ok"},
            {"type": "text", "text": "trailing note"},
        ]
        messages = [{"role": "user", "content": content}]
        before = [dict(b) for b in content]

        _hoist_tool_results_to_front(messages)

        assert messages[0]["content"] is content, "must not reallocate valid content"
        assert messages[0]["content"] == before

    def test_message_without_tool_results_is_untouched(self):
        content = [{"type": "text", "text": "just talking"}]
        messages = [{"role": "user", "content": content}]

        _hoist_tool_results_to_front(messages)

        assert messages[0]["content"] is content

    @pytest.mark.parametrize("content", ["plain string", None, 42])
    def test_non_list_content_does_not_raise(self, content):
        messages = [{"role": "user", "content": content}]
        _hoist_tool_results_to_front(messages)
        assert messages[0]["content"] == content

    def test_assistant_messages_are_untouched(self):
        content = [
            {"type": "text", "text": "thinking out loud"},
            {"type": "tool_use", "id": "t", "name": "x", "input": {}},
        ]
        messages = [{"role": "assistant", "content": content}]

        _hoist_tool_results_to_front(messages)

        assert messages[0]["content"] is content

    def test_repair_is_idempotent(self):
        """A repaired prefix must stay stable so the NEXT turn still caches."""
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "reminder"},
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            {"type": "text", "text": "mid"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "ok2"},
        ]}]
        _hoist_tool_results_to_front(messages)
        once = [dict(b) for b in messages[0]["content"]]

        _hoist_tool_results_to_front(messages)

        assert messages[0]["content"] == once

    def test_reorder_never_drops_or_mutates_a_block(self):
        """Every original block survives verbatim -- this reorders, never edits."""
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "A"},
            {"type": "tool_result", "tool_use_id": "t", "content": "R"},
            {"type": "text", "text": "B"},
        ]}]
        original = sorted(map(repr, messages[0]["content"]))

        _hoist_tool_results_to_front(messages)

        assert sorted(map(repr, messages[0]["content"])) == original


class TestPipelineOrdering:
    def test_hoist_survives_the_consecutive_role_merge(self):
        """Merging a text-only user turn onto a tool_result turn re-creates the
        violation, so hoisting must run after _merge_consecutive_roles."""
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t", "name": "x", "input": {}},
            ]},
            {"role": "user", "content": [{"type": "text", "text": "reminder"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": "ok"},
            ]},
        ]

        merged = _merge_consecutive_roles(messages)
        _hoist_tool_results_to_front(merged)

        assert _tool_result_messages_are_wellformed(merged)

    def test_strip_orphaned_alone_does_not_catch_misordering(self):
        """Documents the gap this fix closes — guards against a future
        'simplification' that assumes the strip pass already covers it."""
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t", "name": "x", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "reminder"},
                {"type": "tool_result", "tool_use_id": "t", "content": "ok"},
            ]},
        ]

        _strip_orphaned_tool_blocks(messages)
        assert not _tool_result_messages_are_wellformed(messages)

        _hoist_tool_results_to_front(messages)
        assert _tool_result_messages_are_wellformed(messages)
