"""Unit tests for AIAgent pre/post-LLM-call guardrails (issue #626).

Covers three static methods introduced in run_agent.py:
  - AIAgent._sanitize_api_messages()   — Phase 1: orphaned tool pair repair
  - AIAgent._cap_delegate_task_calls() — Phase 2a: subagent concurrency limit
  - AIAgent._deduplicate_tool_calls()  — Phase 2b: identical call deduplication
"""

import types

from run_agent import AIAgent
from tools.delegate_tool import MAX_CONCURRENT_CHILDREN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tc(name: str, arguments: str = "{}") -> types.SimpleNamespace:
    """Create a minimal tool_call SimpleNamespace mirroring the OpenAI SDK object."""
    tc = types.SimpleNamespace()
    tc.function = types.SimpleNamespace(name=name, arguments=arguments)
    return tc


def tool_result(call_id: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def assistant_dict_call(call_id: str, name: str = "terminal") -> dict:
    """Dict-style tool_call (as stored in message history, not the SDK object)."""
    return {"id": call_id, "function": {"name": name, "arguments": "{}"}}


# ---------------------------------------------------------------------------
# Phase 1 — _sanitize_api_messages
# ---------------------------------------------------------------------------

class TestSanitizeApiMessages:
    """AIAgent._sanitize_api_messages() repairs orphaned tool pairs."""

    def test_orphaned_result_removed(self):
        """A role=tool message whose call_id has no matching assistant entry is dropped."""
        msgs = [
            {"role": "assistant", "tool_calls": [assistant_dict_call("c1")]},
            tool_result("c1"),
            tool_result("c_ORPHAN"),   # no matching call — must be removed
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert len(out) == 2
        assert all(m.get("tool_call_id") != "c_ORPHAN" for m in out)

    def test_orphaned_call_gets_stub_result(self):
        """An assistant tool_call with no matching result gets a synthetic stub appended."""
        msgs = [
            {"role": "assistant", "tool_calls": [assistant_dict_call("c2")]},
            # deliberately no role=tool message
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert len(out) == 2
        stub = out[1]
        assert stub["role"] == "tool"
        assert stub["tool_call_id"] == "c2"
        assert stub["content"]  # non-empty placeholder

    def test_clean_messages_pass_through_unchanged(self):
        """A well-formed message list is returned as-is (same object)."""
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "tool_calls": [assistant_dict_call("c3")]},
            tool_result("c3"),
            {"role": "assistant", "content": "done"},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert out == msgs

    def test_mixed_orphaned_result_and_orphaned_call(self):
        """One orphaned result is removed AND one orphaned call gets a stub."""
        msgs = [
            # call c4 has a result — clean pair
            {"role": "assistant", "tool_calls": [
                assistant_dict_call("c4"),
                assistant_dict_call("c5"),   # c5 has NO result → stub needed
            ]},
            tool_result("c4"),
            tool_result("c_DANGLING"),       # no call → must be removed
        ]
        out = AIAgent._sanitize_api_messages(msgs)

        ids = [m.get("tool_call_id") for m in out if m.get("role") == "tool"]
        assert "c_DANGLING" not in ids,  "orphaned result survived"
        assert "c4" in ids,              "clean result was removed"
        assert "c5" in ids,              "stub for orphaned call missing"

    def test_empty_list_is_safe(self):
        assert AIAgent._sanitize_api_messages([]) == []

    def test_no_tool_messages_at_all(self):
        """Pure text conversation — nothing to repair."""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert out == msgs

    def test_sdk_object_tool_calls_supported(self):
        """tool_calls stored as SDK-style objects (with .id attribute) are handled."""
        tc_obj = types.SimpleNamespace(id="c6", function=types.SimpleNamespace(
            name="terminal", arguments="{}"
        ))
        msgs = [
            {"role": "assistant", "tool_calls": [tc_obj]},
            # no result → stub expected
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert len(out) == 2
        assert out[1]["tool_call_id"] == "c6"


# ---------------------------------------------------------------------------
# Phase 2a — _cap_delegate_task_calls
# ---------------------------------------------------------------------------

class TestCapDelegateTaskCalls:
    """AIAgent._cap_delegate_task_calls() truncates excess delegate_task calls."""

    def test_excess_delegates_truncated_to_limit(self):
        """More than MAX_CONCURRENT_CHILDREN delegate_task calls → truncated."""
        tcs = [make_tc("delegate_task") for _ in range(MAX_CONCURRENT_CHILDREN + 2)]
        out = AIAgent._cap_delegate_task_calls(tcs)
        delegate_count = sum(1 for tc in out if tc.function.name == "delegate_task")
        assert delegate_count == MAX_CONCURRENT_CHILDREN

    def test_non_delegate_calls_preserved_after_truncation(self):
        """Non-delegate tool calls survive the cap even when delegates are trimmed."""
        tcs = (
            [make_tc("delegate_task") for _ in range(MAX_CONCURRENT_CHILDREN + 1)]
            + [make_tc("terminal"), make_tc("web_search")]
        )
        out = AIAgent._cap_delegate_task_calls(tcs)
        names = [tc.function.name for tc in out]
        assert "terminal" in names
        assert "web_search" in names

    def test_exactly_at_limit_passes_through(self):
        """Exactly MAX_CONCURRENT_CHILDREN delegate_task calls — no truncation."""
        tcs = [make_tc("delegate_task") for _ in range(MAX_CONCURRENT_CHILDREN)]
        out = AIAgent._cap_delegate_task_calls(tcs)
        assert out is tcs   # same object — not rebuilt
        assert len(out) == MAX_CONCURRENT_CHILDREN

    def test_below_limit_passes_through(self):
        """Fewer than MAX_CONCURRENT_CHILDREN — list returned unchanged."""
        tcs = [make_tc("delegate_task") for _ in range(MAX_CONCURRENT_CHILDREN - 1)]
        out = AIAgent._cap_delegate_task_calls(tcs)
        assert out is tcs

    def test_no_delegate_calls_unchanged(self):
        """No delegate_task calls at all — list returned unchanged."""
        tcs = [make_tc("terminal"), make_tc("web_search")]
        out = AIAgent._cap_delegate_task_calls(tcs)
        assert out is tcs

    def test_empty_list_safe(self):
        out = AIAgent._cap_delegate_task_calls([])
        assert out == []

    def test_original_list_not_mutated(self):
        """The input list must not be modified in place."""
        tcs = [make_tc("delegate_task") for _ in range(MAX_CONCURRENT_CHILDREN + 2)]
        original_len = len(tcs)
        AIAgent._cap_delegate_task_calls(tcs)
        assert len(tcs) == original_len

    def test_interleaved_order_preserved(self):
        """Original ordering of delegate and non-delegate calls is preserved."""
        # Build MAX_CONCURRENT_CHILDREN + 1 delegates interleaved with
        # two non-delegate calls after the first and second delegates.
        delegates = [make_tc("delegate_task", f'{{"task":"{i}"}}')
                     for i in range(MAX_CONCURRENT_CHILDREN + 1)]
        t1 = make_tc("terminal", '{"cmd":"ls"}')
        w1 = make_tc("web_search", '{"q":"x"}')
        # Interleave: d0, terminal, d1, web_search, d2, ..., d_excess
        tcs = [delegates[0], t1, delegates[1], w1] + delegates[2:]
        out = AIAgent._cap_delegate_task_calls(tcs)
        # The last delegate should be dropped; everything else stays in order.
        expected = [delegates[0], t1, delegates[1], w1] + delegates[2:MAX_CONCURRENT_CHILDREN]
        assert len(out) == len(expected)
        for i, (actual, exp) in enumerate(zip(out, expected)):
            assert actual is exp, f"mismatch at index {i}"


# ---------------------------------------------------------------------------
# Phase 2b — _deduplicate_tool_calls
# ---------------------------------------------------------------------------

class TestDeduplicateToolCalls:
    """AIAgent._deduplicate_tool_calls() removes identical (name, args) pairs."""

    def test_duplicate_pair_deduplicated(self):
        """Two identical (tool_name, arguments) entries → only first survives."""
        tcs = [
            make_tc("web_search", '{"query":"foo"}'),
            make_tc("web_search", '{"query":"foo"}'),   # duplicate
        ]
        out = AIAgent._deduplicate_tool_calls(tcs)
        assert len(out) == 1
        assert out[0].function.name == "web_search"

    def test_multiple_duplicates_across_tools(self):
        """Duplicates in multiple tool types are all removed."""
        tcs = [
            make_tc("web_search", '{"q":"a"}'),
            make_tc("web_search", '{"q":"a"}'),   # dup
            make_tc("terminal",   '{"cmd":"ls"}'),
            make_tc("terminal",   '{"cmd":"ls"}'),  # dup
            make_tc("terminal",   '{"cmd":"pwd"}'), # different args — keep
        ]
        out = AIAgent._deduplicate_tool_calls(tcs)
        assert len(out) == 3

    def test_same_tool_different_args_not_deduplicated(self):
        """Same tool name but different arguments — both are kept."""
        tcs = [
            make_tc("terminal", '{"cmd":"ls"}'),
            make_tc("terminal", '{"cmd":"pwd"}'),
        ]
        out = AIAgent._deduplicate_tool_calls(tcs)
        assert out is tcs   # unchanged — same object returned

    def test_different_tools_same_args_not_deduplicated(self):
        """Same arguments string on different tools — both are kept."""
        tcs = [
            make_tc("tool_a", '{"x":1}'),
            make_tc("tool_b", '{"x":1}'),
        ]
        out = AIAgent._deduplicate_tool_calls(tcs)
        assert out is tcs

    def test_clean_list_returned_unchanged(self):
        """No duplicates → original list object returned."""
        tcs = [
            make_tc("web_search", '{"q":"x"}'),
            make_tc("terminal",   '{"cmd":"ls"}'),
        ]
        out = AIAgent._deduplicate_tool_calls(tcs)
        assert out is tcs

    def test_single_call_unchanged(self):
        tcs = [make_tc("terminal")]
        out = AIAgent._deduplicate_tool_calls(tcs)
        assert out is tcs

    def test_empty_list_safe(self):
        out = AIAgent._deduplicate_tool_calls([])
        assert out == []

    def test_first_occurrence_kept_not_last(self):
        """When deduplicating, the first instance is preserved."""
        tc1 = make_tc("terminal", '{"cmd":"ls"}')
        tc2 = make_tc("terminal", '{"cmd":"ls"}')
        out = AIAgent._deduplicate_tool_calls([tc1, tc2])
        assert len(out) == 1
        assert out[0] is tc1

    def test_original_list_not_mutated(self):
        """The input list must not be modified in place."""
        tcs = [
            make_tc("web_search", '{"q":"dup"}'),
            make_tc("web_search", '{"q":"dup"}'),
        ]
        original_len = len(tcs)
        AIAgent._deduplicate_tool_calls(tcs)
        assert len(tcs) == original_len

    def test_json_key_order_treated_as_distinct(self):
        """Different JSON key ordering produces different argument strings.

        Deduplication uses raw string comparison, so semantically equivalent
        JSON with different key order is intentionally treated as distinct.
        This test documents that behaviour explicitly.
        """
        tcs = [
            make_tc("web_search", '{"query":"foo","lang":"en"}'),
            make_tc("web_search", '{"lang":"en","query":"foo"}'),
        ]
        out = AIAgent._deduplicate_tool_calls(tcs)
        assert len(out) == 2
        assert out is tcs  # no dedup happened — same object returned


# ---------------------------------------------------------------------------
# _get_tool_call_id_static
# ---------------------------------------------------------------------------

class TestGetToolCallIdStatic:
    """AIAgent._get_tool_call_id_static() extracts call IDs from dicts and objects."""

    def test_dict_with_valid_id(self):
        assert AIAgent._get_tool_call_id_static({"id": "call_123"}) == "call_123"

    def test_dict_with_none_id(self):
        assert AIAgent._get_tool_call_id_static({"id": None}) == ""

    def test_dict_without_id_key(self):
        assert AIAgent._get_tool_call_id_static({"function": {}}) == ""

    def test_object_with_valid_id(self):
        tc = types.SimpleNamespace(id="call_456")
        assert AIAgent._get_tool_call_id_static(tc) == "call_456"

    def test_object_with_none_id(self):
        tc = types.SimpleNamespace(id=None)
        assert AIAgent._get_tool_call_id_static(tc) == ""

    def test_object_without_id_attr(self):
        tc = types.SimpleNamespace()
        assert AIAgent._get_tool_call_id_static(tc) == ""
