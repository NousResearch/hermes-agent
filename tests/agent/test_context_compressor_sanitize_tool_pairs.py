"""``_sanitize_tool_pairs``: orphan stripping, in-flight tool chains, and id whitespace.

Moved byte-for-byte out of ``tests/agent/test_context_compressor.py``
(Part of #79949, #78647).  No test body, name, assertion, decorator or
parametrization was touched.  The shared ``compressor`` fixture stays in the
original module -- other classes there use it -- and is imported here instead
of being copied.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor

# Shared fixture, owned by tests/agent/test_context_compressor.py (used by
# classes that stayed behind too).  Imported, not duplicated.
from tests.agent.test_context_compressor import compressor


class TestSanitizerStripsOrphanedToolCalls:
    """PR #51218 (salvaged from #51225): orphaned tool_calls are stripped from
    assistant messages instead of having stub tool results inserted, avoiding
    the call_id != id mismatch that let downstream repair_message_sequence drop
    the stubs and re-expose orphans."""

    def test_sanitizer_strips_orphaned_tool_calls(self, compressor):
        """Orphaned tool_calls (no matching tool result) are stripped from
        assistant messages instead of having stubs inserted.  #51218"""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_orphan", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "user", "content": "never mind"},
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        # Orphaned tool_call should be stripped, not stub-inserted
        asst = next(m for m in sanitized if m.get("role") == "assistant")
        assert not asst.get("tool_calls"), "orphaned tool_calls should be stripped"
        # No stub tool messages should be added
        assert not any(m.get("role") == "tool" for m in sanitized)
        # Empty assistant should get placeholder content
        assert asst.get("content") == "(tool call removed)"

    def test_sanitizer_strips_orphaned_keeps_valid(self, compressor):
        """When a MID-LIST assistant has both valid and orphaned tool_calls,
        only the orphans are stripped.  #51218

        The shape must sit mid-list: the same shape at the TAIL is
        indistinguishable from a partial multi-call batch whose remaining
        results are still in flight, and the sanitizer now presumes in-flight
        there (#79278) — preserving is safe because the pre-API chokepoint
        injects stub results for genuinely unanswered calls, while stripping
        a live call silently loses its late result.
        """
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_valid", "function": {"name": "read_file", "arguments": "{}"}},
                    {"id": "tc_orphan", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_valid", "content": "file content"},
            # Later turn: the chain above is settled history, not in flight.
            {"role": "assistant", "content": "done"},
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        asst = next(m for m in sanitized if m.get("role") == "assistant")
        assert len(asst["tool_calls"]) == 1
        assert asst["tool_calls"][0]["id"] == "tc_valid"
        # Valid tool result preserved
        tool_msgs = [m for m in sanitized if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "tc_valid"

    def test_sanitizer_strips_orphaned_preserves_text_content(self, compressor):
        """When an assistant has text content AND orphaned tool_calls,
        the text is preserved and only tool_calls are stripped.  #51218"""
        msgs = [
            {
                "role": "assistant",
                "content": "Let me search for that.",
                "tool_calls": [
                    {"id": "tc_orphan", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "user", "content": "thanks"},
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        asst = next(m for m in sanitized if m.get("role") == "assistant")
        assert asst["content"] == "Let me search for that."
        assert not asst.get("tool_calls")
        # The placeholder must NOT overwrite existing text content.
        assert asst["content"] != "(tool call removed)"

    def test_sanitizer_strips_orphaned_with_call_id_mismatch(self, compressor):
        """Stubs with call_id != id used to be dropped by downstream
        repair_message_sequence, re-exposing orphans.  Stripping avoids
        this entirely.  #51218"""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "fc_abc",
                        "call_id": "call_abc",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                ],
            },
            # No tool result for call_abc — orphaned
            {"role": "user", "content": "next"},
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        asst = next(m for m in sanitized if m.get("role") == "assistant")
        assert not asst.get("tool_calls")
        # No stub tool messages (which would have call_id != id mismatch)

    def test_sanitizer_keeps_valid_pair_matching_on_id_not_call_id(self, compressor):
        """A genuinely matching Codex-format pair must survive when the
        result's tool_call_id matches ``id`` rather than ``call_id`` (#58168
        class). ``_get_tool_call_id``'s ``call_id || id`` precedence picks
        ``call_id`` first, so building the known-id set from a single value
        per tool_call misclassified this valid pair as orphaned on BOTH
        sides — dropping the result AND stripping the tool_call, even though
        neither was actually orphaned."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "fc_777",
                        "call_id": "call_777",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "fc_777", "content": "result"},
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        asst = next(m for m in sanitized if m.get("role") == "assistant")
        assert asst.get("tool_calls"), "valid tool_call must not be stripped"
        assert asst["tool_calls"][0]["id"] == "fc_777"
        tool_msgs = [m for m in sanitized if m.get("role") == "tool"]
        assert len(tool_msgs) == 1, "valid tool result must not be dropped as orphaned"
        assert tool_msgs[0]["tool_call_id"] == "fc_777"

    def test_sanitizer_still_drops_genuine_orphan_with_dual_ids(self, compressor):
        """Negative control: registering both id and call_id must not
        over-relax orphan detection. A genuinely orphaned tool_call (no
        result matching either id variant) is still stripped, while a valid
        dual-id pair in the same window survives. A trailing user turn keeps
        the assistant message out of the in-flight protection window (#79278),
        which intentionally preserves a still-pending trailing call."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "fc_1",
                        "call_id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                    {
                        "id": "fc_2",
                        "call_id": "call_2",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "next question"},
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        asst = next(m for m in sanitized if m.get("role") == "assistant")
        surviving_ids = {tc["id"] for tc in asst.get("tool_calls") or []}
        assert surviving_ids == {"fc_1"}


class TestSanitizerPreservesInFlightToolChain:
    """Issue #79278: an in-flight tool call chain must survive compression.

    When compression fires mid-chain — after the model emitted
    ``assistant(tool_calls)`` but before tool_executor.py appended the matching
    ``role="tool"`` result — the final message is a *pending* tool call, not an
    orphan.  Stripping it (or replacing content with ``(tool call removed)``)
    destroys the chain: when the executor later appends the real result,
    repair_message_sequence drops it as an unmatched orphan and the completed
    side effect and final synthesis are lost.  _sanitize_tool_pairs must exempt
    the trailing in-flight call.
    """

    def test_trailing_inflight_tool_call_preserved(self, compressor):
        """A trailing assistant tool_call with no result yet is pending, not
        orphaned — preserve it verbatim.  #79278"""
        msgs = [
            {"role": "user", "content": "summarize"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "summarize", "arguments": "{}"}},
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_2", "function": {"name": "finalize", "arguments": "{}"}},
                ],
            },
            # in-flight: call_2 has been emitted but its tool result is not
            # yet appended by tool_executor.py
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        # The trailing in-flight call must survive intact.
        assert sanitized[-1]["role"] == "assistant"
        assert len(sanitized[-1]["tool_calls"]) == 1
        assert sanitized[-1]["tool_calls"][0]["id"] == "call_2"
        assert sanitized[-1]["content"] != "(tool call removed)"

    def test_inflight_result_arrives_after_compress(self, compressor):
        """After compress() preserves the pending call, appending its tool
        result leaves a well-formed chain — the side effect's result reaches
        the model.  #79278"""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_2", "function": {"name": "finalize", "arguments": "{}"}},
                ],
            },
        ]
        sanitized = compressor._sanitize_tool_pairs(msgs)
        assert sanitized[-1]["tool_calls"][0]["id"] == "call_2"

        # The executor appends the result after compress() returns.
        sanitized.append(
            {"role": "tool", "tool_call_id": "call_2", "content": "side effect done"}
        )

        # Chain now settled: the assistant call is matched, result survives.
        surviving = {tc["id"] for m in sanitized if m["role"] == "assistant"
                     for tc in (m.get("tool_calls") or [])}
        assert "call_2" in surviving
        results = [m for m in sanitized if m["role"] == "tool"]
        assert len(results) == 1 and results[0]["tool_call_id"] == "call_2"

    def test_inflight_preserved_while_true_orphan_still_stripped(self, compressor):
        """Preserving the trailing in-flight call must not weaken the existing
        orphan-stripping behavior for genuinely orphaned calls in the discarded
        region.  #79278"""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_orphan", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "user", "content": "interim"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_pending", "function": {"name": "write", "arguments": "{}"}},
                ],
            },
            # trailing in-flight call_pending
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        middle = [m for m in sanitized if m["role"] == "assistant" and m.get("tool_calls")]
        # Only the trailing in-flight call survives.
        assert len(middle) == 1
        assert middle[0]["tool_calls"][0]["id"] == "call_pending"
        # The genuine orphan at the head was still stripped.
        assert not any(m.get("tool_calls") and m["tool_calls"][0]["id"] == "call_orphan"
                       for m in sanitized)

    def test_side_effect_final_result_returned_end_to_end(self, compressor):
        """Issue #79278 end-to-end: an in-flight tool chain that triggers
        compression must still deliver the completed side effect's result.

        Reproduces the full executor flow:
          1. the model emits ``assistant(tool_calls=call_2)`` while call_2 is
             still IN-FLIGHT — tool_executor.py has not yet appended its
             ``role="tool"`` result;
          2. context compression fires and ``_sanitize_tool_pairs()`` runs on
             a history whose tail is that pending call;
          3. the side effect completes and the executor appends the ``tool``
             result for call_2;
          4. the next pre-call pass, ``repair_message_sequence()``, runs and
             must NOT drop the completed result as an unmatched orphan.

        On the old code, compression stripped the pending call_2 as an
        "orphan"; when the result then arrived, repair_message_sequence
        dropped it as unmatched — so the completed side effect's result and
        the final synthesis built on it were both lost.  The fix preserves the
        trailing in-flight call, so the chain closes and the result survives.
        """
        from agent.agent_runtime_helpers import repair_message_sequence

        history = [
            {"role": "user", "content": "Do the work and use the tools."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "run_side_effect", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": "side effect completed"},
            # in-flight: call_2 emitted, its result not yet appended
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_2", "type": "function",
                 "function": {"name": "gather_final_result", "arguments": "{}"}}]},
        ]

        # Step 1: compression fires mid-chain.
        compressed = compressor._sanitize_tool_pairs([dict(m) for m in history])

        # The in-flight call must survive the compression pass (fix), not be
        # stripped as an orphan (bug) — otherwise the chain is already broken.
        assert compressed[-1]["role"] == "assistant"
        assert [tc["id"] for tc in compressed[-1].get("tool_calls", [])] == ["call_2"]
        assert compressed[-1].get("content") != "(tool call removed)"

        # Step 2: the side effect completes; the executor appends its result.
        messages = [dict(m) for m in compressed] + [
            {"role": "tool", "tool_call_id": "call_2",
             "content": "final computed result: 42"}
        ]

        # Step 3: the next pre-call sanitizer runs on the settled chain.
        repair_message_sequence(None, messages)

        # Step 4: the final result must still be returned — the assistant call
        # and its completed tool result both survive, still paired.
        assistant_ids = {
            tc["id"]
            for m in messages if m["role"] == "assistant"
            for tc in (m.get("tool_calls") or [])
        }
        assert "call_2" in assistant_ids

        results = [m for m in messages
                   if m["role"] == "tool" and m.get("tool_call_id") == "call_2"]
        assert len(results) == 1
        assert "42" in results[0]["content"]

    def test_partial_batch_inflight_calls_preserved(self, compressor):
        """Multi-call batch snapshotted BETWEEN result appends: the executor
        has appended tool(c1) but not yet tool(c2)/tool(c3), so the last
        message is a tool result while c2/c3 are still pending.  The walk-back
        must find the assistant behind the trailing results and preserve the
        whole batch — stripping c2/c3 there loses their late results exactly
        like the tail-is-assistant shape.  #79278 follow-up."""
        msgs = [
            {"role": "user", "content": "run the batch"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "a", "arguments": "{}"}},
                {"id": "c2", "function": {"name": "b", "arguments": "{}"}},
                {"id": "c3", "function": {"name": "c", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "done 1"},
            # snapshot taken here: c2/c3 results not yet appended
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        batch = [m for m in sanitized if m.get("role") == "assistant"][-1]
        assert [tc["id"] for tc in batch["tool_calls"]] == ["c1", "c2", "c3"]
        # And the already-arrived result survives too.
        assert any(
            m.get("role") == "tool" and m.get("tool_call_id") == "c1"
            for m in sanitized
        )


class TestSanitizeToolPairsWhitespace:
    """_sanitize_tool_pairs must strip whitespace from tool_call_id before
    comparing, matching the fix applied to agent_runtime_helpers.py in
    commit fa3ab2ffd.  Without stripping, a valid tool result whose
    tool_call_id has surrounding whitespace is misclassified as orphaned
    and silently replaced with a [Result unavailable] stub.
    """

    def _make(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            return ContextCompressor(model="test/model", quiet_mode=True,
                                     protect_first_n=2, protect_last_n=2)

    def _assistant(self, call_id):
        return {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}],
        }

    def test_leading_whitespace_on_result_id_preserved(self):
        c = self._make()
        msgs = [
            self._assistant("call_abc"),
            {"role": "tool", "tool_call_id": " call_abc", "content": "ok"},
        ]
        out = c._sanitize_tool_pairs(msgs)
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "ok", "valid result must not be treated as orphaned"

    def test_trailing_whitespace_on_result_id_preserved(self):
        c = self._make()
        msgs = [
            self._assistant("call_xyz"),
            {"role": "tool", "tool_call_id": "call_xyz  ", "content": "data"},
        ]
        out = c._sanitize_tool_pairs(msgs)
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "data"

    def test_truly_orphaned_still_removed(self):
        """Whitespace-trimmed ID that still has no match must be removed.
        The assistant's call_real has no matching result, so a stub is
        inserted in its place — the original orphaned entry must be gone."""
        c = self._make()
        msgs = [
            self._assistant("call_real"),
            {"role": "tool", "tool_call_id": " call_orphan ", "content": "stale"},
        ]
        out = c._sanitize_tool_pairs(msgs)
        tool_call_ids = [m.get("tool_call_id") for m in out if m.get("role") == "tool"]
        assert "call_orphan" not in tool_call_ids, "genuinely orphaned result must be removed"
        assert " call_orphan " not in tool_call_ids, "original whitespace form must also be gone"
