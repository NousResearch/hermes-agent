"""Regression coverage for #29824 — the WebUI session viewer (and TUI
chat panel) was showing the ``[CONTEXT COMPACTION — REFERENCE ONLY]``
handoff block in the slot where the user had just been reading the
assistant's actual reply, because the previously-visible reply got
rolled into the compaction summary by the token-budget tail walk.

The fix adds ``_ensure_last_assistant_message_in_tail`` — a mirror of
the existing ``_ensure_last_user_message_in_tail`` (#10896 anchor) —
that pulls ``cut_idx`` back to include the most recent assistant
message with non-empty text content, with the standard tool-group
realignment so we don't orphan a ``tool_call`` / ``tool_result`` pair.

Pinned here:

* ``TestFindLastAssistantMessageIdx`` — pure helper contract:
  finds the most recent **content-bearing** assistant message,
  skips tool-call-only stubs, falls back to "any assistant" only
  when no content-bearing reply exists in the compressible region,
  honours ``head_end``, returns -1 when there's no assistant at all.

* ``TestEnsureLastAssistantMessageInTail`` — direct: walks
  ``cut_idx`` back when the last reply is in the compressed middle,
  is a no-op when it's already in the tail, never crosses
  ``head_end``, re-aligns through tool groups.

* ``TestFindTailCutByTokensAnchorsAssistant`` — integration with
  the existing tail-cut path: the exact reporter scenario (long
  tool-output run after the previously-visible reply) preserves
  the reply; combines with the user anchor for the same-turn
  preservation; soft-ceiling overrun no longer hides the reply.

* ``TestCompactionRollupReproduction`` — end-to-end through
  ``compress()`` with a stubbed summariser: pre-fix the reply
  text is absorbed into the summary (regression demonstrated by
  asserting on the OLD behaviour fails); post-fix the reply text
  is still present in the compressed transcript as a regular
  assistant message.

* ``TestSourceGuardrail`` — static asserts on
  ``agent/context_compressor.py`` so a future refactor can't
  silently drop the anchor.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture()
def compressor():
    """ContextCompressor with mocked deps and a tight tail budget so
    the helpers' anchor behaviour is observable."""
    from agent.context_compressor import ContextCompressor
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
        c.tail_token_budget = 50
        return c


# ---------------------------------------------------------------------------
# Helper: _find_last_assistant_message_idx
# ---------------------------------------------------------------------------


class TestFindLastAssistantMessageIdx:
    def test_finds_content_bearing_assistant(self, compressor):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "the reply"},
        ]
        idx = compressor._find_last_assistant_message_idx(messages, head_end=1)
        assert idx == 2

    def test_skips_tool_call_only_stub_when_text_reply_exists_earlier(
        self, compressor
    ):
        """An assistant message that only carries ``tool_calls`` (no
        text content) is not the user-visible reply — the WebUI
        renders those as small "calling tool X" indicators. The helper
        must prefer the earlier text reply, which is what the user
        actually read."""
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "VISIBLE REPLY"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "t",
                                          "arguments": "{}"}}]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        idx = compressor._find_last_assistant_message_idx(messages, head_end=0)
        assert idx == 1, (
            "Expected the content-bearing assistant reply (1), not the "
            f"trailing tool-call stub. Got {idx}."
        )

    def test_empty_string_content_does_not_count_as_visible(self, compressor):
        """An assistant message with ``content=""`` (only whitespace)
        is not a visible reply either — common pre-flight stub before
        the model streams the real answer."""
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "earlier reply"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "   "},  # blank stub
        ]
        idx = compressor._find_last_assistant_message_idx(messages, head_end=0)
        # Blank-string assistant message does not count — fall back
        # to the earlier real reply.
        assert idx == 1

    def test_multimodal_text_block_counts(self, compressor):
        """An assistant with multimodal list-content carrying a text
        block (Anthropic / GPT-style ``[{type:text,text:...}]``)
        counts as content-bearing."""
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant",
             "content": [{"type": "text", "text": "hello"}]},
        ]
        idx = compressor._find_last_assistant_message_idx(messages, head_end=0)
        assert idx == 1

    def test_fallback_to_any_assistant_when_no_content_bearing(
        self, compressor
    ):
        """When there's no text-bearing assistant in the compressible
        region (fresh multi-step tool sequence), fall back to the
        most recent assistant of any kind so the anchor still works."""
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "t",
                                          "arguments": "{}"}}]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        idx = compressor._find_last_assistant_message_idx(messages, head_end=0)
        assert idx == 1

    def test_returns_negative_one_when_no_assistant(self, compressor):
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "user", "content": "q2"},
        ]
        idx = compressor._find_last_assistant_message_idx(messages, head_end=0)
        assert idx == -1

    def test_respects_head_end_lower_bound(self, compressor):
        """An assistant message at or before ``head_end`` must be
        ignored — it's already in the protected head region."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "in-head reply"},  # idx 1
            {"role": "user", "content": "q"},
        ]
        # head_end=2 means the compressible region starts at index 2;
        # the assistant at index 1 is in the head and must be skipped.
        idx = compressor._find_last_assistant_message_idx(messages, head_end=2)
        assert idx == -1


# ---------------------------------------------------------------------------
# Helper: _ensure_last_assistant_message_in_tail
# ---------------------------------------------------------------------------


class TestEnsureLastAssistantMessageInTail:
    def test_no_op_when_already_in_tail(self, compressor):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "q2"},
        ]
        # cut_idx=1 means tail starts at index 1 — the reply is already in tail.
        new_cut = compressor._ensure_last_assistant_message_in_tail(
            messages, cut_idx=1, head_end=0
        )
        assert new_cut == 1

    def test_walks_cut_idx_back_to_include_reply(self, compressor):
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "REPLY"},  # idx 1
            {"role": "user", "content": "q2"},
            {"role": "user", "content": "q3"},
        ]
        # cut_idx=2 leaves the reply outside the tail; anchor must pull
        # cut_idx back to 1 so messages[1:] contains the reply.
        new_cut = compressor._ensure_last_assistant_message_in_tail(
            messages, cut_idx=2, head_end=0
        )
        assert new_cut == 1
        assert any(
            isinstance(m.get("content"), str) and "REPLY" in m["content"]
            for m in messages[new_cut:]
        )

    def test_never_crosses_head_end(self, compressor):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "in-head"},  # head, must ignore
            {"role": "user", "content": "q"},
        ]
        # head_end=2 ⇒ assistant at idx 1 is in the head; the anchor
        # finds nothing in the compressible region and is a no-op.
        new_cut = compressor._ensure_last_assistant_message_in_tail(
            messages, cut_idx=3, head_end=2
        )
        assert new_cut == 3

    def test_re_aligns_through_preceding_tool_group(self, compressor):
        """When the anchored assistant is preceded by a
        tool_call/result group, ``_align_boundary_backward`` must pull
        ``cut_idx`` even further back so the group isn't split — same
        guarantee as ``_ensure_last_user_message_in_tail``."""
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1",
                             "function": {"name": "t",
                                          "arguments": "{}"}}]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
            {"role": "assistant", "content": "REPLY"},  # idx 3
            {"role": "user", "content": "q2"},
        ]
        # cut_idx=4 leaves the reply outside the tail. Anchor pulls
        # back to 3, then _align_boundary_backward sees the preceding
        # tool group and pulls further back to 1 (before the assistant
        # with tool_calls).
        new_cut = compressor._ensure_last_assistant_message_in_tail(
            messages, cut_idx=4, head_end=0
        )
        assert new_cut <= 3
        # The tool_call assistant (1) and its tool_result (2) must NOT
        # be split: either both in compressed region or both in tail.
        if new_cut <= 1:
            # Both in tail — tool group intact.
            assert messages[new_cut].get("role") == "assistant"
        else:
            # Otherwise the anchor must land at the reply itself (3).
            assert new_cut == 3


# ---------------------------------------------------------------------------
# Integration with _find_tail_cut_by_tokens
# ---------------------------------------------------------------------------


class TestFindTailCutByTokensAnchorsAssistant:
    def test_reporter_repro_long_tool_run_after_visible_reply(
        self, compressor
    ):
        """The exact #29824 scenario: a tight token budget combined
        with a long tail of tool-call/result messages after the
        visible reply. Pre-fix, the token-budget walk hit its ceiling
        on the tool output and parked ``cut_idx`` past the reply.
        Post-fix, the assistant anchor pulls it back."""
        c = compressor
        c.tail_token_budget = 10  # force min-tail behaviour
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg1"},                # head_end=2
            {"role": "user", "content": "q1"},
            {"role": "assistant",
             "content": "PREVIOUSLY VISIBLE REPLY"},           # idx 3
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1",
                             "function": {"name": "t",
                                          "arguments": "{}"}}]},
            {"role": "tool", "content": "x" * 200,
             "tool_call_id": "c1"},
        ]
        cut = c._find_tail_cut_by_tokens(messages, head_end=2)
        tail_contents = [
            m.get("content") for m in messages[cut:]
            if isinstance(m.get("content"), str)
        ]
        assert any(
            "PREVIOUSLY VISIBLE REPLY" in (t or "") for t in tail_contents
        ), (
            "REGRESSION (#29824): the visible reply was rolled into "
            f"the compaction summary. Tail contents: {tail_contents!r}"
        )

    def test_user_and_assistant_anchors_compose(self, compressor):
        """Both anchors run in sequence; the tail must contain both
        the latest user message AND the latest visible assistant
        reply."""
        c = compressor
        c.tail_token_budget = 10
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "VISIBLE REPLY"},
            {"role": "user", "content": "follow-up question"},
            {"role": "user", "content": "and another"},
        ]
        cut = c._find_tail_cut_by_tokens(messages, head_end=0)
        tail_contents = [
            m.get("content") for m in messages[cut:]
            if isinstance(m.get("content"), str)
        ]
        assert any("VISIBLE REPLY" in (t or "") for t in tail_contents)
        assert any("and another" in (t or "") for t in tail_contents)

    def test_oversized_tool_output_does_not_strand_reply(self, compressor):
        """The soft-ceiling logic in ``_find_tail_cut_by_tokens``
        permits a single oversized tail message; the assistant anchor
        must still recover the reply on the other side of it."""
        c = compressor
        c.tail_token_budget = 100  # soft ceiling 150
        messages = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "VISIBLE REPLY"},
            {"role": "user", "content": "read big file"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1",
                             "function": {"name": "read",
                                          "arguments": "{}"}}]},
            # ~500 chars ⇒ ~135 tokens, blows past soft ceiling of 150
            {"role": "tool", "content": "y" * 500,
             "tool_call_id": "c1"},
            {"role": "user", "content": "ok"},
        ]
        cut = c._find_tail_cut_by_tokens(messages, head_end=0)
        tail_contents = [
            m.get("content") for m in messages[cut:]
            if isinstance(m.get("content"), str)
        ]
        assert any("VISIBLE REPLY" in (t or "") for t in tail_contents)


# ---------------------------------------------------------------------------
# End-to-end: compress() preserves the reply
# ---------------------------------------------------------------------------


class TestCompactionRollupReproduction:
    """End-to-end through ``compress()``: the visible reply text must
    survive in the compressed transcript — either as its own
    standalone assistant message OR concatenated onto the merged
    summary-handoff tail message (the compressor's double-collision
    fallback path; the WebUI re-splits these on the END marker so the
    reply renders as a separate bubble — see ``splitCompactionContent``
    in ``web/src/pages/SessionsPage.tsx``)."""

    def test_compress_keeps_visible_reply_text(self, compressor):
        from agent.context_compressor import SUMMARY_PREFIX
        c = compressor
        c.tail_token_budget = 10
        # ``_generate_summary`` normally wraps the LLM body in
        # ``SUMMARY_PREFIX`` via ``_with_summary_prefix``; mimic that so
        # the merge-into-tail branch can identify the boundary.
        _mocked = f"{SUMMARY_PREFIX}\nrolled-up middle summary"
        messages = (
            [{"role": "system", "content": "sys"},
             {"role": "user", "content": "initial"}]   # head (protect_first_n=2)
            # Middle: long enough to be compressible.
            + [
                {"role": "user", "content": f"middle q{i}"}
                if i % 2 == 0
                else {"role": "assistant", "content": f"middle reply {i}"}
                for i in range(12)
            ]
            + [
                {"role": "user", "content": "the visible question"},
                {"role": "assistant",
                 "content": "THE VISIBLE REPLY THE USER JUST READ"},
                {"role": "user", "content": "follow up"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c1",
                                 "function": {"name": "t",
                                              "arguments": "{}"}}]},
                {"role": "tool", "content": "z" * 500,
                 "tool_call_id": "c1"},
            ]
        )
        with patch.object(
            c, "_generate_summary",
            return_value=_mocked,
        ):
            result = c.compress(messages, current_tokens=90_000)
        # 1. A summary message exists (compression actually ran).
        assert any(
            isinstance(m.get("content"), str)
            and m["content"].startswith(SUMMARY_PREFIX)
            for m in result
        ), "compress() did not insert a summary message"
        # 2. The visible reply text must survive somewhere — either
        # as its own message OR concatenated into the merged tail.
        joined = "\n".join(
            m.get("content") for m in result
            if isinstance(m.get("content"), str)
        )
        assert "THE VISIBLE REPLY THE USER JUST READ" in joined, (
            "REGRESSION (#29824): the visible reply was absorbed into "
            "the compaction summary AND erased. Compressed transcript "
            f"({len(result)} msgs): "
            f"{[(m.get('role'), str(m.get('content'))[:50]) for m in result]}"
        )

    def test_standalone_summary_case_keeps_reply_as_own_message(
        self, compressor
    ):
        """When the head and tail roles allow a standalone summary
        message (no double-collision), the visible reply must remain
        as its OWN assistant message — not merged with anything.
        This is the common case; the merge-into-tail path is the
        edge case for double-collision."""
        from agent.context_compressor import SUMMARY_PREFIX
        c = compressor
        c.tail_token_budget = 10
        _mocked = f"{SUMMARY_PREFIX}\nrolled-up middle summary"
        # Head ends with ``assistant`` ⇒ summary_role flips to
        # ``user`` ⇒ no collision with the assistant tail ⇒ standalone
        # summary insert (no merge).
        messages = (
            [
                {"role": "user", "content": "initial"},
                {"role": "assistant", "content": "head reply"},
            ]
            + [
                {"role": "user", "content": f"middle q{i}"}
                if i % 2 == 0
                else {"role": "assistant", "content": f"middle reply {i}"}
                for i in range(12)
            ]
            + [
                {"role": "user", "content": "the visible question"},
                {"role": "assistant",
                 "content": "THE VISIBLE REPLY THE USER JUST READ"},
                {"role": "user", "content": "follow up"},
            ]
        )
        with patch.object(
            c, "_generate_summary",
            return_value=_mocked,
        ):
            result = c.compress(messages, current_tokens=90_000)
        # Standalone summary present:
        summary_rows = [
            m for m in result
            if isinstance(m.get("content"), str)
            and m["content"].startswith(SUMMARY_PREFIX)
        ]
        assert len(summary_rows) == 1
        # Visible reply as its OWN distinct assistant message
        # (NOT merged into the summary row):
        reply_rows = [
            m for m in result
            if m.get("role") == "assistant"
            and isinstance(m.get("content"), str)
            and "THE VISIBLE REPLY THE USER JUST READ" in m["content"]
            and not m["content"].startswith(SUMMARY_PREFIX)
        ]
        assert len(reply_rows) == 1, (
            "REGRESSION (#29824): expected exactly one standalone "
            f"assistant message carrying the visible reply, got "
            f"{len(reply_rows)}"
        )


# ---------------------------------------------------------------------------
# Source guardrail
# ---------------------------------------------------------------------------


class TestFindLastUserMessageIdxSkipsSummaryMarker:
    """A context-compaction handoff banner is inserted with ``role="user"``
    when the head ends in an assistant/tool message (see the summary-role
    selection in ``compress``). ``_find_last_user_message_idx`` must NOT treat
    that banner as the latest user turn — otherwise, on a resumed or
    multi-compaction session, ``_ensure_last_user_message_in_tail`` anchors the
    tail to the summary and rolls the genuine last user message into the next
    compaction, re-triggering the active-task loss the anchor exists to prevent.
    (Salvaged from #36626 / issue #36624.)
    """

    def test_skips_user_role_context_summary_marker(self, compressor):
        from agent.context_compressor import SUMMARY_PREFIX

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "REAL current task"},
            {"role": "assistant", "content": "working on it"},
            # A handoff summary re-inserted as a user-role message after resume.
            {"role": "user", "content": f"{SUMMARY_PREFIX}\n## Active Task\nold"},
            {"role": "assistant", "content": "continuing from the real task"},
        ]
        # Latest *real* user message is index 1, not the summary at index 3.
        assert compressor._find_last_user_message_idx(messages, head_end=1) == 1

    def test_returns_real_user_when_no_summary_present(self, compressor):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert compressor._find_last_user_message_idx(messages, head_end=1) == 3

    def test_all_user_messages_are_summaries_returns_minus_one(self, compressor):
        from agent.context_compressor import SUMMARY_PREFIX

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": f"{SUMMARY_PREFIX}\nhandoff"},
        ]
        assert compressor._find_last_user_message_idx(messages, head_end=1) == -1


class TestSourceGuardrail:
    @pytest.fixture
    def source(self) -> str:
        from pathlib import Path
        return (Path(__file__).resolve().parents[2]
                / "agent" / "context_compressor.py").read_text(
                    encoding="utf-8")

    def test_helper_defined(self, source):
        assert "def _find_last_assistant_message_idx(" in source
        assert "def _ensure_last_assistant_message_in_tail(" in source

    def test_anchor_called_from_find_tail_cut(self, source):
        """Without the call site the helper is dead code and the bug
        regresses silently — pin both the definition AND the wiring."""
        assert "self._ensure_last_assistant_message_in_tail(" in source

    def test_anchor_called_after_user_anchor(self, source):
        """The two anchors must run in sequence; reversing or skipping
        one drops the corresponding side of the guarantee."""
        user_call = "self._ensure_last_user_message_in_tail(messages, cut_idx, head_end)"
        asst_call = "self._ensure_last_assistant_message_in_tail(messages, cut_idx, head_end)"
        user_idx = source.find(user_call)
        asst_idx = source.find(asst_call)
        assert user_idx >= 0 and asst_idx >= 0
        assert asst_idx > user_idx, (
            "The assistant anchor must come AFTER the user anchor in "
            "``_find_tail_cut_by_tokens`` — each anchor walks cut_idx "
            "backward, and ordering keeps the chain monotonic."
        )

    def test_helper_prefers_content_bearing_reply(self, source):
        """The helper must skip tool-call-only stubs — that's the
        whole user-experience difference between #29824 (no visible
        reply) and an in-progress turn (small 'calling tool X' chip)."""
        assert "content.strip()" in source

    def test_issue_number_referenced(self, source):
        assert "#29824" in source
