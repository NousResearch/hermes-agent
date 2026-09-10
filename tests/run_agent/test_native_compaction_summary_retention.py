"""Tests for native compaction summary retention during pre-checkpoint pruning (#90975).

``prune_pre_checkpoint_items`` previously dropped every pre-checkpoint item
whose ``role`` was not ``"user"`` — which silently deleted Hermes' own local
compression summaries (``role="assistant"``) from the wire on every native
compaction turn. These tests cover the fix's summary retention path, its
reliance on the canonical ``agent.context_compressor`` provenance check (not
an ad-hoc heuristic), whole-or-drop truncation, and idempotency.
"""

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
from agent.native_compaction import (
    _extract_item_text,
    _is_summary_item,
    prune_pre_checkpoint_items,
)


def _standalone_summary_content(body: str = "## Active Task\nstuff") -> str:
    return f"{SUMMARY_PREFIX}\n{body}\n\n{_SUMMARY_END_MARKER}"


def _merged_summary_content(tail: str = "preserved prior turn") -> str:
    return (
        f"{_MERGED_PRIOR_CONTEXT_HEADER}\n{tail}\n\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n\n"
        f"{SUMMARY_PREFIX}\nbody\n\n{_SUMMARY_END_MARKER}"
    )


class TestIsSummaryItemCanonical:
    """`_is_summary_item` must delegate to the canonical provenance check —
    exact metadata flag or the canonical prefix classifier — never an
    ad-hoc heuristic (#90975 blocking review)."""

    def test_truthy_metadata_flag_detected(self):
        assert _is_summary_item({COMPRESSED_SUMMARY_METADATA_KEY: True}) is True

    def test_standalone_content_detected_without_metadata(self):
        # The wire sanitizers strip underscore keys, so content-only
        # detection must still work on the canonical prefix.
        assert _is_summary_item({"role": "assistant", "content": _standalone_summary_content()}) is True

    def test_merged_content_detected_without_metadata(self):
        assert _is_summary_item({"role": "assistant", "content": _merged_summary_content()}) is True

    def test_malformed_inputs_are_not_summaries(self):
        assert _is_summary_item(None) is False
        assert _is_summary_item(123) is False
        assert _is_summary_item({}) is False


class TestIsSummaryItemNegativeWitnesses:
    """Content that merely resembles a summary must never be promoted to
    durable retained history — that is authority drift (#90975 blocking
    review, required item 4)."""

    def test_summary_heading_in_ordinary_user_text_is_not_a_summary(self):
        item = {"role": "user", "content": "## Summary\nplease summarize the PR for me"}
        assert _is_summary_item(item) is False

    def test_false_valued_metadata_flag_is_not_a_summary(self):
        item = {"role": "assistant", "content": "hi", COMPRESSED_SUMMARY_METADATA_KEY: False}
        assert _is_summary_item(item) is False

    def test_arbitrary_underscore_summary_key_is_not_a_summary(self):
        item = {"role": "assistant", "content": "hi", "_my_custom_summary_flag": True}
        assert _is_summary_item(item) is False

    def test_non_hermes_assistant_content_is_not_a_summary(self):
        item = {"role": "assistant", "content": "Conversation Summary: I finished the task."}
        assert _is_summary_item(item) is False


class TestExtractItemTextVariations:
    def test_string_content(self):
        assert _extract_item_text({"content": "Hello world"}) == "Hello world"

    def test_multipart_list_content(self):
        item = {
            "content": [
                {"type": "input_text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
                {"type": "other", "output_text": "Part 3"},
            ]
        }
        assert _extract_item_text(item) == "Part 1 Part 2 Part 3"

    def test_output_text_fallback(self):
        assert _extract_item_text({"output_text": "Output fallback"}) == "Output fallback"

    def test_malformed_or_empty(self):
        assert _extract_item_text({"content": None}) is None
        assert _extract_item_text({"content": []}) is None
        assert _extract_item_text(None) is None
        assert _extract_item_text("string_item") is None


class TestPrunePreCheckpointItemsRetainsSummaries:
    def test_retains_summary_and_user_in_original_order(self):
        summary_content = _standalone_summary_content("Step 1 complete")
        items = [
            {"role": "user", "content": "User Ask 1"},
            {"role": "assistant", "content": summary_content, COMPRESSED_SUMMARY_METADATA_KEY: True},
            {"role": "user", "content": "User Ask 2"},
            {"role": "assistant", "content": "Normal chatter to prune"},
            {"type": "compaction", "encrypted_content": "blob_cp"},
            {"role": "user", "content": "User Ask 3"},
        ]

        pruned = prune_pre_checkpoint_items(items, retained_user_token_budget=1000)

        assert pruned[0]["type"] == "compaction"
        contents = [m.get("content") for m in pruned[1:]]
        assert contents == [
            "User Ask 1",
            summary_content,
            "User Ask 2",
            "User Ask 3",
        ]

    def test_role_agnostic_retention_does_not_touch_user_budget(self):
        summary_content = _standalone_summary_content("x" * 2000)
        items = [
            {"role": "assistant", "content": summary_content, COMPRESSED_SUMMARY_METADATA_KEY: True},
            {"role": "user", "content": "short ask"},
            {"type": "compaction", "encrypted_content": "blob_cp"},
        ]

        pruned = prune_pre_checkpoint_items(
            items, retained_user_token_budget=10, retained_summary_token_budget=10_000
        )

        contents = [m.get("content") for m in pruned]
        assert summary_content in contents
        assert "short ask" in contents


class TestPrunePreCheckpointItemsSummaryBudget:
    def test_oversized_summary_is_dropped_whole_not_sliced(self):
        """A summary that cannot fit the remaining budget is dropped
        entirely rather than character-sliced (#90975 blocking review,
        required item 3): slicing can corrupt the handoff prefix / end
        marker that keeps the summary non-active."""
        long_summary = _standalone_summary_content("Summary line " * 500)
        items = [
            {"role": "assistant", "content": long_summary, COMPRESSED_SUMMARY_METADATA_KEY: True},
            {"type": "compaction", "encrypted_content": "blob_cp"},
            {"role": "user", "content": "Ask"},
        ]

        pruned = prune_pre_checkpoint_items(items, retained_summary_token_budget=100)

        assert not any(m.get(COMPRESSED_SUMMARY_METADATA_KEY) for m in pruned)

    def test_summary_that_fits_budget_is_retained_whole(self):
        summary_content = _standalone_summary_content("short body")
        items = [
            {"role": "assistant", "content": summary_content, COMPRESSED_SUMMARY_METADATA_KEY: True},
            {"type": "compaction", "encrypted_content": "blob_cp"},
            {"role": "user", "content": "Ask"},
        ]

        pruned = prune_pre_checkpoint_items(items, retained_summary_token_budget=10_000)

        retained = [m for m in pruned if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
        assert len(retained) == 1
        assert retained[0]["content"] == summary_content


class TestPrunePreCheckpointItemsIdempotency:
    def test_duplicate_summary_text_is_not_retained_twice(self):
        """A repeated checkpoint sequence can leave the same summary text
        present at more than one pre-checkpoint position; retention must
        stay idempotent rather than duplicate it (#90975 blocking review,
        required item 5)."""
        summary_content = _standalone_summary_content("same body")
        items = [
            {"role": "assistant", "content": summary_content, COMPRESSED_SUMMARY_METADATA_KEY: True},
            {"role": "user", "content": "mid ask"},
            {"role": "assistant", "content": summary_content, COMPRESSED_SUMMARY_METADATA_KEY: True},
            {"type": "compaction", "encrypted_content": "blob_cp"},
            {"role": "user", "content": "Ask"},
        ]

        pruned = prune_pre_checkpoint_items(items)

        matches = [m for m in pruned if m.get("content") == summary_content]
        assert len(matches) == 1

    def test_re_pruning_an_already_pruned_result_is_stable(self):
        summary_content = _standalone_summary_content("stable body")
        items = [
            {"role": "assistant", "content": summary_content, COMPRESSED_SUMMARY_METADATA_KEY: True},
            {"role": "user", "content": "ask"},
            {"type": "compaction", "encrypted_content": "blob_cp"},
        ]

        once = prune_pre_checkpoint_items(items)
        twice = prune_pre_checkpoint_items(once)
        assert once == twice


class TestPrunePreCheckpointItemsLiveCompressorEmissions:
    """Exercise the real ``ContextCompressor`` marker renderer instead of a
    hand-built stand-in, for both standalone and merge-into-tail shapes
    (#90975 blocking review, required item 5)."""

    def test_standalone_live_marker_is_retained(self):
        rendered = ContextCompressor._render_micro_marker_content("Live handoff body")
        assert ContextCompressor.classify_summary_content(rendered) == "standalone"

        items = [
            {"role": "assistant", "content": rendered, COMPRESSED_SUMMARY_METADATA_KEY: True},
            {"type": "compaction", "encrypted_content": "blob_cp"},
            {"role": "user", "content": "Ask"},
        ]
        pruned = prune_pre_checkpoint_items(items)
        assert any(m.get("content") == rendered for m in pruned)

    def test_merged_tail_summary_is_retained_and_classified_merged(self):
        merged = _merged_summary_content("earlier preserved turn text")
        assert ContextCompressor.classify_summary_content(merged) == "merged"

        items = [
            {"role": "assistant", "content": merged, COMPRESSED_SUMMARY_METADATA_KEY: True},
            {"type": "compaction", "encrypted_content": "blob_cp"},
            {"role": "user", "content": "Ask"},
        ]
        pruned = prune_pre_checkpoint_items(items)
        assert any(m.get("content") == merged for m in pruned)


class TestPrunePreCheckpointItemsEnableSummaryRetentionToggle:
    def test_disabling_summary_retention_drops_pre_checkpoint_summaries(self):
        summary_content = _standalone_summary_content("Old")
        items = [
            {"role": "assistant", "content": summary_content, COMPRESSED_SUMMARY_METADATA_KEY: True},
            {"type": "compaction", "encrypted_content": "blob"},
            {"role": "user", "content": "New ask"},
        ]

        pruned_disabled = prune_pre_checkpoint_items(items, enable_summary_retention=False)
        contents = [m.get("content") for m in pruned_disabled]
        assert summary_content not in contents


def _checkpoint_message(item_id: str = "rs_cp1", blob: str = "cp_blob_1"):
    """An assistant message carrying a replayable native-compaction checkpoint."""
    return {
        "role": "assistant",
        "content": "",
        "codex_reasoning_items": [
            {"type": "compaction", "encrypted_content": blob, "id": item_id},
        ],
    }


class TestChatMessagesToResponsesInputSummaryCarrierLoss:
    """Adapter-level witnesses for the second blocking review (#90976):
    ``prune_pre_checkpoint_items`` only ever saw whatever ``_is_summary_item``
    could recover from an already-converted Responses ``item`` — but two
    real merge-into-tail carrier shapes lose or shadow the summary content
    during ``_chat_messages_to_responses_input`` itself, *before* pruning
    ever runs:

    * a tool-result carrier becomes a typed ``function_call_output`` (no
      ``content``/``role`` survive the conversion at all), and
    * an assistant carrier with a stale ``codex_message_items`` sidecar
      replays the pre-merge exact message item instead of the rewritten
      (summary-bearing) ``content``.

    These feed real chat messages, shaped exactly the way
    ``ContextCompressor.compress()`` merge-into-tail produces them (same
    ``COMPRESSED_SUMMARY_METADATA_KEY`` stamp, same merge delimiters/end
    marker), through the real ``_chat_messages_to_responses_input`` with a
    replayed checkpoint — not a hand-built Responses item passed straight
    to the pruner.
    """

    def test_tool_result_merge_carrier_summary_survives_the_adapter(self):
        from agent.codex_responses_adapter import _chat_messages_to_responses_input

        merged = _merged_summary_content("preserved tool context")
        messages = [
            {"role": "user", "content": "please do the thing"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "do_thing", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": merged,
                COMPRESSED_SUMMARY_METADATA_KEY: True,
            },
            _checkpoint_message(),
            {"role": "user", "content": "next ask after checkpoint"},
        ]

        items = _chat_messages_to_responses_input(
            messages, native_compaction_eligible=True,
        )

        # The summary survives, exactly once, as a plain message item —
        # never as a `function_call_output` (which the pruner cannot see,
        # and which would orphan the dropped `function_call` it used to
        # pair with).
        assert not any(
            isinstance(it, dict) and it.get("type") == "function_call_output"
            for it in items
        )
        matches = [
            it for it in items
            if isinstance(it, dict) and _extract_item_text(it) == merged
        ]
        assert len(matches) == 1
        assert matches[0].get("type") != "function_call_output"

        # And the newest checkpoint still leads the wire.
        assert items[0].get("type") == "compaction"

    def test_assistant_merge_carrier_with_stale_replay_summary_survives(self):
        from agent.codex_responses_adapter import _chat_messages_to_responses_input

        merged = _merged_summary_content("preserved assistant context")
        messages = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                # Rewritten by the compressor merge — this is what must
                # reach the wire.
                "content": merged,
                COMPRESSED_SUMMARY_METADATA_KEY: True,
                # Stale sidecar captured BEFORE the merge rewrote the
                # content above. The exact-replay path prefers this over
                # `content` for prefix-cache continuity, which is exactly
                # what shadows the summary (#90976).
                "codex_message_items": [{
                    "type": "message",
                    "role": "assistant",
                    "id": "msg_stale_1",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "stale pre-merge answer"}],
                }],
            },
            _checkpoint_message(),
            {"role": "user", "content": "next ask"},
        ]

        items = _chat_messages_to_responses_input(
            messages, native_compaction_eligible=True,
        )

        assert not any(
            isinstance(it, dict) and _extract_item_text(it) == "stale pre-merge answer"
            for it in items
        )
        matches = [
            it for it in items
            if isinstance(it, dict) and _extract_item_text(it) == merged
        ]
        assert len(matches) == 1
        assert items[0].get("type") == "compaction"


class TestPrunePreCheckpointItemsMalformedInputs:
    def test_handles_none_non_dict_and_empty_items_safely(self):
        assert prune_pre_checkpoint_items(None) is None
        assert prune_pre_checkpoint_items([]) == []

        items = [
            None,
            123,
            "raw_string",
            {"role": "user", "content": "Valid user ask"},
            {"type": "compaction", "encrypted_content": "blob"},
        ]
        pruned = prune_pre_checkpoint_items(items)
        assert len(pruned) == 2
        assert pruned[0]["type"] == "compaction"
        assert pruned[1]["content"] == "Valid user ask"
