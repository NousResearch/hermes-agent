"""Privacy regressions for compression derived from hidden contextual rows."""

from unittest.mock import patch

from agent.context_compressor import (
    COMPRESSED_SUMMARY_HAS_USER_TURN_KEY,
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    ContextCompressor,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
from hermes_state import SessionDB


_HIDDEN_DERIVATION_METADATA = {
    "kind": "compression_summary",
    "contains_hidden": True,
}


def _compressor(*, protect_first_n: int = 0) -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        instance = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=protect_first_n,
            protect_last_n=1,
            quiet_mode=True,
        )
    instance.tail_token_budget = 1
    return instance


def _summary_rows(messages):
    return [
        message
        for message in messages
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]


def _persist_compaction(tmp_path, session_id, original, compressed):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="test", model="test/model")
    db.append_messages_batch(session_id, original)
    db.archive_and_compact(session_id, compressed)
    return db


def test_standalone_summary_derived_from_hidden_row_is_hidden_after_persistence(
    tmp_path,
):
    canary = "HIDDEN-CONTEXT-COMPRESSION-CANARY"
    messages = [
        {"role": "system", "content": "system prompt"},
        {
            "role": "user",
            "content": canary + (" x" * 300),
            "display_kind": "hidden",
            "display_metadata": {"execution_id": "exec-hidden"},
        },
        {"role": "assistant", "content": "historical answer 1 " + ("x" * 400)},
        {"role": "user", "content": "historical request 2 " + ("x" * 400)},
        {"role": "assistant", "content": "historical answer 2 " + ("x" * 400)},
        {"role": "user", "content": "historical request 3 " + ("x" * 400)},
        {"role": "assistant", "content": "historical answer 3 " + ("x" * 400)},
        {"role": "user", "content": "historical request 4 " + ("x" * 400)},
        {"role": "assistant", "content": "visible protected tail " + ("x" * 400)},
    ]
    compressor = _compressor()

    with patch.object(
        compressor,
        "_generate_summary",
        return_value=f"{SUMMARY_PREFIX}\nsummary derived from {canary}",
    ):
        compressed = compressor.compress(
            messages,
            current_tokens=900_000,
            force=True,
        )

    summaries = _summary_rows(compressed)
    assert len(summaries) == 1
    assert summaries[0]["display_kind"] == "hidden"
    assert summaries[0]["display_metadata"] == _HIDDEN_DERIVATION_METADATA

    db = _persist_compaction(
        tmp_path,
        "hidden-standalone-summary",
        messages,
        compressed,
    )
    try:
        public = db.get_messages("hidden-standalone-summary")
        privileged = db.get_messages(
            "hidden-standalone-summary",
            include_hidden=True,
        )
    finally:
        db.close()

    assert canary not in repr(public)
    persisted_summary = next(
        message for message in privileged if canary in str(message.get("content"))
    )
    assert persisted_summary["display_kind"] == "hidden"
    assert persisted_summary["display_metadata"] == _HIDDEN_DERIVATION_METADATA


def test_merged_summary_derived_from_hidden_row_fails_closed_as_hidden():
    canary = "HIDDEN-MERGED-COMPRESSION-CANARY"
    visible_tail = "visible request retained as the merge carrier"
    messages = [
        {
            "role": "user",
            "content": canary + (" x" * 300),
            "display_kind": "hidden",
            "display_metadata": {"execution_id": "exec-hidden-merge"},
        },
        {"role": "assistant", "content": "historical answer 1 " + ("x" * 400)},
        {"role": "user", "content": "historical request 2 " + ("x" * 400)},
        {"role": "assistant", "content": "historical answer 2 " + ("x" * 400)},
        {"role": "user", "content": "historical request 3 " + ("x" * 400)},
        {"role": "assistant", "content": "historical answer 3 " + ("x" * 400)},
        {"role": "user", "content": "historical request 4 " + ("x" * 400)},
        {"role": "assistant", "content": "historical answer 4 " + ("x" * 400)},
        {"role": "user", "content": visible_tail + (" x" * 300)},
    ]
    compressor = _compressor(protect_first_n=0)

    with patch.object(
        compressor,
        "_generate_summary",
        return_value=f"{SUMMARY_PREFIX}\nsummary derived from {canary}",
    ):
        compressed = compressor.compress(
            messages,
            current_tokens=900_000,
            force=True,
        )

    summaries = _summary_rows(compressed)
    assert len(summaries) == 1
    merged = summaries[0]
    merged_content = str(merged["content"])
    assert (
        _MERGED_SUMMARY_DELIMITER in merged_content
        or (
            _SUMMARY_END_MARKER in merged_content
            and not merged_content.rstrip().endswith(_SUMMARY_END_MARKER)
        )
    )
    assert canary in merged_content
    assert merged["display_kind"] == "hidden"
    assert merged["display_metadata"] == _HIDDEN_DERIVATION_METADATA


def test_hidden_summary_taint_survives_sqlite_round_trip_and_recompression(tmp_path):
    first_canary = "HIDDEN-ROLLING-SUMMARY-CANARY"
    first_messages = [
        {"role": "system", "content": "system prompt"},
        {
            "role": "user",
            "content": first_canary + (" x" * 300),
            "display_kind": "hidden",
        },
        {"role": "assistant", "content": "historical answer 1 " + ("x" * 400)},
        {"role": "user", "content": "historical request 2 " + ("x" * 400)},
        {"role": "assistant", "content": "historical answer 2 " + ("x" * 400)},
        {"role": "user", "content": "historical request 3 " + ("x" * 400)},
        {"role": "assistant", "content": "historical answer 3 " + ("x" * 400)},
        {"role": "user", "content": "historical request 4 " + ("x" * 400)},
        {"role": "assistant", "content": "protected tail " + ("x" * 400)},
    ]
    first_compressor = _compressor()
    with patch.object(
        first_compressor,
        "_generate_summary",
        return_value=f"{SUMMARY_PREFIX}\nfirst summary with {first_canary}",
    ):
        first = first_compressor.compress(
            first_messages,
            current_tokens=900_000,
            force=True,
        )

    db = _persist_compaction(
        tmp_path,
        "hidden-summary-restart",
        first_messages,
        first,
    )
    try:
        resumed_messages = db.get_messages_as_conversation(
            "hidden-summary-restart",
            include_hidden=True,
        )
    finally:
        db.close()

    persisted = next(
        message
        for message in resumed_messages
        if first_canary in str(message.get("content"))
    )
    assert COMPRESSED_SUMMARY_METADATA_KEY not in persisted
    assert COMPRESSED_SUMMARY_HAS_USER_TURN_KEY not in persisted
    assert persisted["display_kind"] == "hidden"
    assert persisted["display_metadata"] == _HIDDEN_DERIVATION_METADATA

    resumed = _compressor()
    second_input = [
        *resumed_messages,
        {"role": "user", "content": "new visible request 1 " + ("x" * 400)},
        {"role": "assistant", "content": "new visible work 1 " + ("x" * 400)},
        {"role": "user", "content": "new visible request 2 " + ("x" * 400)},
        {"role": "assistant", "content": "new visible work 2 " + ("x" * 400)},
        {"role": "user", "content": "new visible request 3 " + ("x" * 400)},
        {"role": "assistant", "content": "new protected tail " + ("x" * 400)},
    ]
    with patch.object(
        resumed,
        "_generate_summary",
        return_value=f"{SUMMARY_PREFIX}\nfresh rolling summary",
    ):
        second = resumed.compress(
            second_input,
            current_tokens=900_000,
            force=True,
        )

    second_summaries = _summary_rows(second)
    assert len(second_summaries) == 1
    assert second_summaries[0]["display_kind"] == "hidden"
    assert second_summaries[0]["display_metadata"] == _HIDDEN_DERIVATION_METADATA
