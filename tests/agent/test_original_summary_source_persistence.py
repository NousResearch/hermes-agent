"""Original-source windows follow the durable conversation after rewind and compaction."""

import pytest

from agent.context_compressor import (
    ContextCompressor,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
from hermes_state import SessionDB


def _compact(db, label, tail, carrier_kind):
    checkpoint = {
        "role": "assistant", "content": f"{SUMMARY_PREFIX}\n{label}",
        "_compressed_summary": True,
    }
    if carrier_kind == "merged":
        checkpoint = {
            **tail[0],
            "content": (
                f"{_MERGED_PRIOR_CONTEXT_HEADER}\n{tail[0]['content']}\n\n"
                f"{_MERGED_SUMMARY_DELIMITER}\n\n{SUMMARY_PREFIX}\n{label}\n\n{_SUMMARY_END_MARKER}"
            ),
            "_compressed_summary": True,
        }
        compacted = [checkpoint, *tail[1:]]
    else:
        compacted = [checkpoint, *tail]
    db.archive_and_compact("s", compacted, tail_count=len(tail))


def _live_raw(db):
    raw = [
        ContextCompressor._strip_context_summary_handoff_message(row)
        for row in db.get_messages_as_conversation("s", include_summary_markers=True)
    ]
    return [row for row in raw if row is not None]


def _source(db, current_window, windows):
    compressor = ContextCompressor(
        model="test", quiet_mode=True, config_context_length=100_000,
        summary_source="original", summary_source_windows=windows,
    )
    compressor.bind_session_state(db, "s")
    result = compressor._load_original_summary_source(current_window)
    assert result is not None
    return result


@pytest.mark.parametrize("carrier_kind", ["standalone", "merged"])
@pytest.mark.parametrize("windows", [1, 2])
def test_original_source_excludes_rewound_rows_and_keeps_checkpoint_boundary(tmp_path, carrier_kind, windows):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("s", source="test")
        db.append_messages_batch("s", [
            {"role": "user", "content": "outside-window-user"},
            {"role": "assistant", "content": "outside-window-answer"},
            {"role": "user", "content": "recent-carried-user"},
        ])
        live = db.get_messages_as_conversation("s", include_row_ids=True)
        _compact(db, "older-checkpoint", live[-1:], carrier_kind)
        db.append_messages_batch("s", [
            {"role": "assistant", "content": "recent-carried-answer"},
            {"role": "user", "content": "recent-user"},
            {"role": "assistant", "content": "recent-answer"},
        ])
        withdrawn_id = db.append_message("s", "user", "withdrawn-user")
        db.append_message("s", "assistant", "withdrawn-answer")
        assert db.rewind_to_message("s", withdrawn_id)["rewound_count"] == 2
        db.append_messages_batch("s", [
            {"role": "user", "content": "replacement-user"},
            {"role": "assistant", "content": "replacement-answer"},
        ])
        live = db.get_messages_as_conversation("s", include_row_ids=True)
        _compact(db, "newer-checkpoint", live[-2:], carrier_kind)
        db.append_messages_batch("s", [
            {"role": "user", "content": "current-user"},
            {"role": "assistant", "content": "current-answer"},
        ])
        current = _live_raw(db)

        turns, frozen = _source(db, current, windows)

        assert frozen == ("newer-checkpoint" if windows == 1 else "older-checkpoint")
        previous = [] if windows == 1 else [
            "recent-carried-user", "recent-carried-answer", "recent-user", "recent-answer",
        ]
        assert [row["content"] for row in turns] == [
            *previous, "replacement-user", "replacement-answer", "current-user", "current-answer",
        ]


@pytest.mark.parametrize("carrier_kind", ["standalone", "merged"])
@pytest.mark.parametrize("windows", [1, 2])
def test_original_source_keeps_carried_identities_once_and_excludes_protected_tail(tmp_path, carrier_kind, windows):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("s", source="test")
        db.append_messages_batch("s", [
            {"role": "user", "content": "outside-window-user"},
            {"role": "assistant", "content": "outside-window-answer"},
            {"role": "user", "content": "carried-user-0"},
            {"role": "assistant", "content": "same-answer"},
            {"role": "user", "content": "current-user-1"},
            {"role": "assistant", "content": "same-answer"},
            {"role": "user", "content": "protected-live-user"},
            {"role": "assistant", "content": "protected-live-answer"},
        ])
        live = db.get_messages_as_conversation("s", include_row_ids=True)
        previous_ids = [row["message_uid"] for row in live[2:4]]
        current_ids = [row["message_uid"] for row in live[4:6]]
        protected_ids = {row["message_uid"] for row in live[-2:]}
        _compact(db, "checkpoint-1", live[-6:], carrier_kind)
        live = db.get_messages_as_conversation("s", include_row_ids=True)
        _compact(db, "checkpoint-2", live[-4:], carrier_kind)
        current = [row for row in _live_raw(db) if row["message_uid"] in current_ids]
        assert [row["message_uid"] for row in current] == current_ids

        turns, frozen = _source(db, current, windows)

        assert frozen == ("checkpoint-2" if windows == 1 else "checkpoint-1")
        expected_ids = current_ids if windows == 1 else [*previous_ids, *current_ids]
        assert [row["message_uid"] for row in turns] == expected_ids
        assert protected_ids.isdisjoint(row["message_uid"] for row in turns)
        # Equal text from distinct original messages remains meaningful evidence.
        assert sum(row["content"] == "same-answer" for row in turns) == windows


@pytest.mark.parametrize("summarize_merged_user", [False, True])
def test_original_source_tracks_absorbed_users_in_current_and_protected_windows(tmp_path, summarize_merged_user):
    from agent.agent_runtime_helpers import _merge_consecutive_users

    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("s", source="test")
        first = {"role": "user", "content": "first-user"}
        absorbed = {"role": "user", "content": "second-user"}
        db.append_messages_batch("s", [
            {"role": "assistant", "content": "older-answer"}, first, absorbed,
        ])
        # Replay restores durable UIDs even when callers do not request physical row IDs.
        first, absorbed = db.get_messages_as_conversation("s")[-2:]
        _merge_consecutive_users([first, absorbed])
        assert first["_absorbed_message_uids"] == [absorbed["message_uid"]]
        _compact(db, "checkpoint", [first], "standalone")
        archived = db.get_messages_as_conversation("s", include_inactive=True)
        assert any(row["message_uid"] == absorbed["message_uid"] for row in archived)
        live = _live_raw(db)
        assert len(live) == 1
        assert live[0]["_absorbed_message_uids"] == [absorbed["message_uid"]]

        turns, frozen = _source(db, live if summarize_merged_user else [], windows=2)

        assert frozen == ""
        expected = ["older-answer"]
        if summarize_merged_user:
            expected.append("first-user\n\nsecond-user")
        assert [row["content"] for row in turns] == expected


def test_original_source_keeps_parent_history_in_selected_compression_lineage(tmp_path):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("parent", source="test")
        db.append_messages_batch("parent", [
            {"role": "user", "content": "parent-user"},
            {"role": "assistant", "content": "parent-answer"},
        ])
        db.publish_compression_child(
            parent_session_id="parent", child_session_id="s", source="test",
            messages=[{
                "role": "assistant", "content": f"{SUMMARY_PREFIX}\ncheckpoint",
                "_compressed_summary": True,
            }],
            require_compression_lease=False,
        )
        db.append_messages_batch("s", [
            {"role": "user", "content": "current-user"},
            {"role": "assistant", "content": "current-answer"},
        ])

        turns, frozen = _source(db, _live_raw(db), windows=2)

        assert frozen == ""
        assert [row["content"] for row in turns] == [
            "parent-user", "parent-answer", "current-user", "current-answer",
        ]
