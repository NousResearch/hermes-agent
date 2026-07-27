"""Behavioral contracts for durable transcript revision fencing."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_compression import CompressionSnapshotStaleError
from hermes_state import (
    CompressionSessionBusyError,
    CompressionTranscriptRevisionError,
    DurableTranscriptRevision,
    SessionDB,
    _durable_revision_from_rows,
)


def _build_compression_agent(
    db: SessionDB,
    session_id: str,
    *,
    install_mock_compressor: bool = True,
) -> Any:
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    if install_mock_compressor:
        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]
        compressor.compression_count = 1
        compressor.protect_first_n = 3
        compressor.protect_last_n = 20
        compressor.threshold_tokens = 100_000
        compressor.context_length = 200_000
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor.last_real_prompt_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        compressor._last_aux_model_failure_model = None
        compressor._last_aux_model_failure_error = None
        agent.context_compressor = compressor
        agent.compression_in_place = False
    return agent


def _append_bypassing_compression_lease(
    db: SessionDB,
    session_id: str,
    *,
    role: str,
    content: str,
) -> None:
    """Simulate a legacy/external writer that bypasses application lease checks."""

    def _write(conn) -> None:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, active) "
            "VALUES (?, ?, ?, ?, 1)",
            (session_id, role, content, 1.0),
        )
        conn.execute(
            "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
            (session_id,),
        )

    db._execute_write(_write)


class _FinalMessage:
    content = "done"
    tool_calls = None
    reasoning = None
    reasoning_content = None
    reasoning_details = None

    @staticmethod
    def model_dump(exclude_none: bool = False):
        return {"role": "assistant", "content": "done"}


class _FinalResponse:
    def __init__(self) -> None:
        self.id = "resp-revision"
        self.model = "test/model"
        self.created = 0
        self.system_fingerprint = None
        self.choices = [
            SimpleNamespace(
                message=_FinalMessage(),
                finish_reason="stop",
            )
        ]
        self.usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=1,
            total_tokens=11,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )


class _RecordingAPI:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args[0] if args else kwargs)
        return _FinalResponse()


class _SequenceAPI:
    def __init__(self, *responses) -> None:
        self.calls = []
        self.responses = list(responses)

    def __call__(self, *args, **kwargs):
        self.calls.append(args[0] if args else kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _tool_response():
    tool_call = SimpleNamespace(
        id="call_revision",
        type="function",
        function=SimpleNamespace(name="web_search", arguments='{"query":"x"}'),
    )
    message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    return SimpleNamespace(
        id="resp-tool-revision",
        model="test/model",
        created=0,
        system_fingerprint=None,
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=SimpleNamespace(
            prompt_tokens=150_000,
            completion_tokens=10,
            total_tokens=150_010,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )


def _synthetic_stale_error(db: SessionDB, session_id: str):
    expected = db.get_active_message_revision(session_id)
    observed = DurableTranscriptRevision(
        session_id=session_id,
        active_message_count=expected.active_message_count + 1,
        max_active_message_id=expected.max_active_message_id + 1,
    )
    return CompressionSnapshotStaleError(
        session_id=session_id,
        expected_revision=expected,
        observed_revision=observed,
    )


def test_messages_and_revision_share_one_ordered_active_snapshot(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "revision-snapshot"
    db.create_session(session_id, source="test")
    first_id = db.append_message(session_id, "user", "first")
    second_id = db.append_message(session_id, "assistant", "second")

    messages, revision = db.get_messages_as_conversation(
        session_id, with_revision=True
    )

    assert [message["content"] for message in messages] == ["first", "second"]
    assert revision == DurableTranscriptRevision(
        session_id=session_id,
        active_message_count=2,
        max_active_message_id=second_id,
    )
    assert second_id > first_id


def test_revision_aware_projection_preserves_opt_in_row_ids(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "revision-row-ids"
    db.create_session(session_id=session_id, source="test")
    first_id = db.append_message(session_id, "user", "hello")
    second_id = db.append_message(session_id, "assistant", "world")

    messages, revision = db.get_messages_as_conversation(
        session_id,
        include_row_ids=True,
        with_revision=True,
    )

    assert [message["_row_id"] for message in messages] == [first_id, second_id]
    assert revision.active_message_count == 2


def test_revision_progresses_after_append_and_active_set_replacement(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "revision-progress"
    db.create_session(session_id, source="test")
    db.append_message(session_id, "user", "before")
    _messages, before = db.get_messages_as_conversation(
        session_id, with_revision=True
    )

    appended_id = db.append_message(session_id, "assistant", "after")
    after_append = db.get_active_message_revision(session_id)

    assert after_append == DurableTranscriptRevision(
        session_id=session_id,
        active_message_count=before.active_message_count + 1,
        max_active_message_id=appended_id,
    )

    db.archive_and_compact(
        session_id,
        [{"role": "user", "content": "compacted live transcript"}],
    )
    messages, after_compaction = db.get_messages_as_conversation(
        session_id, with_revision=True
    )

    assert [message["content"] for message in messages] == [
        "compacted live transcript"
    ]
    assert after_compaction.active_message_count == 1
    assert after_compaction.max_active_message_id > after_append.max_active_message_id
    assert after_compaction != after_append


def test_missing_and_empty_sessions_have_zero_active_revision(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")

    missing_messages, missing_revision = db.get_messages_as_conversation(
        "missing-session", with_revision=True
    )
    assert missing_messages == []
    assert missing_revision == DurableTranscriptRevision(
        session_id="missing-session",
        active_message_count=0,
        max_active_message_id=0,
    )

    db.create_session("empty-session", source="test")
    assert db.get_active_message_revision(
        "empty-session"
    ) == DurableTranscriptRevision(
        session_id="empty-session",
        active_message_count=0,
        max_active_message_id=0,
    )


def test_revision_ignores_inactive_rows_when_all_are_loaded(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "revision-inactive"
    db.create_session(session_id, source="test")
    inactive_id = db.append_message(session_id, "assistant", "inactive")
    active_id = db.append_message(session_id, "user", "active")

    def _mark_non_active(conn) -> None:
        conn.execute("UPDATE messages SET active = 0 WHERE id = ?", (inactive_id,))

    db._execute_write(_mark_non_active)

    messages, revision = db.get_messages_as_conversation(
        session_id,
        include_inactive=True,
        with_revision=True,
    )

    assert [message["content"] for message in messages] == [
        "inactive",
        "active",
    ]
    assert revision == DurableTranscriptRevision(
        session_id=session_id,
        active_message_count=1,
        max_active_message_id=active_id,
    )


def test_revision_builder_treats_null_active_as_inactive() -> None:
    revision = _durable_revision_from_rows(
        "legacy-session",
        [
            {"id": 41, "session_id": "legacy-session", "active": None},
            {"id": 42, "session_id": "legacy-session", "active": 0},
            {"id": 43, "session_id": "legacy-session", "active": 1},
        ],
    )

    assert revision == DurableTranscriptRevision(
        session_id="legacy-session",
        active_message_count=1,
        max_active_message_id=43,
    )


def test_stale_error_carries_only_session_and_revision_fence() -> None:
    expected = DurableTranscriptRevision("session-a", 2, 12)
    observed = DurableTranscriptRevision("session-a", 3, 13)

    error = CompressionSnapshotStaleError(
        session_id="session-a",
        expected_revision=expected,
        observed_revision=observed,
    )

    assert error.session_id == "session-a"
    assert error.expected_revision == expected
    assert error.observed_revision == observed
    assert "session-a" in str(error)
    assert "first" not in str(error)


def test_projection_length_mismatch_with_equal_revision_still_compresses(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "projection-mismatch"
    db.create_session(session_id, source="webui")
    for index in range(247):
        role = "user" if index % 2 == 0 else "assistant"
        db.append_message(session_id, role, f"durable {index}")
    expected_revision = db.get_active_message_revision(session_id)
    projection = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"projected {index}",
            "_db_persisted": True,
        }
        for index in range(209)
    ]
    agent = _build_compression_agent(db, session_id)
    agent._durable_transcript_revision = expected_revision

    compressed, _system_prompt = agent._compress_context(
        projection, "sys", approx_tokens=120_000
    )

    assert compressed is not projection
    assert compressed[0]["content"] == "[CONTEXT COMPACTION] summary"
    agent.context_compressor.compress.assert_called_once()


def test_external_append_after_snapshot_raises_stale_without_compressing(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "true-durable-race"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "snapshot row")
    expected_revision = db.get_active_message_revision(session_id)
    projection = [
        {"role": "user", "content": "snapshot row", "_db_persisted": True}
    ]
    db.append_message(session_id, "assistant", "external committed row")
    agent = _build_compression_agent(db, session_id)
    agent._durable_transcript_revision = expected_revision

    with pytest.raises(CompressionSnapshotStaleError) as caught:
        agent._compress_context(projection, "sys", approx_tokens=120_000)

    assert caught.value.expected_revision == expected_revision
    assert caught.value.observed_revision == db.get_active_message_revision(session_id)
    agent.context_compressor.compress.assert_not_called()
    assert db.find_live_compression_child(session_id) is None
    assert [
        message["content"]
        for message in db.get_messages_as_conversation(session_id)
    ] == ["snapshot row", "external committed row"]
    assert db.get_compression_lock_holder(session_id) is None


def test_rotation_commit_fence_rejects_mutation_after_precheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "rotation-commit-fence"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "snapshot row")
    expected_revision = db.get_active_message_revision(session_id)
    agent = _build_compression_agent(db, session_id)
    setattr(agent, "_durable_transcript_revision", expected_revision)
    original_publish = db.publish_compression_child

    def _publish_after_late_write(**kwargs) -> None:
        _append_bypassing_compression_lease(
            db,
            session_id,
            role="assistant",
            content="late durable row",
        )
        original_publish(**kwargs)

    monkeypatch.setattr(db, "publish_compression_child", _publish_after_late_write)

    with pytest.raises(CompressionSnapshotStaleError) as caught:
        agent._compress_context(
            [{"role": "user", "content": "snapshot row", "_db_persisted": True}],
            "sys",
            approx_tokens=120_000,
        )

    assert caught.value.expected_revision == expected_revision
    assert caught.value.observed_revision == db.get_active_message_revision(session_id)
    getattr(agent, "context_compressor").compress.assert_called_once()
    assert db.find_live_compression_child(session_id) is None
    assert [
        message["content"]
        for message in db.get_messages(session_id)
    ] == ["snapshot row", "late durable row"]
    assert db.get_compression_lock_holder(session_id) is None


def test_in_place_commit_fence_rejects_mutation_after_precheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "in-place-commit-fence"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "snapshot row")
    expected_revision = db.get_active_message_revision(session_id)
    agent = _build_compression_agent(db, session_id)
    setattr(agent, "compression_in_place", True)
    setattr(agent, "_durable_transcript_revision", expected_revision)
    original_archive = db.archive_and_compact

    def _archive_after_late_write(*args, **kwargs) -> int:
        _append_bypassing_compression_lease(
            db,
            session_id,
            role="assistant",
            content="late durable row",
        )
        return original_archive(*args, **kwargs)

    monkeypatch.setattr(db, "archive_and_compact", _archive_after_late_write)

    with pytest.raises(CompressionSnapshotStaleError) as caught:
        agent._compress_context(
            [{"role": "user", "content": "snapshot row", "_db_persisted": True}],
            "sys",
            approx_tokens=120_000,
        )

    assert caught.value.expected_revision == expected_revision
    assert caught.value.observed_revision == db.get_active_message_revision(session_id)
    getattr(agent, "context_compressor").compress.assert_called_once()
    all_rows = db.get_messages(session_id, include_inactive=True)
    assert [row["content"] for row in all_rows] == [
        "snapshot row",
        "late durable row",
    ]
    assert all(row["active"] == 1 and row["compacted"] == 0 for row in all_rows)
    assert db.get_compression_lock_holder(session_id) is None


def test_replace_messages_rejects_another_writer_compression_lease(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "replace-under-lease"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "original")
    assert db.try_acquire_compression_lock(
        session_id, "compression-owner", ttl_seconds=60
    )

    with pytest.raises(RuntimeError, match="being compressed"):
        db.replace_messages(
            session_id,
            [{"role": "user", "content": "stale replacement"}],
        )

    assert [row["content"] for row in db.get_messages(session_id)] == ["original"]


def test_rewind_rejects_another_writer_compression_lease(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "rewind-under-lease"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "keep")
    db.append_message(session_id, "assistant", "would be rewound")
    target_message_id = db.get_messages(session_id)[0]["id"]
    assert db.try_acquire_compression_lock(
        session_id, "compression-owner", ttl_seconds=60
    )

    with pytest.raises(RuntimeError, match="being compressed"):
        db.rewind_to_message(session_id, target_message_id)

    assert [row["content"] for row in db.get_messages(session_id)] == [
        "keep",
        "would be rewound",
    ]


def test_archive_and_compact_rejects_another_writer_compression_lease(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "archive-under-lease"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "original")
    assert db.try_acquire_compression_lock(
        session_id, "compression-owner", ttl_seconds=60
    )

    with pytest.raises(RuntimeError, match="being compressed"):
        db.archive_and_compact(
            session_id,
            [{"role": "user", "content": "stale summary"}],
        )

    assert [row["content"] for row in db.get_messages(session_id)] == ["original"]
    assert db.get_messages(session_id, include_inactive=True)[0]["compacted"] == 0


def test_restore_rewound_rejects_another_writer_compression_lease(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "restore-under-lease"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "rewound row")
    target_message_id = db.get_messages(session_id)[0]["id"]
    db.rewind_to_message(session_id, target_message_id)
    assert db.try_acquire_compression_lock(
        session_id, "compression-owner", ttl_seconds=60
    )

    with pytest.raises(RuntimeError, match="being compressed"):
        db.restore_rewound(session_id, target_message_id)

    assert db.get_messages(session_id) == []
    assert db.get_messages(session_id, include_inactive=True)[0]["active"] == 0


def test_in_place_compaction_also_rejects_stale_revision(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "in-place-stale"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "snapshot row")
    expected_revision = db.get_active_message_revision(session_id)
    db.append_message(session_id, "assistant", "external committed row")
    agent = _build_compression_agent(db, session_id)
    agent.compression_in_place = True
    agent._durable_transcript_revision = expected_revision

    with pytest.raises(CompressionSnapshotStaleError):
        agent._compress_context(
            [{"role": "user", "content": "snapshot row", "_db_persisted": True}],
            "sys",
            approx_tokens=120_000,
        )

    agent.context_compressor.compress.assert_not_called()
    assert db.get_compression_lock_holder(session_id) is None
    assert [
        message["content"]
        for message in db.get_messages_as_conversation(session_id)
    ] == ["snapshot row", "external committed row"]


def test_in_place_compaction_rebinds_revision_before_next_append(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "in-place-rebind"
    db.create_session(session_id, source="cli")
    db.append_message(session_id, "user", "original row")
    agent = _build_compression_agent(db, session_id)
    agent.compression_in_place = True

    compressed, _ = agent._compress_context(
        [{"role": "user", "content": "original row", "_db_persisted": True}],
        "sys",
        approx_tokens=120_000,
    )

    assert agent.session_id == session_id
    assert agent._durable_transcript_revision == db.get_active_message_revision(
        session_id
    )

    next_message = {"role": "assistant", "content": "next append"}
    agent._flush_messages_to_session_db([next_message], compressed)
    assert agent._durable_transcript_revision == db.get_active_message_revision(
        session_id
    )


def test_session_db_owner_tracks_revision_without_explicit_transport(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "compatibility-owner"
    db.create_session(session_id, source="cli")
    db.append_message(session_id, "user", "existing user")
    db.append_message(session_id, "assistant", "existing assistant")

    agent = _build_compression_agent(db, session_id)
    assert agent._durable_transcript_revision == db.get_active_message_revision(
        session_id
    )

    owned_message = {"role": "user", "content": "owned append"}
    agent._session_db_created = True
    agent._flush_messages_to_session_db([owned_message], None)
    assert owned_message["_db_persisted"] is True
    assert agent._durable_transcript_revision == db.get_active_message_revision(
        session_id
    )

    compressed, _ = agent._compress_context(
        [owned_message],
        system_message="system",
        force=True,
    )

    assert compressed != [owned_message]
    agent.context_compressor.compress.assert_called_once()


def test_legacy_history_without_revision_reloads_atomic_durable_snapshot(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "legacy-atomic-reload"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "A")
    db.append_message(session_id, "assistant", "B")
    captured_history = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
    ]
    db.append_message(session_id, "user", "C")
    agent = _build_compression_agent(db, session_id)
    agent.compression_enabled = True
    agent.context_compressor.protect_first_n = 0
    agent.context_compressor.protect_last_n = 0
    agent.context_compressor.threshold_tokens = 1
    agent.context_compressor.context_length = 100
    agent.context_compressor.should_defer_preflight_to_real_usage.return_value = False
    agent.context_compressor.get_active_compression_failure_cooldown.return_value = None
    agent.context_compressor.should_compress.return_value = True
    agent._disable_streaming = True
    agent._interruptible_api_call = _RecordingAPI()

    agent.run_conversation("D", conversation_history=captured_history)

    compressed_input = agent.context_compressor.compress.call_args.args[0]
    assert "C" in [message.get("content") for message in compressed_input]


def test_client_managed_history_survives_when_no_durable_rows_exist(
    tmp_path: Path,
) -> None:
    """Re-anchoring must not erase a stateless caller's own projection.

    ``/responses`` lets a client supply ``conversation_history`` against a
    brand-new session id. There are no durable rows to be stale against, so
    the supplied bytes are the only context and must reach the provider.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "client-managed-stateless"
    db.create_session(session_id, source="api")
    agent = _build_compression_agent(db, session_id, install_mock_compressor=False)
    agent._persist_disabled = True
    agent.compression_enabled = False
    api = _RecordingAPI()
    agent._disable_streaming = True
    agent._interruptible_api_call = api

    result = agent.run_conversation(
        "client C",
        conversation_history=[
            {"role": "user", "content": "client A"},
            {"role": "assistant", "content": "client B"},
        ],
    )

    assert result["final_response"] == "done"
    assert [
        message.get("content")
        for message in api.calls[0]["messages"]
        if message.get("role") in {"user", "assistant"}
    ] == ["client A", "client B", "client C"]
    assert agent._durable_transcript_revision.active_message_count == 0


def test_supplied_revision_keeps_a_longer_caller_projection(tmp_path: Path) -> None:
    """A tokened projection is never silently re-anchored onto durable rows.

    The gateway's FTS write-corruption guard (#50502) replays a live in-memory
    transcript that is longer than the persisted rows; re-anchoring it would
    reinstate exactly the amnesia that guard exists to prevent.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "live-longer-than-durable"
    db.create_session(session_id, source="telegram")
    db.append_message(session_id, "user", "persisted only")
    revision = db.get_active_message_revision(session_id)
    agent = _build_compression_agent(db, session_id, install_mock_compressor=False)
    agent._persist_disabled = True
    agent.compression_enabled = False
    api = _RecordingAPI()
    agent._disable_streaming = True
    agent._interruptible_api_call = api

    agent.run_conversation(
        "next",
        conversation_history=[
            {"role": "user", "content": "persisted only"},
            {"role": "assistant", "content": "live reply"},
            {"role": "user", "content": "live follow-up"},
            {"role": "assistant", "content": "live reply 2"},
        ],
        conversation_history_revision=revision,
    )

    assert [
        message.get("content")
        for message in api.calls[0]["messages"]
        if message.get("role") in {"user", "assistant"}
    ] == [
        "persisted only",
        "live reply",
        "live follow-up",
        "live reply 2",
        "next",
    ]


def test_legacy_history_reload_failure_is_projection_unverifiable_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "legacy-reload-failure"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "durable")
    agent = _build_compression_agent(db, session_id)
    api = _RecordingAPI()
    agent._disable_streaming = True
    agent._interruptible_api_call = api
    monkeypatch.setattr(
        db,
        "get_messages_as_conversation",
        MagicMock(side_effect=RuntimeError("read failed")),
    )

    result = agent.run_conversation(
        "new turn",
        conversation_history=[{"role": "user", "content": "unverified"}],
    )

    assert result["error"] == "compression_projection_unverifiable"
    assert result["compression_projection_unverifiable"] is True
    assert api.calls == []
    agent.context_compressor.compress.assert_not_called()


def test_in_place_commit_failure_restores_mutated_messages_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "in-place-lease-loss"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "original")
    agent = _build_compression_agent(db, session_id)
    agent.compression_in_place = True
    messages = [{"role": "user", "content": "original", "_db_persisted": True}]

    def _mutating_compress(live_messages, **_kwargs):
        live_messages[:] = [{"role": "user", "content": "mutated summary"}]
        return [{"role": "user", "content": "compressed summary"}]

    agent.context_compressor.compress.side_effect = _mutating_compress
    monkeypatch.setattr(
        db,
        "archive_and_compact",
        MagicMock(side_effect=CompressionSessionBusyError("lease expired")),
    )

    from agent import conversation_compression

    with pytest.raises(conversation_compression.CompressionCommitFailedError):
        agent._compress_context(messages, "system", force=True)

    assert [message["content"] for message in messages] == ["original"]
    assert [row["content"] for row in db.get_messages(session_id)] == ["original"]
    assert db.get_compression_lock_holder(session_id) is None


def test_stale_replace_after_compaction_release_is_rejected(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "replace-after-compression"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "T0")
    revision0 = db.get_active_message_revision(session_id)
    db.archive_and_compact(
        session_id,
        [{"role": "user", "content": "T1 summary"}],
        expected_revision=revision0,
    )

    with pytest.raises(CompressionTranscriptRevisionError):
        db.replace_messages(
            session_id,
            [{"role": "user", "content": "stale T0 rewrite"}],
            expected_revision=revision0,
        )

    assert [row["content"] for row in db.get_messages(session_id)] == ["T1 summary"]


def test_api_content_backfill_rejects_adverse_compression_lease(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "api-content-under-lease"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "clean")
    assert db.try_acquire_compression_lock(session_id, "owner", ttl_seconds=60)

    with pytest.raises(CompressionSessionBusyError):
        db.set_latest_user_api_content(session_id, "clean", "wire")

    assert db.get_messages(session_id)[0]["api_content"] is None


def test_clear_messages_rejects_adverse_compression_lease(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "clear-under-lease"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "keep")
    assert db.try_acquire_compression_lock(session_id, "owner", ttl_seconds=60)

    with pytest.raises(CompressionSessionBusyError):
        db.clear_messages(session_id)

    assert [row["content"] for row in db.get_messages(session_id)] == ["keep"]


def test_delete_session_rejects_adverse_compression_lease(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "delete-under-lease"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "keep")
    assert db.try_acquire_compression_lock(session_id, "owner", ttl_seconds=60)

    with pytest.raises(CompressionSessionBusyError):
        db.delete_session(session_id)

    assert db.get_session(session_id) is not None
    assert [row["content"] for row in db.get_messages(session_id)] == ["keep"]


def test_bulk_delete_rejects_if_any_target_has_adverse_lease(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    for session_id in ("bulk-free", "bulk-leased"):
        db.create_session(session_id, source="webui")
        db.append_message(session_id, "user", session_id)
    assert db.try_acquire_compression_lock("bulk-leased", "owner", ttl_seconds=60)

    with pytest.raises(CompressionSessionBusyError):
        db.delete_sessions(["bulk-free", "bulk-leased"])

    assert db.get_session("bulk-free") is not None
    assert db.get_session("bulk-leased") is not None


def test_prune_rejects_target_with_adverse_lease(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "prune-leased"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "keep")
    db.end_session(session_id, "completed")
    assert db.try_acquire_compression_lock(session_id, "owner", ttl_seconds=60)

    with pytest.raises(CompressionSessionBusyError):
        db.prune_sessions(older_than_days=-1)

    assert db.get_session(session_id) is not None


def test_rewind_rejects_stale_revision_after_late_append(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "rewind-stale"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "first")
    target_id = db.get_messages(session_id)[0]["id"]
    revision = db.get_active_message_revision(session_id)
    db.append_message(session_id, "assistant", "late")

    with pytest.raises(CompressionTranscriptRevisionError):
        db.rewind_to_message(
            session_id,
            target_id,
            expected_revision=revision,
        )

    assert [message["content"] for message in db.get_messages_as_conversation(session_id)] == [
        "first",
        "late",
    ]


def test_recent_user_messages_and_revision_share_one_snapshot(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "rewind-atomic-load"
    db.create_session(session_id, source="telegram")
    user_id = db.append_message(session_id, "user", "target")
    db.append_message(session_id, "assistant", "answer")

    recents, revision = db.list_recent_user_messages(
        session_id,
        with_revision=True,
    )

    assert recents[0]["id"] == user_id
    assert revision == db.get_active_message_revision(session_id)


def test_restore_rejects_stale_revision_after_late_append(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "restore-stale"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "rewound")
    target_id = db.get_messages(session_id)[0]["id"]
    revision0 = db.get_active_message_revision(session_id)
    db.rewind_to_message(session_id, target_id, expected_revision=revision0)
    revision1 = db.get_active_message_revision(session_id)
    db.append_message(session_id, "user", "late active")

    with pytest.raises(CompressionTranscriptRevisionError):
        db.restore_rewound(
            session_id,
            target_id,
            expected_revision=revision1,
        )

    rows = db.get_messages(session_id, include_inactive=True)
    assert [(row["content"], row["active"]) for row in rows] == [
        ("rewound", False),
        ("late active", True),
    ]


def test_api_content_backfill_advances_revision_without_count_or_max_id_change(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "api-content-revision"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "clean")
    before = db.get_active_message_revision(session_id)

    assert db.set_latest_user_api_content(session_id, "clean", "wire") == 1

    after = db.get_active_message_revision(session_id)
    assert after.active_message_count == before.active_message_count
    assert after.max_active_message_id == before.max_active_message_id
    assert after != before


def test_api_content_backfill_returns_the_committed_revision(tmp_path: Path) -> None:
    """The backfill's own writer must be able to re-fence without a second read."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "api-content-revision-return"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "clean")

    updated, revision = db.set_latest_user_api_content(
        session_id, "clean", "wire", with_revision=True
    )

    assert updated == 1
    assert revision == db.get_active_message_revision(session_id)


def test_display_only_stamp_does_not_move_the_durable_revision(
    tmp_path: Path,
) -> None:
    """Presentation sidecars are not model-facing, so they must not fence.

    ``set_latest_matching_message_display_kind`` runs outside the compression
    lease on every gateway/CLI turn; counting it as a revision change would
    abort the next compression as spuriously stale.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "display-only-stamp"
    db.create_session(session_id, source="cli")
    db.append_message(session_id, "user", "hello")
    before = db.get_active_message_revision(session_id)

    assert db.set_latest_matching_message_display_kind(
        session_id,
        role="user",
        content="hello",
        display_kind="voice",
        display_metadata={"source": "stt"},
    )

    assert db.get_active_message_revision(session_id) == before


def test_rotation_rebinds_revision_before_child_append(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_session_id = "revision-parent"
    db.create_session(parent_session_id, source="cli")
    db.append_message(parent_session_id, "user", "parent row")
    agent = _build_compression_agent(db, parent_session_id)
    projection = [
        {"role": "user", "content": "parent row", "_db_persisted": True}
    ]

    compressed, _ = agent._compress_context(
        projection,
        system_message="system",
        force=True,
    )

    child_session_id = agent.session_id
    assert child_session_id != parent_session_id
    assert agent._durable_transcript_revision == db.get_active_message_revision(
        child_session_id
    )
    assert agent._durable_transcript_revision.session_id == child_session_id

    child_append = {"role": "assistant", "content": "child append"}
    agent._flush_messages_to_session_db([child_append], compressed)

    assert child_append["_db_persisted"] is True
    assert agent._durable_transcript_revision == db.get_active_message_revision(
        child_session_id
    )


def test_preflight_stale_revision_returns_actionable_result_without_api_call(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "preflight-stale"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "snapshot user")
    expected_revision = db.get_active_message_revision(session_id)
    db.append_message(session_id, "assistant", "external answer")

    agent = _build_compression_agent(db, session_id)
    agent.compression_enabled = True
    agent.context_compressor.protect_first_n = 0
    agent.context_compressor.protect_last_n = 0
    agent.context_compressor.threshold_tokens = 1
    agent.context_compressor.context_length = 100
    agent.context_compressor.compression_count = 0
    agent.context_compressor.should_defer_preflight_to_real_usage.return_value = False
    agent.context_compressor.get_active_compression_failure_cooldown.return_value = None
    agent.context_compressor.should_compress.return_value = True
    api = _RecordingAPI()
    agent._disable_streaming = True
    agent._interruptible_api_call = api

    result = agent.run_conversation(
        "new user turn",
        conversation_history=[{"role": "user", "content": "snapshot user"}],
        conversation_history_revision={
            "session_id": session_id,
            "active_message_count": expected_revision.active_message_count,
            "max_active_message_id": expected_revision.max_active_message_id,
        },
    )

    assert result["completed"] is False
    assert result["failed"] is False
    assert result["partial"] is True
    assert result["compression_snapshot_stale"] is True
    assert "reload" in result["final_response"].lower()
    assert api.calls == []
    agent.context_compressor.compress.assert_not_called()
    assert db.find_live_compression_child(session_id) is None
    assert [
        message["content"]
        for message in db.get_messages_as_conversation(session_id)
    ] == ["snapshot user", "external answer"]
    assert db.get_compression_lock_holder(session_id) is None


def test_pre_api_stale_revision_stops_before_provider_call(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "pre-api-stale"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "history user")
    db.append_message(session_id, "assistant", "history assistant")
    agent = _build_compression_agent(db, session_id)
    agent.compression_enabled = True
    agent._disable_streaming = True
    agent._persist_disabled = True
    agent.context_compressor.should_defer_preflight_to_real_usage.return_value = False
    agent.context_compressor.get_active_compression_failure_cooldown.return_value = None
    agent.context_compressor.should_compress_preflight.return_value = False
    agent.context_compressor.should_compress.return_value = True
    api = _RecordingAPI()
    agent._interruptible_api_call = api

    with patch.object(
        agent,
        "_compress_context",
        side_effect=_synthetic_stale_error(db, session_id),
    ) as compress:
        result = agent.run_conversation(
            "new question",
            conversation_history=[
                {"role": "user", "content": "history user"},
                {"role": "assistant", "content": "history assistant"},
            ],
        )

    assert result["compression_snapshot_stale"] is True
    assert result["partial"] is True
    assert api.calls == []
    compress.assert_called_once()


@pytest.mark.parametrize(
    ("status_code", "message", "provider"),
    [
        (413, "Request entity too large", "unknown"),
        (
            400,
            "This endpoint's maximum context length is 128000 tokens; "
            "the request has too many tokens",
            "unknown",
        ),
        (
            429,
            "Extra usage is required for long context requests over 200k tokens",
            "anthropic",
        ),
    ],
)
def test_provider_overflow_stale_revision_prevents_retry_payload(
    tmp_path: Path,
    status_code: int,
    message: str,
    provider: str,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = f"overflow-stale-{status_code}"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "history user")
    db.append_message(session_id, "assistant", "history assistant")
    agent = _build_compression_agent(db, session_id)
    agent.provider = provider
    agent.compression_enabled = True
    agent._disable_streaming = True
    agent._persist_disabled = True
    agent.context_compressor.should_defer_preflight_to_real_usage.return_value = False
    agent.context_compressor.get_active_compression_failure_cooldown.return_value = None
    agent.context_compressor.should_compress_preflight.return_value = False
    agent.context_compressor.should_compress.return_value = False
    provider_error = Exception(message)
    setattr(provider_error, "status_code", status_code)
    api = _SequenceAPI(provider_error)
    agent._interruptible_api_call = api

    with patch.object(
        agent,
        "_compress_context",
        side_effect=_synthetic_stale_error(db, session_id),
    ) as compress:
        result = agent.run_conversation(
            "new question",
            conversation_history=[
                {"role": "user", "content": "history user"},
                {"role": "assistant", "content": "history assistant"},
            ],
        )

    assert result["compression_snapshot_stale"] is True
    assert result["partial"] is True
    assert len(api.calls) == 1
    compress.assert_called_once()


def test_post_tool_stale_revision_prevents_next_provider_payload(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "post-tool-stale"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "history user")
    agent = _build_compression_agent(db, session_id)
    agent.compression_enabled = True
    agent._disable_streaming = True
    agent._persist_disabled = True
    agent.context_compressor.last_prompt_tokens = 150_000
    agent.context_compressor.should_defer_preflight_to_real_usage.return_value = True
    agent.context_compressor.get_active_compression_failure_cooldown.return_value = None
    agent.context_compressor.should_compress_preflight.return_value = False
    agent.context_compressor.should_compress.return_value = True
    agent.valid_tool_names = {"web_search"}
    api = _SequenceAPI(_tool_response())
    agent._interruptible_api_call = api

    with (
        patch.object(
            agent,
            "_compress_context",
            side_effect=_synthetic_stale_error(db, session_id),
        ) as compress,
        patch("run_agent.handle_function_call", return_value='{"ok":true}'),
    ):
        result = agent.run_conversation("use a tool")

    assert result["compression_snapshot_stale"] is True
    assert result["partial"] is True
    assert len(api.calls) == 1
    compress.assert_called_once()


def test_post_tool_stale_preserves_effect_once_without_replay_instruction(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "post-tool-stale-persisted"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "history user")
    agent = _build_compression_agent(db, session_id)
    agent.compression_enabled = True
    agent._disable_streaming = True
    agent.context_compressor.last_prompt_tokens = 150_000
    agent.context_compressor.should_defer_preflight_to_real_usage.return_value = True
    agent.context_compressor.get_active_compression_failure_cooldown.return_value = None
    agent.context_compressor.should_compress_preflight.return_value = False
    agent.context_compressor.should_compress.return_value = True
    agent.valid_tool_names = {"web_search"}
    api = _SequenceAPI(_tool_response())
    agent._interruptible_api_call = api
    effects = []

    def _counting_tool(*_args, **_kwargs):
        effects.append("ran")
        return '{"ok":true}'

    with (
        patch.object(
            agent,
            "_compress_context",
            side_effect=_synthetic_stale_error(db, session_id),
        ),
        patch("run_agent.handle_function_call", side_effect=_counting_tool),
    ):
        result = agent.run_conversation("use a tool")

    assert effects == ["ran"]
    assert len(api.calls) == 1
    assert result["compression_effects_preserved"] is True
    guidance = result["final_response"].lower()
    assert "again" not in guidance
    assert "resend" not in guidance
    assert "retry" not in guidance


def test_run_conversation_normalizes_revision_without_provider_payload_leak(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "revision-transport"
    db.create_session(session_id, source="webui")
    first_id = db.append_message(session_id, "user", "existing")
    agent = _build_compression_agent(
        db,
        session_id,
        install_mock_compressor=False,
    )
    agent._persist_disabled = True
    agent.compression_enabled = False
    api = _RecordingAPI()
    agent._disable_streaming = True
    agent._interruptible_api_call = api
    raw_revision = {
        "session_id": session_id,
        "active_message_count": 1,
        "max_active_message_id": first_id,
    }
    captured = {}

    from agent import conversation_loop

    real_build_turn_context = conversation_loop.build_turn_context

    def _capture_turn_context(*args, **kwargs):
        context = real_build_turn_context(*args, **kwargs)
        captured["context"] = context
        return context

    with patch.object(
        conversation_loop,
        "build_turn_context",
        side_effect=_capture_turn_context,
    ):
        result = agent.run_conversation(
            "new question",
            conversation_history=[{"role": "user", "content": "existing"}],
            conversation_history_revision=raw_revision,
        )

    expected = DurableTranscriptRevision(
        session_id=session_id,
        active_message_count=1,
        max_active_message_id=first_id,
    )
    assert result["final_response"] == "done"
    assert captured["context"].durable_transcript_revision == expected
    assert agent._durable_transcript_revision == expected
    assert len(api.calls) == 1
    assert "conversation_history_revision" not in api.calls[0]
    assert all(
        "conversation_history_revision" not in message
        for message in api.calls[0]["messages"]
    )
