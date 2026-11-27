"""Regression (#92231): resumed transcripts must not re-append to state.db.

Root cause: ``get_messages_as_conversation`` / ``get_resume_conversations``
returned plain dicts WITHOUT ``_DB_PERSISTED_MARKER``. Any flush that received
that loaded history without a matching ``conversation_history=`` identity
boundary (compression durable-snapshot adoption, incremental tool-call
persists, rotation preflight on cold resume) treated every loaded row as new
and re-appended the ENTIRE transcript. Compression cycles then doubled the
copies: the incident session grew 998 → 1995 → 3990 → 7981 rows across three
aborted rotations (15,962 active rows / 472 distinct contents).

Fix: rows are stamped durable at materialization time in the store's
conversation read, so a dict built FROM a durable row is persisted by
construction, no matter which caller loads it or how it is later handed to a
flush.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent.context_compressor import _DB_PERSISTED_MARKER as _DB_PERSISTED_MARKER_KEY
from run_agent import AIAgent
from tests.fixtures.storage_read import conversation, transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _make_flush_agent(db, session_id: str):
    """Minimal agent shell that owns the real flush implementation."""
    agent = SimpleNamespace(
        _session_db=db,
        persistence=db,
        _session_db_created=True,
        _persist_disabled=False,
        session_id=session_id,
        _session_persist_lock=None,
        _flushed_db_message_ids=set(),
        _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0,
        _persist_user_message_idx=None,
        _persist_user_message_override=None,
        _persist_user_message_timestamp=None,
        _pending_cli_user_message=None,
    )
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = (
        AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    )
    agent._flush_messages_to_session_db_unlocked = (
        AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    )
    return agent


def _seed_session(db, sid: str, turns: int = 3) -> None:
    seed_session(db, sid, "cli")
    for i in range(turns):
        seed_messages(db, sid, {"role": "user", "content": f"question {i}"},
                      {"role": "assistant", "content": f"answer {i}"})


def test_marker_constant_in_sync() -> None:
    """One shared literal across every module that touches the marker.

    ``agent.turn_finalizer`` and ``agent.session_persistence`` import the
    constant from ``agent.context_compressor`` (single source).
    """
    import agent.context_compressor as cc
    import agent.turn_finalizer as tf
    import agent.session_persistence as sp

    assert _DB_PERSISTED_MARKER_KEY == sp._DB_PERSISTED_MARKER
    assert _DB_PERSISTED_MARKER_KEY == cc._DB_PERSISTED_MARKER
    assert _DB_PERSISTED_MARKER_KEY == tf._DB_PERSISTED_MARKER


def test_loaded_rows_are_stamped_durable(tmp_path: Path) -> None:
    db = storage_handle(tmp_path)
    _seed_session(db, "S1")

    loaded = conversation(db, "S1")
    assert loaded
    assert all(m.get(_DB_PERSISTED_MARKER_KEY) is True for m in loaded)

    resumed = db.messages.read_resume_conversations("S1")
    model_history = [m.as_openai_message() for m in resumed.model]
    display_history = [m.as_openai_message() for m in resumed.display]
    assert model_history and display_history
    assert all(m.get(_DB_PERSISTED_MARKER_KEY) is True for m in model_history)
    assert all(m.get(_DB_PERSISTED_MARKER_KEY) is True for m in display_history)


def test_repeated_identityless_flushes_do_not_amplify(tmp_path: Path) -> None:
    """The #92231 shape: reload + flush cycles must keep the row count flat.

    Each cycle simulates what compression's durable-snapshot adoption (or any
    identity-losing handoff) used to do: load the transcript fresh from the DB
    and flush it with NO ``conversation_history`` boundary. Pre-fix each cycle
    doubled the row count (998 → 1995 → 3990 → 7981 in the incident session).
    """
    db = storage_handle(tmp_path)
    _seed_session(db, "S2", turns=4)
    baseline = len(transcript_rows(db, "S2"))

    for _ in range(3):
        loaded = conversation(db, "S2")
        agent = _make_flush_agent(db, "S2")  # fresh agent: no identity state
        agent._flush_messages_to_session_db(loaded)

    assert len(transcript_rows(db, "S2")) == baseline


def test_new_tail_after_loaded_history_still_flushes(tmp_path: Path) -> None:
    """Guard against over-skipping: only loaded rows are exempt, new turns write."""
    db = storage_handle(tmp_path)
    _seed_session(db, "S3", turns=1)

    loaded = conversation(db, "S3")
    live = [
        *loaded,
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer"},
    ]
    agent = _make_flush_agent(db, "S3")
    agent._flush_messages_to_session_db(live)

    contents = [m.get("content") for m in transcript_rows(db, "S3")]
    assert contents == ["question 0", "answer 0", "new question", "new answer"]


def test_compaction_copy_strips_stamp_so_child_flush_writes(tmp_path: Path) -> None:
    """Rotation handoff: compression copies must stay flushable to the child.

    ``_fresh_compaction_message_copy`` / ``_strip_persistence_markers`` remove
    the marker from assembled compaction output so the rotation flush WRITES
    the compacted transcript to the child session (#57491). Load-stamping must
    not defeat that: a loaded row that goes through the compaction copy is
    written to the child exactly once.
    """
    from agent.context_compressor import _fresh_compaction_message_copy

    db = storage_handle(tmp_path)
    _seed_session(db, "PARENT", turns=2)
    seed_session(db, "CHILD", "cli", parent_session_id="PARENT")

    loaded = conversation(db, "PARENT")
    compacted = [_fresh_compaction_message_copy(m) for m in loaded]
    assert all(_DB_PERSISTED_MARKER_KEY not in m for m in compacted)

    agent = _make_flush_agent(db, "CHILD")
    agent._flush_messages_to_session_db(compacted)
    child_rows = transcript_rows(db, "CHILD")
    assert len(child_rows) == len(loaded)

    # Idempotent thereafter: the flush stamped the copies on write.
    agent._flush_messages_to_session_db(compacted)
    assert len(transcript_rows(db, "CHILD")) == len(loaded)


def test_noop_progress_check_is_marker_insensitive(tmp_path: Path) -> None:
    """A marker-swept no-op copy must still compare equal to stamped input.

    The commit layer's "no progress" check compares compress() output against
    a pre-dispatch deepcopy of the live messages. Load-stamping (#92231) puts
    ``_db_persisted`` on cold-resumed dicts while compress() output is
    marker-swept — a raw ``==`` would misclassify the semantically-identical
    no-op as progress and rotate the session for nothing.
    """
    from agent.conversation_compression import _strip_marker_for_comparison

    db = storage_handle(tmp_path)
    _seed_session(db, "NOOP", turns=2)

    stamped = conversation(db, "NOOP")
    assert all(m.get(_DB_PERSISTED_MARKER_KEY) is True for m in stamped)
    # What a no-op engine hands back after the terminal marker sweep.
    swept = [
        {k: v for k, v in m.items() if k != _DB_PERSISTED_MARKER_KEY}
        for m in stamped
    ]
    assert swept != stamped  # raw == is marker-sensitive — the bug shape
    assert _strip_marker_for_comparison(swept) == _strip_marker_for_comparison(
        stamped
    )

    # Genuine progress must still register as inequality.
    compacted = swept[:-1]
    assert _strip_marker_for_comparison(compacted) != _strip_marker_for_comparison(
        stamped
    )

    # Defensive passthrough shapes.
    assert _strip_marker_for_comparison(None) is None
    assert _strip_marker_for_comparison(["not-a-dict"]) == ["not-a-dict"]
