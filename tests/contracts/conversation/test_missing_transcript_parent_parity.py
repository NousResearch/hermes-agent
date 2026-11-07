"""Transcript mutations leave absent owning sessions absent."""

from datetime import UTC, datetime

import pytest

from persistence import SessionRowMissing
from persistence.conversation.messages import CooldownSnapshot, NewMessage, ReplaceMode, TranscriptRead, TranscriptView
from tests.contracts.conversation.test_messages import Totals, totals

STAMP = datetime(2026, 9, 28, tzinfo=UTC).timestamp()


@pytest.mark.parametrize("operation", ["append", "batch", "replace", "compact"])
def test_nonempty_transcript_write_requires_an_existing_session(backend, scope_a, operation):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    message = NewMessage("user", "orphan", timestamp=STAMP)
    with pytest.raises(SessionRowMissing, match="FOREIGN KEY constraint failed") as refused:
        if operation == "append":
            repo.append("absent", message)
        elif operation == "batch":
            repo.append_batch("absent", [message])
        elif operation == "replace":
            repo.replace_transcript("absent", [message], ReplaceMode.DESTRUCTIVE)
        else:
            repo.compact("absent", [message])
    assert (refused.value.cause, refused.value.entity_id) == ("session_row_missing", "absent")
    assert store_a.sessions.get("absent") is None
    assert repo.read_transcript("absent", TranscriptRead(view=TranscriptView.ALL)) == ()
    assert totals(backend, scope_a, "absent") == Totals(0, 0, 0)


@pytest.mark.parametrize("operation", ["batch", "replace", "compact"])
def test_empty_transcript_write_leaves_the_session_absent(backend, scope_a, operation):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    if operation == "batch":
        assert (repo.append_batch("absent", [])).inserted == 0
    elif operation == "replace":
        assert (repo.replace_transcript("absent", [], ReplaceMode.DESTRUCTIVE)).survivor_ids == ()
    else:
        assert (repo.compact("absent", [])).active_count == 0
    assert store_a.sessions.get("absent") is None
    assert repo.read_transcript("absent", TranscriptRead(view=TranscriptView.ALL)) == ()
    assert totals(backend, scope_a, "absent") == Totals(0, 0, 0)


@pytest.mark.parametrize("patch", [{}, {"tokens": 7}])
@pytest.mark.parametrize("with_rows", [False, True])
def test_compaction_patch_refuses_a_missing_session_before_inserting(backend, scope_a, patch, with_rows):
    store_a = backend.open(scope_a)
    rows = [NewMessage("assistant", "summary", timestamp=STAMP)] if with_rows else []
    with pytest.raises(ValueError, match="Session not found: absent"):
        store_a.messages.compact("absent", rows, model_config_patch=patch)
    assert store_a.sessions.get("absent") is None
    assert store_a.messages.read_transcript("absent", TranscriptRead(view=TranscriptView.ALL)) == ()
    assert totals(backend, scope_a, "absent") == Totals(0, 0, 0)


@pytest.mark.parametrize("operation", ["record", "restore", "clear"])
def test_cooldown_writes_do_not_create_a_missing_session(backend, scope_a, operation):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    if operation == "record":
        repo.record_compression_cooldown("absent", STAMP, "failure")
    elif operation == "restore":
        repo.restore_compression_cooldown("absent", CooldownSnapshot(True, STAMP, "failure"))
    else:
        repo.clear_compression_cooldown("absent")
    assert store_a.sessions.get("absent") is None
    assert repo.read_compression_cooldown_snapshot("absent") == CooldownSnapshot(False, None, None)
