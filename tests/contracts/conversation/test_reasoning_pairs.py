"""Reasoning pairs have one replay meaning on every persistence backend (#125273)."""

from dataclasses import replace

import pytest

from persistence.conversation.messages import ConversationRequest, NewMessage, RowDisposition
from persistence.conversation.sessions import NewSession
from persistence.core.identity import SessionId
from persistence import RowVisibility
from tests.contracts._reference.conversation.messages_compact import _replays_as
from tests.contracts._reference.conversation.messages_writes import _repair_target, _same_as_stored
from tests.contracts._reference.conversation.transcript_store import _StoredRow, _Transcript, _row_snapshot

SID = SessionId("reasoning-pair")
TEXT = "compare both branches"


@pytest.mark.parametrize("reasoning", [None, TEXT])
def test_reasoning_content_only_and_equal_pairs_replay_and_dedupe_the_same(backend, scope_a, reasoning):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SID, source="cli"))
    store.messages.append(SID, NewMessage(role="user", content="ask"))
    row_id = store.messages.append(SID, NewMessage(role="assistant", content="answer", reasoning=reasoning, reasoning_content=TEXT))
    row = store.messages.read_transcript(SID)[-1]
    assert (row.reasoning, row.reasoning_content) == (TEXT, TEXT)
    record = store.messages.read_around(SID, row_id).window[-1]
    assert (record.reasoning, record.reasoning_content) == (TEXT, TEXT)
    conversation = store.messages.read_conversation(SID, ConversationRequest())
    message = conversation[-1].as_openai_message()
    assert (message["reasoning"], message["reasoning_content"]) == (TEXT, TEXT)
    again = NewMessage(role="assistant", content="answer", reasoning=TEXT, reasoning_content=TEXT,
                       message_uid=row.message_uid, restored_from_store=True)
    result = store.messages.append_batch(SID, [again])
    assert result.inserted == 0
    assert result.outcomes[0].id == row_id
    original = NewMessage(role="assistant", content="answer", reasoning=TEXT, reasoning_content=TEXT,
                          repair_target=row_id, db_row_snapshot=result.outcomes[0].db_row_snapshot)
    winner = store.messages.append_batch(SID, [replace(original, content="winner")]).outcomes[0]
    adopted = store.messages.append_batch(SID, [replace(original, content="late")]).outcomes[0]
    assert adopted.disposition is RowDisposition.ADOPTED
    assert adopted.db_row_snapshot == winner.db_row_snapshot
    assert adopted.canonical_row["reasoning"] == adopted.canonical_row["reasoning_content"] == TEXT


def test_reference_legacy_rows_share_versions_dedupe_and_clone_identity():
    legacy = NewMessage(role="assistant", content="answer", timestamp=1.0, reasoning=TEXT, reasoning_content=TEXT)
    folded = replace(legacy, reasoning=None)
    assert _row_snapshot(legacy) == _row_snapshot(folded)
    assert _same_as_stored(legacy, folded) and _same_as_stored(folded, legacy)
    original = _StoredRow(1, legacy, RowVisibility.REWOUND)
    clone = _StoredRow(2, folded)
    transcript = _Transcript(rows=[original, clone])
    assert _repair_target(transcript, replace(legacy, repair_target=1)) is clone
    clone.message = replace(folded, reasoning="different")
    assert _repair_target(transcript, replace(legacy, repair_target=1)) is original
    assert _row_snapshot(legacy) != _row_snapshot(clone.message)


def test_reference_coverage_compares_restored_reasoning_not_a_null_wildcard():
    held = NewMessage(role="assistant", content="answer", reasoning=TEXT, reasoning_content=TEXT)
    row = _StoredRow(1, replace(held, reasoning="different"))
    assert not _replays_as(row, held, True)
    assert _replays_as(row, replace(held, reasoning="different"), True)
    assert _replays_as(_StoredRow(2, held), replace(held, reasoning=None), True)
