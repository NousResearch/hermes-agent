"""Contract cases: ``persistence.conversation.messages.MessageRepository.append_batch`` carries the row state (stored timestamp, ownership snapshot, canonical row) and treats ``NewMessage.db_row_snapshot`` as a compare-and-swap version.

The callers: ``agent/transcript_repair.py`` ``resolve_and_repair_transcript_batch`` (snapshot match rewrites the owned payload, mismatch adopts the durable row, metadata other writers stamp is outside the version) and ``sync_flushed_message_markers`` (the metadata-only / content-only / full-row modes). Snapshot bytes are backend-specific, so every case compares snapshots only with snapshots from the same backend.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from persistence.core.identity import SessionId
from persistence.conversation.messages import NewMessage, RowDisposition, TranscriptRead, TranscriptView, Watermark
from persistence.conversation.sessions import NewSession
from tests.contracts.conversation.test_messages import Totals, totals

ROOT = SessionId("snapshot-root")


@pytest.fixture
def backend(backend, scope_a, scope_b):
    for scope in (scope_a, scope_b):
        backend.open(scope).sessions.create(NewSession(session_id=ROOT, source="cli"))
    return backend


def ts(i: int) -> float:
    return datetime(2026, 9, 28, 10, 0, i, tzinfo=UTC).timestamp()


def msg(content: str, i: int = 0, role: str = "assistant", **kwargs: object) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=ts(i), **kwargs)  # type: ignore[arg-type]


def seeded(repo, **kwargs: object):
    first = repo.append_batch(ROOT, [msg("ask", 0, role="user"), msg("draft", 1, **kwargs)])
    return first.outcomes[1]


def test_matching_snapshot_rewrites_the_owned_payload_in_place(backend, scope_a):
    repo = backend.open(scope_a).messages
    mine = seeded(repo)
    counted = totals(backend, scope_a, ROOT)
    result = repo.append_batch(ROOT, [msg("final", 9, repair_target=mine.id, db_row_snapshot=mine.db_row_snapshot)])
    (outcome,) = result.outcomes
    assert (outcome.id, outcome.disposition, outcome.canonical_row) == (mine.id, RowDisposition.REPAIRED, None)
    assert outcome.timestamp == mine.timestamp == ts(1)
    assert outcome.db_row_snapshot != mine.db_row_snapshot
    assert result.inserted == 0
    assert [r.content for r in repo.read_transcript(ROOT)] == ["ask", "final"]
    assert totals(backend, scope_a, ROOT) == counted
    again = repo.append_batch(ROOT, [msg("final", 9, repair_target=mine.id, db_row_snapshot=outcome.db_row_snapshot)])
    assert again.outcomes[0].db_row_snapshot == outcome.db_row_snapshot
    assert again.outcomes[0].disposition is RowDisposition.REPAIRED


def test_stale_snapshot_adopts_the_winner_and_writes_nothing(backend, scope_a):
    repo = backend.open(scope_a).messages
    mine = seeded(repo)
    winner = (repo.append_batch(ROOT, [msg("winner", 1, repair_target=mine.id, db_row_snapshot=mine.db_row_snapshot)])).outcomes[0]
    late = (repo.append_batch(ROOT, [msg("late", 1, repair_target=mine.id, db_row_snapshot=mine.db_row_snapshot)])).outcomes[0]
    assert (late.id, late.disposition, late.db_row_snapshot) == (mine.id, RowDisposition.ADOPTED, winner.db_row_snapshot)
    assert late.canonical_row is not None
    assert (late.canonical_row["role"], late.canonical_row["content"]) == ("assistant", "winner")
    assert not {"_content_only", "_metadata_only"} & set(late.canonical_row)
    assert [r.content for r in repo.read_transcript(ROOT)] == ["ask", "winner"]


def test_matched_row_lends_only_metadata_the_caller_lacks(backend, scope_a):
    repo = backend.open(scope_a).messages
    mine = seeded(repo, display_kind="hidden")
    lent = (repo.append_batch(ROOT, [msg("final", 1, repair_target=mine.id, db_row_snapshot=mine.db_row_snapshot)])).outcomes[0]
    assert lent.disposition is RowDisposition.REPAIRED
    assert lent.canonical_row == {"display_kind": "hidden", "_metadata_only": True}
    kept = (repo.append_batch(ROOT, [msg("final", 1, repair_target=mine.id, db_row_snapshot=lent.db_row_snapshot, display_kind="note")])).outcomes[0]
    assert (kept.disposition, kept.canonical_row) == (RowDisposition.REPAIRED, None)
    rows = repo.read_transcript(ROOT)
    assert [(r.content, r.display_kind) for r in rows] == [("ask", None), ("final", "hidden")]


def test_a_reaction_does_not_move_the_ownership_version(backend, scope_a):
    repo = backend.open(scope_a).messages
    mine = seeded(repo)
    assert repo.set_reaction(ROOT, mine.id, "👍", author="user") is not None
    result = (repo.append_batch(ROOT, [msg("final", 1, repair_target=mine.id, db_row_snapshot=mine.db_row_snapshot)])).outcomes[0]
    assert result.disposition is RowDisposition.REPAIRED
    assert [r.content for r in repo.read_transcript(ROOT)] == ["ask", "final"]


def compacted_clone(repo, mine):
    """Compact with nothing seen: the watermark tail clones every active row, ``mine`` included, under fresh ids after the summary."""
    repo.compact(ROOT, [msg("summary", 5)], watermark=Watermark(None))
    clone = (repo.read_transcript(ROOT))[-1]
    assert clone.id != mine.id and clone.content == "draft"
    return clone


def test_a_repair_through_an_archived_row_lands_on_its_exact_clone(backend, scope_a):
    repo = backend.open(scope_a).messages
    mine = seeded(repo)
    assert repo.set_reaction(ROOT, mine.id, "👍", author="user") is not None
    clone = compacted_clone(repo, mine)
    # CONTRACT: the clone carries the original's reactions.
    page = repo.read_transcript(ROOT, TranscriptRead(view=TranscriptView.ACTIVE))
    assert [(r["emoji"], r["author"]) for r in page[-1].display_metadata["reactions"]] == [("👍", "user")]
    outcome = (repo.append_batch(ROOT, [msg("final", 1, repair_target=mine.id, db_row_snapshot=mine.db_row_snapshot)])).outcomes[0]
    # CONTRACT: the compaction clone copies every durable field, reactions included, so the repair addressed to the archived original rewrites the one active clone.
    assert (outcome.id, outcome.disposition) == (clone.id, RowDisposition.REPAIRED)
    assert [r.content for r in repo.read_transcript(ROOT)] == ["summary", "ask", "final"]


def test_a_reaction_on_the_clone_breaks_exact_clone_identity(backend, scope_a):
    repo = backend.open(scope_a).messages
    mine = seeded(repo)
    clone = compacted_clone(repo, mine)
    assert repo.set_reaction(ROOT, clone.id, "👍", author="user") is not None
    outcome = (repo.append_batch(ROOT, [msg("final", 1, repair_target=mine.id, db_row_snapshot=mine.db_row_snapshot)])).outcomes[0]
    # CONTRACT: the exact-clone match compares display_metadata, where reactions live: a clone whose reactions diverged is no longer the exact clone, so the repair falls back to the addressed row and the clone keeps its content. The ownership snapshot is a separate comparison and still matches.
    assert (outcome.id, outcome.disposition) == (mine.id, RowDisposition.REPAIRED)
    assert [r.content for r in repo.read_transcript(ROOT)] == ["summary", "ask", "draft"]


def test_snapshotless_assistant_over_a_filled_row_adopts_content_only(backend, scope_a):
    repo = backend.open(scope_a).messages
    mine = seeded(repo)
    adopted = (repo.append_batch(ROOT, [msg("loser", 5, repair_target=mine.id)])).outcomes[0]
    assert (adopted.disposition, adopted.canonical_row) == (RowDisposition.ADOPTED, {"content": "draft", "_content_only": True})
    assert (adopted.timestamp, adopted.db_row_snapshot) == (mine.timestamp, mine.db_row_snapshot)
    assert totals(backend, scope_a, ROOT) == Totals(2, 0, 0)


def test_outcomes_are_isolated_per_profile(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_b).messages
    mine = seeded(backend.open(scope_a).messages)
    other = (repo.append_batch(ROOT, [msg("theirs", 1, repair_target=mine.id, db_row_snapshot=mine.db_row_snapshot)])).outcomes[0]
    assert other.disposition is RowDisposition.APPENDED
    assert [r.content for r in store_a.messages.read_transcript(ROOT)] == ["ask", "draft"]
