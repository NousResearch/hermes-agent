"""andrexibiza 26: the session counts that page a listing count a local reset lineage once, the
same way list_sessions_rich lists it (an earlier reset segment is not a second conversation)."""
import pytest

from tests.gateway import test_local_reset_lineage_one_conversation as one_conversation
from persistence.conversation.sessions import SessionCountQuery, SessionQuery, SessionWindow
from tests.fixtures.storage_holder import raw_holder

reset_lineage = one_conversation.reset_lineage  # the shared fixture


@pytest.mark.asyncio
async def test_reset_lineage_counts_as_one_conversation(reset_lineage):
    sessions = reset_lineage.authority.persistence.sessions
    listed = [entry.as_list_row() for entry in sessions.list_sessions(SessionQuery(), SessionWindow(limit=50))]
    assert [r['id'] for r in listed] == [reset_lineage.s1]
    assert sessions.count_sessions(SessionCountQuery(exclude_children=True)) == len(listed) == 1
    assert sessions.count_by_source(exclude_children=True) == {'gui': 1}
    # The raw row count (exclude_children=False) still sees both physical rows.
    assert sessions.count_sessions() == 2


@pytest.mark.asyncio
async def test_pinned_backfill_lists_a_reset_lineage_once(reset_lineage):
    db = reset_lineage.authority.persistence
    raw_holder(db)._execute_write(lambda c: c.execute('UPDATE sessions SET pinned=1 WHERE id IN (?,?)',
                                                      (reset_lineage.s0, reset_lineage.s1)))
    listed = db.sessions.list_sessions(SessionQuery(include_pinned=True), SessionWindow(limit=50))
    assert [entry.as_list_row()['id'] for entry in listed] == [reset_lineage.s1]
