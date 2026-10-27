"""andrexibiza 26: the session counts that page a listing count a local reset lineage once, the
same way list_sessions_rich lists it (an earlier reset segment is not a second conversation)."""
import pytest

from tests.gateway import test_local_reset_lineage_one_conversation as one_conversation

reset_lineage = one_conversation.reset_lineage  # the shared fixture


@pytest.mark.asyncio
async def test_reset_lineage_counts_as_one_conversation(reset_lineage):
    db = reset_lineage.authority.db
    listed = db.list_sessions_rich(limit=50)
    assert [r['id'] for r in listed] == [reset_lineage.s1]
    assert db.session_count(exclude_children=True) == len(listed) == 1
    assert db.session_count_by_source(exclude_children=True) == {'gui': 1}
    # The raw row count (exclude_children=False) still sees both physical rows.
    assert db.session_count() == 2


@pytest.mark.asyncio
async def test_pinned_backfill_lists_a_reset_lineage_once(reset_lineage):
    db = reset_lineage.authority.db
    db._execute_write(lambda c: c.execute('UPDATE sessions SET pinned=1 WHERE id IN (?,?)',
                                          (reset_lineage.s0, reset_lineage.s1)))
    assert [r['id'] for r in db.list_sessions_rich(limit=50, include_pinned=True)] == [reset_lineage.s1]
