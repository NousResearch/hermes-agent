"""#105535 — the projected lineage cost survives a chain member vanishing mid-read.

``list_sessions_rich`` resolves a compressed conversation's spend in two reads: the chain walk
(``get_compression_chain``) and then ``_lineage_cost_by_root``'s single chunked SUM. Each opens
its own read connection and a segment can be deleted between them — ``delete_session`` orphans
children while the Desktop UI deletes rows out from under a live sidebar listing — so an id the
walk resolved may be absent from the sum query's rows. The projection must price the members
that remain instead of raising out of the sidebar read.

The case comes from the whole-lineage basis being summed per member: unlike the old frozen root
snapshot, the projected figure is assembled from rows that the same request just enumerated.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    handle = SessionDB(tmp_path / "state.db")
    try:
        yield handle
    finally:
        handle.close()


def _priced_chain(db: SessionDB):
    """root ($0.50, ended by compression) -> mid ($1.00, ended by compression) -> tip ($0.25)."""
    db.create_session("root", source="cli")
    db.create_session("mid", source="cli", parent_session_id="root")
    db.create_session("tip", source="cli", parent_session_id="mid")
    for sid in ("root", "mid", "tip"):
        db.append_message(sid, "user", f"message in {sid}")
    for sid, cost, ended in (("root", 0.50, True), ("mid", 1.00, True), ("tip", 0.25, False)):
        end_clause = ", end_reason = 'compression'" if ended else ""
        db._write_rowcount(
            f"UPDATE sessions SET estimated_cost_usd = ?{end_clause} WHERE id = ?",
            (cost, sid),
        )


def test_projection_prices_remaining_members_when_one_is_deleted_mid_read(db, monkeypatch):
    """A member the chain walk resolved but the cost query no longer returns contributes 0.

    Simulated at the seam that actually races: the walk is untouched, the SUM's rows come back
    one member short.
    """
    _priced_chain(db)
    real_read_all = SessionDB._read_all

    def racing_read_all(self, sql, params=()):
        rows = list(real_read_all(self, sql, params))
        if " AS cost" in sql:  # the lineage sum, run after the walk already resolved the chain
            return [row for row in rows if row["id"] != "mid"]
        return rows

    monkeypatch.setattr(SessionDB, "_read_all", racing_read_all)

    (row,) = db.list_sessions_rich(source="cli", limit=20, min_message_count=1, compact_rows=True)

    # The projection still ran (tip identity surfaced onto the root's row) ...
    assert row["id"] == "tip"
    assert row["_lineage_root_id"] == "root"
    # ... and the vanished segment is priced at 0 rather than aborting the listing: root + tip.
    assert row["estimated_cost_usd"] == pytest.approx(0.75)
