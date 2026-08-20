"""Row-cap pruning must never delete undelivered obligations.

The ledger exists so a reply that was owed but not yet delivered survives a
gateway crash. The row-cap branch of ``_prune`` ordered candidates with
settled states first, but did not FILTER on them — once the settled rows were
exhausted, the excess deletions consumed pending/attempting/failed rows, i.e.
exactly the output the ledger is supposed to protect.
"""
import sqlite3

import pytest

from gateway import delivery_ledger as dl


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


def _record(n: int, prefix: str) -> list[str]:
    ids = []
    for i in range(n):
        oid = f"{prefix}-{i:04d}"
        dl.record_obligation(
            obligation_id=oid,
            session_key=f"sess-{prefix}",
            platform="telegram",
            chat_id="chat-1",
            thread_id=None,
            content=f"reply {oid}",
        )
        ids.append(oid)
    return ids


def _states() -> dict[str, int]:
    conn = sqlite3.connect(dl._db_path())
    try:
        rows = conn.execute(
            "SELECT state, COUNT(*) FROM delivery_obligations GROUP BY state"
        ).fetchall()
        return {state: count for state, count in rows}
    finally:
        conn.close()


def test_row_cap_prune_never_deletes_undelivered(monkeypatch):
    monkeypatch.setattr(dl, "_MAX_ROWS", 10)

    delivered = _record(8, "done")
    for oid in delivered:
        dl.mark_delivered(oid)
    pending = _record(12, "owed")  # every record_obligation() call prunes

    states = _states()
    # All 12 undelivered obligations survive, even though the table exceeds
    # the cap after the 8 settled rows are gone.
    assert states.get("pending", 0) == len(pending)
    # Settled rows were consumed to honour the cap as far as possible.
    assert states.get("delivered", 0) <= 10


def test_row_cap_prune_still_removes_settled_excess(monkeypatch):
    monkeypatch.setattr(dl, "_MAX_ROWS", 5)

    settled = _record(9, "done")
    for oid in settled:
        dl.mark_delivered(oid)
    _record(1, "owed")  # triggers a prune with 9 settled + 1 pending

    states = _states()
    assert states.get("pending", 0) == 1
    # 10 total - cap 5 = 5 excess, all taken from the settled rows.
    assert states.get("delivered", 0) == 4
