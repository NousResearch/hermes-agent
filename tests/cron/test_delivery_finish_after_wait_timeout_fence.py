"""A delivery this process sent stays recordable after a wait-timeout fences the row.

The worker's wait deadline is a wall clock and the send is not: the fence can move the row from
``delivering`` to ``unknown`` after the send has already gone out. The process that owns the row
then holds the real outcome, and refusing to write it leaves the operator with "outcome is unknown
and was not retried" for a delivery whose result is in hand. Ownership still decides: a row owned
by another process is never rewritten.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def queue(tmp_path, monkeypatch):
    from cron import delivery_queue

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    return delivery_queue


def _claim(queue):
    queue.enqueue("exec-1", {"id": "job"}, "hello")
    claimed = queue.claim_next()
    assert claimed is not None and claimed["execution_id"] == "exec-1"


def test_finish_records_the_outcome_after_a_wait_timeout_fence(queue):
    _claim(queue)
    with queue._transaction() as conn:
        conn.execute(
            "UPDATE deliveries SET status='unknown', error=? WHERE execution_id='exec-1'",
            ("delivery outcome unknown: worker deadline passed",),
        )

    assert queue._finish("exec-1", error=None) is True
    assert queue.get_status("exec-1")["status"] == "delivered"


def test_finish_still_refuses_a_row_another_process_owns(queue):
    _claim(queue)
    with queue._transaction() as conn:
        conn.execute(
            "UPDATE deliveries SET status='unknown', owner_process_id='another-process' "
            "WHERE execution_id='exec-1'"
        )

    assert queue._finish("exec-1", error=None) is False
    assert queue.get_status("exec-1")["status"] == "unknown"
