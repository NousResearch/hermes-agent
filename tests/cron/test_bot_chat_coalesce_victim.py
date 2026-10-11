"""``bot_chat_coalesce: latest`` must never retire loss-of-signal receipts (#133349)."""
from cron import bot_chat_delivery as queue


def test_coalesce_latest_never_suppresses_failure_or_degraded_receipts(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {"id": "snap", "bot_chat_coalesce": "latest"}
    queue.defer("f" * 64, job, "failure notice", "", tmp_path, for_failure=True)
    queue.defer("d" * 64, job, "degraded marker", "", tmp_path, degraded=True)
    fresh = queue.defer("s" * 64, job, "fresh snapshot", "", tmp_path)
    # Failure notices and degraded markers are loss-of-signal receipts: a later snapshot
    # must not retire them, or one failed run's alert is swallowed by the next success.
    assert queue.read_pending("f" * 64)["status"] == "queued"
    assert queue.read_pending("d" * 64)["status"] == "queued"
    assert fresh["status"] == "queued"
