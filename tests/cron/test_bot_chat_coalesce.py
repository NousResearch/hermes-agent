"""``bot_chat_coalesce: latest`` lets a fresh snapshot retire the same job's older
still-queued deferred receipts (#133349) without touching anything else."""
from cron import bot_chat_delivery as queue
from cron import scheduler_delivery as delivery


def test_coalesce_latest_suppresses_older_queued_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {"id": "snap", "bot_chat_coalesce": "latest"}
    queue.defer("a" * 64, job, "older snapshot", "", tmp_path)
    newer = queue.defer("b" * 64, job, "newer snapshot", "", tmp_path)
    superseded = queue.read_pending("a" * 64)
    assert superseded["status"] == "suppressed"
    assert superseded["error"] == f"superseded by receipt seq {newer['sequence']}"
    assert newer["status"] == "queued"
    seen = []
    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", lambda j, c, p, **kw: seen.append(c))
    queue.drain()
    assert seen == ["newer snapshot"]
    assert queue.read_pending("b" * 64)["status"] == "settled"
    assert queue.read_pending("a" * 64)["status"] == "suppressed"


def test_coalesce_latest_leaves_other_receipts_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    other_home = tmp_path / "elsewhere"
    other_home.mkdir()
    snap = {"id": "snap", "bot_chat_coalesce": "latest"}
    queue.defer("c" * 64, {"id": "plain"}, "no opt-in", "", tmp_path)
    queue.defer("f" * 64, {"id": "plain"}, "no opt-in later", "", tmp_path)
    queue.defer("d" * 64, snap, "elsewhere snapshot", "", other_home)
    queue.defer("g" * 64, snap, "own-home snapshot", "", tmp_path)
    queue.defer("h" * 64, snap, "degraded marker", "", tmp_path, degraded=True)
    # Only the same (job id, home) queued snapshot lane coalesces: the plain job keeps
    # both receipts, the other home and the degraded marker stay queued.
    for key in "cdfgh":
        assert queue.read_pending(key * 64)["status"] == "queued"
