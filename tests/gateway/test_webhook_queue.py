import time

from gateway import webhook_queue


def test_webhook_queue_enqueue_claim_done(tmp_path, monkeypatch):
    monkeypatch.setattr(webhook_queue, "get_hermes_home", lambda: tmp_path)
    record = webhook_queue.make_record(
        route_name="fizzy-kira-os",
        delivery_id="delivery-1",
        event_type="card_published",
        payload={"card": {"id": 208}},
        prompt="process card 208",
        deliver_config={"deliver": "slack", "deliver_extra": {}},
    )

    queue_id = webhook_queue.enqueue(record)
    duplicate_id = webhook_queue.enqueue(dict(record))

    assert duplicate_id == queue_id
    assert webhook_queue.stats()["pending"] == 1

    claimed = webhook_queue.claim_due(now=time.time())
    assert len(claimed) == 1
    assert claimed[0]["id"] == queue_id
    assert claimed[0]["attempts"] == 1
    assert webhook_queue.stats()["inflight"] == 1

    webhook_queue.mark_done(queue_id)
    assert webhook_queue.stats()["total"] == 0


def test_webhook_queue_retry_and_reclaim(tmp_path, monkeypatch):
    monkeypatch.setattr(webhook_queue, "get_hermes_home", lambda: tmp_path)
    record = webhook_queue.make_record(
        route_name="fizzy-kira-os",
        delivery_id="delivery-2",
        event_type="card_published",
        payload={"card": {"id": 208}},
        prompt="process card 208",
        deliver_config={"deliver": "slack", "deliver_extra": {}},
    )
    record["queued_at"] = 1000.0
    record["next_attempt_at"] = 1000.0
    queue_id = webhook_queue.enqueue(record)
    first = webhook_queue.claim_due(now=1000.0)
    assert len(first) == 1

    webhook_queue.mark_retry(queue_id, "gateway draining", retry_delay_seconds=60)
    assert webhook_queue.claim_due(now=1001.0) == []

    second = webhook_queue.claim_due(now=time.time() + 120)
    assert len(second) == 1
    assert second[0]["attempts"] == 2


def test_webhook_queue_mark_inflight(tmp_path, monkeypatch):
    monkeypatch.setattr(webhook_queue, "get_hermes_home", lambda: tmp_path)
    record = webhook_queue.make_record(
        route_name="notion-sync",
        delivery_id="delivery-3",
        event_type="comment.created",
        payload={"id": "delivery-3"},
        prompt="process notion comment",
        deliver_config={"deliver": "log", "deliver_extra": {}},
    )
    queue_id = webhook_queue.enqueue(record)

    webhook_queue.mark_inflight(queue_id, now=2000.0)

    stats = webhook_queue.stats()
    assert stats["inflight"] == 1
    claimed = webhook_queue.claim_due(now=2001.0)
    assert claimed == []
    stale = webhook_queue.claim_due(now=2000.0 + webhook_queue.DEFAULT_STALE_INFLIGHT_SECONDS + 1)
    assert len(stale) == 1
    assert stale[0]["id"] == queue_id
    assert stale[0]["attempts"] == 2
