"""Background-review notices wait for main-response delivery and never outlive their turn."""

from types import SimpleNamespace

from gateway.config import Platform
from gateway.run_turn_runner import TurnRunner


def _review_callbacks():
    current = [True]
    ledger = {}
    metadata = {"thread_id": "topic-1", "_feishu_topic_delivery": ledger}
    ctx = SimpleNamespace(
        _status_adapter=object(),
        _run_still_current=lambda: current[0],
        _status_thread_metadata=metadata,
        source=SimpleNamespace(platform=Platform.DISCORD),
    )
    runner = TurnRunner(SimpleNamespace(), ctx)
    delivered = []
    runner._send_status_text = lambda text, meta, log: delivered.append((text, meta))
    send, release = runner._make_bg_review_callbacks()
    return send, release, current, delivered, metadata


def test_reviews_wait_for_delivery_then_send_once_with_interim_metadata():
    send, release, _current, delivered, original = _review_callbacks()
    send("Memory updated")
    send("Review ready")
    assert delivered == []

    release()
    release()
    send("Later review")

    assert [text for text, _metadata in delivered] == ["Memory updated", "Review ready", "Later review"]
    for _text, metadata in delivered:
        assert metadata["_interim_send"] is True
        assert metadata["non_conversational"] is True
        assert metadata["thread_id"] == "topic-1"
        assert metadata["_feishu_topic_delivery"] is original["_feishu_topic_delivery"]
    assert "_interim_send" not in original
    assert "non_conversational" not in original


def test_stale_turn_drops_buffered_and_later_review_notices():
    send, release, current, delivered, _metadata = _review_callbacks()
    send("Buffered review")
    current[0] = False
    release()
    send("Late review")
    assert delivered == []
