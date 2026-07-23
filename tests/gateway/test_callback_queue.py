"""Regression coverage for bounded gateway callback backlogs."""

import threading

from gateway.run import _BoundedCallbackQueue


def test_callback_queue_rejects_saturation_without_blocking():
    callback_queue = _BoundedCallbackQueue(maxsize=2)

    assert callback_queue.offer("first") is True
    assert callback_queue.offer("second") is True

    result = []
    producer = threading.Thread(
        target=lambda: result.append(callback_queue.offer("overflow")),
        daemon=True,
    )
    producer.start()
    producer.join(timeout=0.5)

    assert producer.is_alive() is False
    assert result == [False]
    assert callback_queue.qsize() == 2
    assert callback_queue.dropped == 1
    assert callback_queue.get_nowait() == "first"
    assert callback_queue.get_nowait() == "second"


def test_callback_queue_keeps_accepting_after_consumer_drains_space():
    callback_queue = _BoundedCallbackQueue(maxsize=1)

    assert callback_queue.offer("first") is True
    assert callback_queue.offer("rejected") is False
    assert callback_queue.get_nowait() == "first"
    assert callback_queue.offer("next") is True

    assert callback_queue.get_nowait() == "next"
    assert callback_queue.dropped == 1


def test_callback_queue_prioritizes_control_event_over_stale_telemetry():
    callback_queue = _BoundedCallbackQueue(maxsize=2)
    callback_queue.offer("oldest")
    callback_queue.offer("newer")

    assert callback_queue.offer_latest(("__reset__",)) is True

    assert callback_queue.get_nowait() == "newer"
    assert callback_queue.get_nowait() == ("__reset__",)
    assert callback_queue.dropped == 1
