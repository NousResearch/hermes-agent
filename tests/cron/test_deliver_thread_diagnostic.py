"""Thread-loss delivery diagnostic must not warn on deliberate fan-out.

A job created inside a thread stamps ``origin.thread_id``. Pointing
``deliver`` at a DIFFERENT channel is an ordinary fan-out configuration —
that target never had the origin's thread lane, so nothing was "lost" and
the diagnostic must stay at debug. Only the origin conversation itself
arriving without its thread lane is a routing fault worth a WARNING
(#89650: the old unconditional warning fired 10-24x/day on correct setups,
burying real thread-routing problems).
"""

import logging

import pytest

from cron import scheduler


@pytest.fixture
def _stub_delivery(monkeypatch):
    """Stub everything past the diagnostic so _deliver_result exits early."""

    class _StubConfig:
        platforms = {}

    monkeypatch.setattr(
        "gateway.config.load_gateway_config", lambda: _StubConfig()
    )
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport",
        lambda platform, config, adapters: None,
    )
    monkeypatch.setattr(scheduler, "load_config", lambda: {})


def _job(origin_thread="t9", deliver="discord:456"):
    return {
        "id": "job-1",
        "name": "job-1",
        "deliver": deliver,
        "origin": {
            "platform": "discord",
            "chat_id": "123",
            "thread_id": origin_thread,
        },
    }


def _run(monkeypatch, caplog, target):
    monkeypatch.setattr(
        scheduler, "_resolve_delivery_targets", lambda job: [target]
    )
    with caplog.at_level(logging.DEBUG, logger=scheduler.logger.name):
        scheduler._deliver_result(_job(), "content")
    return [
        r for r in caplog.records if "delivery target lost it" in r.getMessage()
    ]


def test_fanout_target_losing_thread_is_debug(monkeypatch, caplog, _stub_delivery):
    """A target for another chat never had the origin thread: debug, not warning."""
    records = _run(
        monkeypatch, caplog,
        {"platform": "discord", "chat_id": "456", "thread_id": None},
    )
    assert [r.levelno for r in records] == [logging.DEBUG]


def test_origin_chat_losing_thread_still_warns(monkeypatch, caplog, _stub_delivery):
    """The origin conversation arriving without its thread lane is a real fault."""
    records = _run(
        monkeypatch, caplog,
        {"platform": "discord", "chat_id": "123", "thread_id": None},
    )
    assert [r.levelno for r in records] == [logging.WARNING]
