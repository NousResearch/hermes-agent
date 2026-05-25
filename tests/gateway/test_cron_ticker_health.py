import logging
import threading


def test_cron_ticker_records_failure_and_recovery(monkeypatch, caplog):
    from gateway.run import _start_cron_ticker

    stop_event = threading.Event()
    statuses = []
    calls = {"count": 0}

    def fake_tick(*, verbose, adapters, loop):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("too many open files")
        stop_event.set()

    monkeypatch.setattr("cron.scheduler.tick", fake_tick)
    monkeypatch.setattr(
        "gateway.status.write_runtime_status",
        lambda **kwargs: statuses.append(kwargs),
    )

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _start_cron_ticker(stop_event, interval=0)

    assert statuses[0]["cron"]["state"] == "failing"
    assert statuses[0]["cron"]["consecutive_failures"] == 1
    assert "OSError: too many open files" == statuses[0]["cron"]["last_error"]
    assert statuses[1]["cron"]["state"] == "healthy"
    assert statuses[1]["cron"]["consecutive_failures"] == 0
    assert "Cron tick failed (1 consecutive failure)" in caplog.text
    assert "Cron tick recovered after 1 consecutive failure" in caplog.text
