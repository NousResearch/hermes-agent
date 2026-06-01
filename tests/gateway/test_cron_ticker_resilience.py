import sys
import types

import gateway.run as gateway_run


class _CronTickFatal(BaseException):
    pass


def test_cron_ticker_survives_baseexception(monkeypatch, caplog):
    caplog.set_level("WARNING", logger="gateway.run")

    stop_event = gateway_run.threading.Event()
    calls = {"n": 0}

    def _fake_tick(verbose=False, adapters=None, loop=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _CronTickFatal("boom")
        stop_event.set()

    monkeypatch.setitem(sys.modules, "cron.scheduler", types.SimpleNamespace(tick=_fake_tick))
    monkeypatch.setitem(
        sys.modules,
        "gateway.platforms.base",
        types.SimpleNamespace(
            cleanup_image_cache=lambda max_age_hours=24: 0,
            cleanup_document_cache=lambda max_age_hours=24: 0,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.debug",
        types.SimpleNamespace(_sweep_expired_pastes=lambda: (0, 0)),
    )

    gateway_run._start_cron_ticker(stop_event=stop_event, interval=0)

    assert calls["n"] == 2
    assert "Cron tick fatal error:" in caplog.text
