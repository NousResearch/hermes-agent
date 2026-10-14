"""Multiplex cron resolves the primary gateway routing config once per cycle."""

from contextlib import contextmanager
import json
import threading
from unittest.mock import patch


def _write_jobs(home, jobs):
    cron_dir = home / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    (cron_dir / "jobs.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")


def test_multiplex_resolves_primary_routes_once_per_cycle(tmp_path, monkeypatch):
    from cron.scheduler_provider import InProcessCronScheduler

    primary, satellite_a, satellite_b = (
        tmp_path / name for name in ("primary", "satellite-a", "satellite-b")
    )
    for home in (primary, satellite_a, satellite_b):
        _write_jobs(home, [{"id": "scheduled"}])

    config_reads = []
    ticked = []
    stop = threading.Event()

    def fake_tick(*_args, **_kwargs):
        ticked.append(True)
        if len(ticked) == 3:
            stop.set()
        return 0

    monkeypatch.setattr(InProcessCronScheduler, "recover_interrupted", lambda self: 0)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: primary)
    with patch("cron.scheduler.tick", side_effect=fake_tick), \
         patch("gateway.config_loader.read_yaml_layers",
               side_effect=lambda _home: config_reads.append(True) or {"profile_routes": []}), \
         patch("cron.jobs.record_ticker_heartbeat", lambda **_kwargs: None):
        InProcessCronScheduler().start(
            stop,
            interval=0,
            profile_homes=[("primary", primary), ("satellite-a", satellite_a),
                           ("satellite-b", satellite_b)],
            adapters={"telegram": object()},
            default_profile="primary",
        )

    assert len(ticked) == 3
    assert len(config_reads) == 1
