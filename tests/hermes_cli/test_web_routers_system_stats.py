"""``GET /api/system/stats`` contract for the desktop statusbar's System Resources item.

The desktop reads ``hostname``, ``cpu_percent``, ``memory`` and ``disk`` from this route on the
ACTIVE backend (an SSH remote included), polling while the item is shown. The route samples
``psutil.cpu_percent(interval=0.1)``, which sleeps 100ms: it must stay a plain ``def`` so FastAPI
runs it in the threadpool instead of holding the event loop (and every chat socket on it).
"""

from __future__ import annotations

import inspect


def test_system_stats_route_does_not_run_on_the_event_loop():
    from hermes_cli.web_routers import status

    assert not inspect.iscoroutinefunction(status.get_system_stats), (
        "get_system_stats sleeps 100ms in psutil.cpu_percent; as `async def` it blocks the loop"
    )


def test_system_stats_reports_the_fields_the_statusbar_reads(monkeypatch, tmp_path):
    from hermes_cli.web_routers import status

    monkeypatch.setattr(status, "get_hermes_home", lambda: tmp_path)

    info = status.get_system_stats()

    assert isinstance(info["hostname"], str)
    assert info["cpu_count"] == __import__("os").cpu_count()
    if info.get("psutil"):
        assert 0 <= info["cpu_percent"] <= 100
        assert info["memory"]["total"] >= info["memory"]["available"] >= 0
        assert info["disk"]["total"] >= info["disk"]["used"] >= 0
