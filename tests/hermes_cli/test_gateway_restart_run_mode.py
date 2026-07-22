"""Regression: `hermes gateway restart` must relaunch a manually-run gateway.

Bug: ``restart()`` did ``stop()`` then ``start()``, and ``start()``'s
"service not installed" branch prompts to install and returns WITHOUT starting.
So restarting a manually-run gateway (no scheduled task / startup entry) left it
DOWN -- a silent outage on every manual-gateway restart.

``restart()`` must preserve the run mode: relaunch a manual gateway directly via
``_spawn_detached()``, only routing through ``start()`` when a service/startup
entry is installed.
"""
import hermes_cli.gateway_windows as gw


def _neutralize(monkeypatch, calls):
    monkeypatch.setattr(gw, "_assert_windows", lambda: None)
    monkeypatch.setattr(gw, "stop", lambda: None)
    monkeypatch.setattr(gw, "_wait_for_gateway_absent", lambda **k: True)
    monkeypatch.setattr(gw, "_wait_for_gateway_ready", lambda **k: True)
    monkeypatch.setattr(gw, "_report_gateway_start", lambda *a, **k: None)
    monkeypatch.setattr(gw.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_spawn_detached", lambda: calls.__setitem__("spawn", calls["spawn"] + 1) or 4321)
    monkeypatch.setattr(gw, "start", lambda: calls.__setitem__("start", calls["start"] + 1))


def test_restart_relaunches_manual_gateway_directly(monkeypatch):
    calls = {"spawn": 0, "start": 0}
    _neutralize(monkeypatch, calls)
    monkeypatch.setattr(gw, "is_task_registered", lambda: False)
    monkeypatch.setattr(gw, "is_startup_entry_installed", lambda: False)
    gw.restart()
    assert calls["spawn"] == 1, "manual gateway must be relaunched via _spawn_detached"
    assert calls["start"] == 0, "manual restart must NOT route through start() (install prompt)"


def test_restart_uses_start_when_task_installed(monkeypatch):
    calls = {"spawn": 0, "start": 0}
    _neutralize(monkeypatch, calls)
    monkeypatch.setattr(gw, "is_task_registered", lambda: True)
    monkeypatch.setattr(gw, "is_startup_entry_installed", lambda: False)
    gw.restart()
    assert calls["start"] == 1, "an installed-service gateway must restart via start()"
    assert calls["spawn"] == 0


def test_restart_uses_start_when_startup_entry_installed(monkeypatch):
    calls = {"spawn": 0, "start": 0}
    _neutralize(monkeypatch, calls)
    monkeypatch.setattr(gw, "is_task_registered", lambda: False)
    monkeypatch.setattr(gw, "is_startup_entry_installed", lambda: True)
    gw.restart()
    assert calls["start"] == 1
    assert calls["spawn"] == 0
