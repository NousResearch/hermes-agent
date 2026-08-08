from __future__ import annotations

import threading
import time
from unittest.mock import patch


def _wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


def test_control_plane_cycle_disabled_is_noop(monkeypatch):
    from cron_control import runner

    calls = []
    monkeypatch.setattr(runner, "collect_shadow_snapshot", lambda **_kwargs: calls.append("collect"))
    monkeypatch.setattr(runner, "evaluate_snapshot", lambda snapshot: calls.append("evaluate"))
    monkeypatch.setattr(runner, "persist_verdicts", lambda *args, **kwargs: calls.append("persist"))
    monkeypatch.setattr(runner, "execute_verdict_actions", lambda *args, **kwargs: calls.append("execute"))

    result = runner.run_control_plane_cycle(
        config={"cron": {"control_plane": {"enabled": False, "approve_actions": False}}}
    )

    assert result["enabled"] is False
    assert calls == []


def test_control_plane_cycle_enabled_runs_shadow_verdict_and_action(monkeypatch):
    from cron_control import runner

    calls = []

    def fake_collect(**_kwargs):
        calls.append("collect")
        return {"jobs": [{"id": "job-1"}], "evidence": [{"job_id": "job-1"}]}

    def fake_evaluate(snapshot):
        calls.append(("evaluate", len(snapshot["jobs"]), len(snapshot["evidence"])))
        return [{"verdict_id": "vd_1", "job_id": "job-1", "incident_id": "inc_1"}]

    def fake_persist(snapshot, verdicts, control_plane_path=None):
        calls.append(("persist", len(verdicts), control_plane_path))

    def fake_execute(verdicts, *, approved=False, control_plane_path=None):
        calls.append(("execute", approved, control_plane_path))
        return [{"status": "verified"}]

    monkeypatch.setattr(runner, "collect_shadow_snapshot", fake_collect)
    monkeypatch.setattr(runner, "evaluate_snapshot", fake_evaluate)
    monkeypatch.setattr(runner, "persist_verdicts", fake_persist)
    monkeypatch.setattr(runner, "execute_verdict_actions", fake_execute)

    result = runner.run_control_plane_cycle(
        config={"cron": {"control_plane": {"enabled": True, "approve_actions": True, "persist_shadow": False}}}
    )

    assert result["enabled"] is True
    assert result["approve_actions"] is True
    assert calls == [
        "collect",
        ("evaluate", 1, 1),
        ("persist", 1, None),
        ("execute", True, None),
    ]


def test_inprocess_provider_runs_control_plane_cycle_when_enabled(monkeypatch):
    import hermes_cli.config as cfg
    from cron.scheduler_provider import InProcessCronScheduler

    calls = []
    stop = threading.Event()

    def fake_tick(*args, **kwargs):
        calls.append("tick")
        stop.set()
        return 0

    def fake_cycle(*args, **kwargs):
        calls.append("control")
        return {"enabled": True}

    monkeypatch.setattr(
        cfg,
        "load_config",
        lambda: {"cron": {"control_plane": {"enabled": True, "approve_actions": True}}},
    )

    with patch("cron.scheduler.tick", side_effect=fake_tick), patch(
        "cron_control.runner.run_control_plane_cycle", side_effect=fake_cycle
    ):
        t = threading.Thread(
            target=InProcessCronScheduler().start,
            args=(stop,),
            kwargs={"interval": 0},
            daemon=True,
        )
        t.start()
        assert _wait_until(lambda: calls[:2] == ["tick", "control"]), "control-plane cycle was not invoked"
        t.join(timeout=5)

    assert not t.is_alive(), "provider did not exit after stop_event was set"
    assert calls[:2] == ["tick", "control"]


def test_inprocess_provider_skips_control_plane_when_disabled(monkeypatch):
    import hermes_cli.config as cfg
    from cron.scheduler_provider import InProcessCronScheduler

    calls = []
    stop = threading.Event()

    def fake_tick(*args, **kwargs):
        calls.append("tick")
        stop.set()
        return 0

    def fake_cycle(*args, **kwargs):
        calls.append("control")
        return {"enabled": True}

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"control_plane": {"enabled": False}}})

    with patch("cron.scheduler.tick", side_effect=fake_tick), patch(
        "cron_control.runner.run_control_plane_cycle", side_effect=fake_cycle
    ):
        t = threading.Thread(
            target=InProcessCronScheduler().start,
            args=(stop,),
            kwargs={"interval": 0},
            daemon=True,
        )
        t.start()
        assert _wait_until(lambda: calls == ["tick"]), "expected one tick and no control-plane cycle"
        t.join(timeout=5)

    assert not t.is_alive(), "provider did not exit after stop_event was set"
    assert calls == ["tick"]
