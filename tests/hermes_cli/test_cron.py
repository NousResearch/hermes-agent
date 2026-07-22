"""Tests for hermes_cli.cron command handling."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from cron.jobs import create_job, get_job, list_jobs
from hermes_cli import cron as cron_cli
from hermes_cli.cron import cron_command


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestCronCommandLifecycle:
    def test_pause_resume_run(self, tmp_cron_dir, capsys):
        job = create_job(prompt="Check server status", schedule="every 1h")

        cron_command(Namespace(cron_command="pause", job_id=job["id"]))
        paused = get_job(job["id"])
        assert paused["state"] == "paused"

        cron_command(Namespace(cron_command="resume", job_id=job["id"]))
        resumed = get_job(job["id"])
        assert resumed["state"] == "scheduled"

        cron_command(Namespace(cron_command="run", job_id=job["id"]))
        triggered = get_job(job["id"])
        assert triggered["state"] == "scheduled"

        out = capsys.readouterr().out
        assert "Paused job" in out
        assert "Resumed job" in out
        assert "Triggered job" in out

    def test_edit_can_replace_and_clear_skills(self, tmp_cron_dir, capsys):
        job = create_job(
            prompt="Combine skill outputs",
            schedule="every 1h",
            skill="blogwatcher",
        )

        cron_command(
            Namespace(
                cron_command="edit",
                job_id=job["id"],
                schedule="every 2h",
                prompt="Revised prompt",
                name="Edited Job",
                deliver=None,
                repeat=None,
                skill=None,
                skills=["maps", "blogwatcher"],
                clear_skills=False,
                add_skills=None,
                remove_skills=None,
                script=None,
                workdir=None,
                no_agent=None,
            )
        )
        updated = get_job(job["id"])
        assert updated["skills"] == ["maps", "blogwatcher"]
        assert updated["name"] == "Edited Job"
        assert updated["prompt"] == "Revised prompt"
        assert updated["schedule_display"] == "every 120m"

        cron_command(
            Namespace(
                cron_command="edit",
                job_id=job["id"],
                schedule=None,
                prompt=None,
                name=None,
                deliver=None,
                repeat=None,
                skill=None,
                skills=None,
                clear_skills=True,
                add_skills=None,
                remove_skills=None,
                script=None,
                workdir=None,
                no_agent=None,
            )
        )
        cleared = get_job(job["id"])
        assert cleared["skills"] == []
        assert cleared["skill"] is None

        out = capsys.readouterr().out
        assert "Updated job" in out

    def test_create_with_multiple_skills(self, tmp_cron_dir, capsys):
        cron_command(
            Namespace(
                cron_command="create",
                schedule="every 1h",
                prompt="Use both skills",
                name="Skill combo",
                deliver=None,
                repeat=None,
                skill=None,
                skills=["blogwatcher", "maps"],
                script=None,
                workdir=None,
                no_agent=False,
            )
        )
        out = capsys.readouterr().out
        assert "Created job" in out

        jobs = list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["skills"] == ["blogwatcher", "maps"]
        assert jobs[0]["name"] == "Skill combo"

    def test_list_does_not_crash_when_repeat_is_null(self, tmp_cron_dir, capsys):
        """A one-shot job can be persisted with ``"repeat": null``. `cron
        list` must render it as ∞ rather than crashing on .get(...)\\.get."""
        from cron.jobs import load_jobs, save_jobs

        create_job(prompt="One shot", schedule="every 1h")
        # Force the present-but-null shape that .get("repeat", {}) mishandles.
        jobs = load_jobs()
        jobs[0]["repeat"] = None
        save_jobs(jobs)

        cron_command(Namespace(cron_command="list", all=True))

        out = capsys.readouterr().out
        assert "Repeat:    ∞" in out

    def test_list_does_not_crash_when_deliver_is_null(self, tmp_cron_dir, capsys):
        """A job can be persisted with ``"deliver": null`` (present-but-null).
        `cron list` must fall back to the default channel rather than crashing
        on ``", ".join(None)`` — same dict-default pitfall as ``repeat`` (#32896).
        """
        from cron.jobs import load_jobs, save_jobs

        create_job(prompt="No deliver", schedule="every 1h")
        jobs = load_jobs()
        jobs[0]["deliver"] = None
        save_jobs(jobs)

        cron_command(Namespace(cron_command="list", all=True))

        out = capsys.readouterr().out
        assert "Deliver:   local" in out


class TestGatewayNotRunningWarning:
    """`cron create` / `cron list` must warn when the gateway (and thus the
    cron ticker) isn't running, since jobs only fire inside the gateway.
    Regression guard for #51038 — the most common cron 'jobs never fired'
    report was simply a gateway that was never started.
    """

    def test_create_warns_when_gateway_absent(self, tmp_cron_dir, capsys, monkeypatch):
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
        cron_command(
            Namespace(
                cron_command="create",
                schedule="0 11 * * *",
                prompt="Daily report",
                name="Daily 1130",
                deliver=None,
                repeat=None,
                skill=None,
                skills=None,
                script=None,
                workdir=None,
                no_agent=False,
            )
        )
        out = capsys.readouterr().out
        assert "Created job" in out
        assert "Gateway is not running" in out

    def test_create_silent_when_gateway_running(self, tmp_cron_dir, capsys, monkeypatch):
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [4242])
        cron_command(
            Namespace(
                cron_command="create",
                schedule="0 11 * * *",
                prompt="Daily report",
                name="Daily 1130",
                deliver=None,
                repeat=None,
                skill=None,
                skills=None,
                script=None,
                workdir=None,
                no_agent=False,
            )
        )
        out = capsys.readouterr().out
        assert "Created job" in out
        assert "Gateway is not running" not in out

    def test_list_warns_when_gateway_absent(self, tmp_cron_dir, capsys, monkeypatch):
        create_job(prompt="Daily report", schedule="0 11 * * *")
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
        cron_command(Namespace(cron_command="list", all=True))
        out = capsys.readouterr().out
        assert "Gateway is not running" in out


class TestExternalCronProviderStatus:
    """With an external cron provider (e.g. Chronos), jobs fire via a
    NAS-mediated webhook, NOT the in-process ticker. The ticker-heartbeat /
    gateway-process heuristics are meaningless there, so neither
    `cron status` nor the create/list warning must claim the gateway being
    absent means jobs won't fire — that was a false-negative on every healthy
    Chronos instance (the heartbeat is intentionally never written).
    """

    def test_status_reports_provider_not_ticker_for_chronos(
        self, tmp_cron_dir, capsys, monkeypatch
    ):
        create_job(prompt="Ping", schedule="every 2m")
        monkeypatch.setattr(
            "hermes_cli.cron._active_cron_provider_name", lambda: "chronos"
        )
        # Even with NO gateway process and NO ticker heartbeat, Chronos status
        # must NOT report a stall / "not firing".
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
        cron_command(Namespace(cron_command="status"))
        out = capsys.readouterr().out
        assert "chronos" in out
        assert "managed scheduler" in out
        assert "not firing" not in out.lower()
        assert "STALLED" not in out
        assert "Gateway is not running" not in out
        # Still surfaces the active-job summary.
        assert "active job(s)" in out

    def test_status_unchanged_for_builtin(self, tmp_cron_dir, capsys, monkeypatch):
        create_job(prompt="Ping", schedule="every 2m")
        monkeypatch.setattr(
            "hermes_cli.cron._active_cron_provider_name", lambda: "builtin"
        )
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
        cron_command(Namespace(cron_command="status"))
        out = capsys.readouterr().out
        # Built-in path is the historical ticker-based report.
        assert "Gateway is not running" in out
        assert "managed scheduler" not in out

    def test_create_silent_for_chronos_even_without_gateway(
        self, tmp_cron_dir, capsys, monkeypatch
    ):
        # The create-time "gateway not running" nag is a ticker-only concern;
        # an external provider doesn't depend on a live in-process ticker.
        monkeypatch.setattr(
            "hermes_cli.cron._active_cron_provider_name", lambda: "chronos"
        )
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
        cron_command(
            Namespace(
                cron_command="create",
                schedule="every 2m",
                prompt="Ping",
                name="Ping",
                deliver=None,
                repeat=None,
                skill=None,
                skills=None,
                script=None,
                workdir=None,
                no_agent=False,
            )
        )
        out = capsys.readouterr().out
        assert "Created job" in out
        assert "Gateway is not running" not in out


def test_cron_list_warns_when_gateway_not_running(monkeypatch, capsys):
    monkeypatch.setattr(
        "cron.jobs.list_jobs",
        lambda include_disabled=False: [
            {
                "id": "job-1",
                "name": "Nightly docs",
                "schedule_display": "every day",
                "state": "scheduled",
                "enabled": True,
                "next_run_at": "2026-06-01T00:00:00Z",
                "deliver": ["local"],
            }
        ],
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
    monkeypatch.setattr(cron_cli, "_active_cron_provider_name", lambda: "builtin")

    cron_cli.cron_list()

    out = capsys.readouterr().out
    assert "Gateway is not running" in out
    assert "Nightly docs" in out


def test_cron_status_reports_running_gateway(monkeypatch, capsys):
    monkeypatch.setattr(cron_cli, "_active_cron_provider_name", lambda: "builtin")
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [1234, 5678])
    monkeypatch.setattr(
        "cron.jobs.list_jobs",
        lambda include_disabled=False: [
            {"next_run_at": "2026-06-01T00:00:00Z"},
            {"next_run_at": "2026-05-31T12:00:00Z"},
        ],
    )

    cron_cli.cron_status()

    out = capsys.readouterr().out
    assert "Gateway is running" in out
    assert "1234, 5678" in out
    assert "2 active job(s)" in out
    assert "2026-05-31T12:00:00Z" in out


def test_cron_tick_invokes_scheduler_tick_with_verbose(monkeypatch):
    calls = []
    monkeypatch.setattr("cron.scheduler.tick", lambda verbose=False: calls.append(verbose))

    cron_cli.cron_tick()

    assert calls == [True]


def test_cron_create_success_prints_job_details(monkeypatch, capsys):
    monkeypatch.setattr(
        cron_cli,
        "_cron_api",
        lambda **kwargs: {
            "success": True,
            "job_id": "job-1",
            "name": "Nightly docs",
            "schedule": "every day",
            "skills": ["docs"],
            "next_run_at": "2026-06-01T00:00:00Z",
            "job": {
                "script": "scripts/build_docs.py",
                "no_agent": True,
                "workdir": "/tmp/repo",
            },
        },
    )
    monkeypatch.setattr(cron_cli, "_warn_if_gateway_not_running", lambda: None)

    args = SimpleNamespace(
        schedule="every day",
        prompt="refresh docs",
        name="Nightly docs",
        deliver=None,
        repeat=None,
        skill="docs",
        skills=None,
        script="scripts/build_docs.py",
        workdir="/tmp/repo",
        no_agent=True,
    )

    rc = cron_cli.cron_create(args)

    out = capsys.readouterr().out
    assert rc == 0
    assert "Created job: job-1" in out
    assert "Skills: docs" in out
    assert "Script: scripts/build_docs.py" in out
    assert "Mode: no-agent" in out
    assert "Workdir: /tmp/repo" in out
    assert "Next run: 2026-06-01T00:00:00Z" in out


def test_cron_create_failure_returns_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(cron_cli, "_cron_api", lambda **kwargs: {"success": False, "error": "boom"})

    args = SimpleNamespace(
        schedule="every day",
        prompt="refresh docs",
        name=None,
        deliver=None,
        repeat=None,
        skill=None,
        skills=None,
        script=None,
        workdir=None,
        no_agent=False,
    )

    rc = cron_cli.cron_create(args)

    out = capsys.readouterr().out
    assert rc == 1
    assert "Failed to create job: boom" in out


# ── Background ticker unit tests (#67102) ────────────────────────────────────


def _wait_until_cron(predicate, timeout=10.0, interval=0.005):
    """Block until ``predicate()`` is truthy or ``timeout`` elapses.

    Returns the predicate's final value. Used instead of a fixed
    ``time.sleep`` before asserting that a background ticker thread has called
    tick() at least N times — under loaded CI the worker thread may not be
    scheduled within a short fixed sleep, which made these tests flake.
    """
    import time as _t

    deadline = _t.monotonic() + timeout
    while _t.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        _t.sleep(interval)
    return predicate()


class TestCronTickerBackground:
    """Dedicated unit tests for the InProcessCronScheduler background ticker
    loop — the in-process cron trigger that fires due jobs every N seconds
    inside the gateway.

    These test the ticker's core loop contract: interval-based firing,
    graceful shutdown via stop_event, adapter pass-through, and error
    resilience (the ticker must keep looping even when tick() raises).
    """

    def test_cron_ticker_fires_at_interval(self):
        """The ticker fires tick() repeatedly at the configured interval, and
        successive calls are spaced at least ``interval`` seconds apart."""
        import time as _t
        import threading
        from unittest.mock import patch

        from cron.scheduler_provider import InProcessCronScheduler

        timestamps = []
        stop = threading.Event()

        def _fake_tick(*_a, **_k):
            timestamps.append(_t.monotonic())
            return 0

        with patch("cron.scheduler.tick", side_effect=_fake_tick), \
             patch("cron.jobs.record_ticker_heartbeat"):
            prov = InProcessCronScheduler()
            interval = 0.01  # 10 ms — fast but measurable
            t = threading.Thread(
                target=prov.start,
                args=(stop,),
                kwargs={"interval": interval},
                daemon=True,
            )
            t.start()

            assert _wait_until_cron(
                lambda: len(timestamps) >= 3
            ), f"Expected ≥3 ticks, got {len(timestamps)}"

            stop.set()
            t.join(timeout=5)

        assert not t.is_alive(), "ticker did not exit after stop_event was set"
        assert len(timestamps) >= 3, f"Expected ≥3 ticks, got {len(timestamps)}"

        # Each successive tick must be at least ~half the interval apart
        # (allows a tiny bit of scheduling jitter but catches a tight loop).
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            assert gap >= interval * 0.5, (
                f"Tick {i} came too soon after tick {i - 1} "
                f"(gap={gap:.4f}s, expected ≥{interval * 0.5:.4f}s)"
            )

    def test_cron_ticker_stop_event(self):
        """The ticker exits cleanly (thread terminates) when stop_event is
        set, even if tick() was in the middle of a cycle."""
        import threading
        from unittest.mock import patch

        from cron.scheduler_provider import InProcessCronScheduler

        calls = []
        stop = threading.Event()

        with patch("cron.scheduler.tick",
                   side_effect=lambda *_a, **_k: calls.append(1) or 0), \
             patch("cron.jobs.record_ticker_heartbeat"):
            prov = InProcessCronScheduler()
            t = threading.Thread(
                target=prov.start,
                args=(stop,),
                kwargs={"interval": 0},
                daemon=True,
            )
            t.start()

            assert _wait_until_cron(
                lambda: len(calls) >= 1
            ), "ticker never called tick()"

            stop.set()
            t.join(timeout=5)

        assert not t.is_alive(), "ticker did not exit after stop_event was set"
        assert len(calls) >= 1, "ticker never called tick()"

    def test_cron_ticker_with_adapter(self):
        """Adapters are passed through to tick() on every iteration so the
        gateway can deliver cron results to live messaging platforms."""
        import threading
        from unittest.mock import patch

        from cron.scheduler_provider import InProcessCronScheduler

        tick_kwargs = []
        stop = threading.Event()
        fake_adapters = {"slack": object(), "telegram": object()}

        def _fake_tick(**kwargs):
            tick_kwargs.append(kwargs)
            return 0

        with patch("cron.scheduler.tick", side_effect=_fake_tick), \
             patch("cron.jobs.record_ticker_heartbeat"):
            prov = InProcessCronScheduler()
            t = threading.Thread(
                target=prov.start,
                args=(stop,),
                kwargs={"interval": 0, "adapters": fake_adapters},
                daemon=True,
            )
            t.start()

            assert _wait_until_cron(
                lambda: len(tick_kwargs) >= 1
            ), "ticker never called tick() with adapters"

            stop.set()
            t.join(timeout=5)

        assert len(tick_kwargs) >= 1
        # The adapter dict should be passed through verbatim
        assert tick_kwargs[0].get("adapters") is fake_adapters

    def test_cron_ticker_handles_adapter_error(self):
        """When tick() raises an exception (e.g. an adapter is unreachable),
        the ticker logs the error and continues looping — it does NOT exit."""
        import threading
        from unittest.mock import patch

        from cron.scheduler_provider import InProcessCronScheduler

        calls = []
        stop = threading.Event()

        def _failing_tick(*_a, **_k):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("simulated adapter failure")
            return 0

        with patch("cron.scheduler.tick", side_effect=_failing_tick), \
             patch("cron.jobs.record_ticker_heartbeat"):
            prov = InProcessCronScheduler()
            t = threading.Thread(
                target=prov.start,
                args=(stop,),
                kwargs={"interval": 0},
                daemon=True,
            )
            t.start()

            # First call raises → second call must still happen (survival).
            assert _wait_until_cron(
                lambda: len(calls) >= 2
            ), f"Ticker stopped after error instead of continuing (got {len(calls)} ticks)"

            stop.set()
            t.join(timeout=5)

        assert len(calls) >= 2, (
            f"Ticker should keep looping after an error, "
            f"but only got {len(calls)} tick(s)"
        )
        assert not t.is_alive(), "ticker did not exit after stop_event was set"
