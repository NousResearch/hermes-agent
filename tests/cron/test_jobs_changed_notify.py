"""Tests for on_jobs_changed wiring (Phase 4F.1).

After a store mutation via the consumer surfaces (model tool / CLI / REST), the
active scheduler provider's on_jobs_changed() must be invoked so an external
provider (Chronos) re-provisions/cancels. The built-in's no-op default means
the default path is unchanged.
"""

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import cron.jobs as jobs_mod

    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", tmp_path)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", cron_dir / "output")
    yield tmp_path


def test_notify_helper_calls_provider_on_jobs_changed(monkeypatch):
    """cron.scheduler._notify_provider_jobs_changed resolves the provider and
    calls on_jobs_changed exactly once."""
    import cron.scheduler_provider as sp
    import cron.scheduler as sched

    calls = []

    class Spy(sp.CronScheduler):
        @property
        def name(self):
            return "spy"

        def start(self, stop_event, **kw):
            pass

        def on_jobs_changed(self):
            calls.append(1)

    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: Spy())
    sched._notify_provider_jobs_changed()
    assert calls == [1]


def test_notify_helper_swallows_provider_errors(monkeypatch):
    """A provider that raises in on_jobs_changed must not propagate into the
    caller (best-effort notify)."""
    import cron.scheduler_provider as sp
    import cron.scheduler as sched

    class Boom(sp.CronScheduler):
        @property
        def name(self):
            return "boom"

        def start(self, stop_event, **kw):
            pass

        def on_jobs_changed(self):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: Boom())
    sched._notify_provider_jobs_changed()  # must not raise


def test_builtin_notify_is_harmless(monkeypatch):
    """With the built-in provider (default), notify is a no-op and never
    raises."""
    import cron.scheduler as sched
    # default resolution → built-in; just assert it doesn't blow up.
    sched._notify_provider_jobs_changed()


def test_tool_create_notifies_provider(temp_home, monkeypatch):
    """Creating a job via the cronjob tool path invokes on_jobs_changed."""
    import cron.scheduler as sched
    calls = []
    monkeypatch.setattr(sched, "_notify_provider_jobs_changed",
                        lambda: calls.append("changed"))

    from tools.cronjob_tools import cronjob
    import json

    out = json.loads(cronjob(action="create", prompt="echo hi", schedule="every 5m", name="w"))
    assert out["success"] is True
    assert calls == ["changed"]


def test_tool_remove_notifies_provider(temp_home, monkeypatch):
    """Removing a job via the tool path invokes on_jobs_changed."""
    import json
    from tools.cronjob_tools import cronjob

    created = json.loads(cronjob(action="create", prompt="x", schedule="every 5m", name="r"))
    jid = created["job_id"]

    import cron.scheduler as sched
    calls = []
    monkeypatch.setattr(sched, "_notify_provider_jobs_changed",
                        lambda: calls.append("changed"))

    out = json.loads(cronjob(action="remove", job_id=jid))
    assert out["success"] is True
    assert calls == ["changed"]
