"""A dashboard cron fire must not leak os.environ writes from its callee."""

import os
from pathlib import Path

import hermes_cli.web_server_cron as _web_server_cron


def test_dashboard_fire_leaves_os_environ_unchanged(tmp_path, monkeypatch):
    key = "RSR_DASHBOARD_FIRE_LEAK_PROBE"
    monkeypatch.delenv(key, raising=False)
    home = Path(tmp_path)
    monkeypatch.setattr(_web_server_cron, "_cron_profile_home", lambda profile: (profile, home))

    class LeakingProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None, manual=None):
            os.environ[key] = "leaked"
            os.environ.pop("PATH", None)
            return True

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: LeakingProvider(),
    )
    env_before = dict(os.environ)

    assert _web_server_cron._fire_cron_job_for_profile("worker_alpha", "job-1") is True

    assert dict(os.environ) == env_before
    assert key not in os.environ
