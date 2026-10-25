"""External cron workers must see the routed profile's own non-credential .env settings (#136257).

The multiplex dotenv guard in ``hermes_cli/env_loader.py::load_hermes_dotenv`` skips the
process-global dotenv load for a routed profile home, and the gateway strips the launch
profile's residue before spawn. Nothing re-applies the OWNING profile's own ``.env`` to the
worker's ``os.environ``, so a plugin reading a plain tuning key (``LCM_IGNORE_SESSION_PATTERNS``)
ran on defaults in every scheduled job while CLI and in-process fires read it fine. The worker
is a single-profile process: restoring the owning profile's own non-credential settings is safe,
and credentials keep resolving through the profile secret scope (never ``os.environ``).
"""
from __future__ import annotations

import json
import os

import pytest

PROFILE_ENV = (
    "LCM_IGNORE_SESSION_PATTERNS=session-.*-noisy\n"
    "OPENAI_API_KEY=sk-should-not-leak\n"
    "GATEWAY_RELAY_SECRET=relay-should-not-leak\n"
    "HERMES_KANBAN_DB=/profile/kanban.db\n"
)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A launch home whose residue must not survive and the routed profile owning the job."""
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / ".env").write_text("LCM_IGNORE_SESSION_PATTERNS=launch-residue-value\n", encoding="utf-8")
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / ".env").write_text(PROFILE_ENV, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    yield launch, profile


@pytest.fixture
def captured_env(monkeypatch):
    """Snapshot os.environ around the worker run so restored keys cannot leak between tests."""
    before = dict(os.environ)
    seen: list[dict] = []
    yield seen
    for key in list(os.environ):
        if key not in before:
            del os.environ[key]
        elif os.environ[key] != before[key]:
            os.environ[key] = before[key]


def run_worker(scheduler, profile, tmp_path, monkeypatch, captured_env, *, multiplex_active):
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({"job": {"id": "job-1", "execution_id": "exec-1"},
                    "profile_home": str(profile), "multiplex_active": multiplex_active}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cron.executions.adopt_claimed_execution",
        lambda execution_id: {"id": execution_id, "status": "running"},
    )

    def capture(*_args, **_kwargs):
        captured_env.append(dict(os.environ))
        return True

    monkeypatch.setattr(scheduler, "run_one_job", capture)
    assert scheduler._run_external_worker_payload(payload, tmp_path / "exec-1.ready") is True
    assert captured_env, "run_one_job never captured the worker environment"


def test_worker_sees_owning_profile_noncredential_settings(homes, tmp_path, monkeypatch, captured_env):
    import cron.scheduler as scheduler

    launch, profile = homes
    run_worker(scheduler, profile, tmp_path, monkeypatch, captured_env, multiplex_active=True)
    worker_env = captured_env[0]
    assert worker_env.get("LCM_IGNORE_SESSION_PATTERNS") == "session-.*-noisy"


def test_worker_mono_profile_also_sees_settings(homes, tmp_path, monkeypatch, captured_env):
    import cron.scheduler as scheduler

    launch, profile = homes
    run_worker(scheduler, profile, tmp_path, monkeypatch, captured_env, multiplex_active=False)
    worker_env = captured_env[0]
    assert worker_env.get("LCM_IGNORE_SESSION_PATTERNS") == "session-.*-noisy"


def test_worker_does_not_restore_credentials_or_globals(homes, tmp_path, monkeypatch, captured_env):
    import cron.scheduler as scheduler

    launch, profile = homes
    monkeypatch.setenv("HERMES_KANBAN_DB", "/gateway/kanban.db")
    run_worker(scheduler, profile, tmp_path, monkeypatch, captured_env, multiplex_active=True)
    worker_env = captured_env[0]
    assert worker_env.get("OPENAI_API_KEY") != "sk-should-not-leak"
    assert "OPENAI_API_KEY" not in worker_env
    assert "GATEWAY_RELAY_SECRET" not in worker_env
    assert worker_env.get("HERMES_KANBAN_DB") == "/gateway/kanban.db"
