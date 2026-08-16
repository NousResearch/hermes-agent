"""Tests for cronjob attach_to_session persistence (#84802).

The cronjob tool schema documents ``attach_to_session`` as a per-job opt-in
that makes a cron delivery continuable, but the registered tool handler
silently dropped it: create reported success without persisting the field, and
an update touching only that field returned ``No updates provided.``.

The fix forwards the argument in the registry adapter (create AND update),
exposes it through ``_format_job`` (so ``cronjob(action='list')`` shows it),
and surfaces a warning when the value cannot take effect (a local-only job
with no delivery target) instead of accepting it silently.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tools.cronjob_tools import cronjob  # noqa: F401  (ensures registry registration)


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Clear cached module-level paths
    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


def _handler():
    """The registered cronjob tool handler (the registry adapter under test)."""
    from tools.registry import registry

    entry = registry.get_entry("cronjob")
    assert entry is not None, "cronjob tool must be registered"
    return entry.handler


# ---------------------------------------------------------------------------
# Registry handler forwarding (the actual bug: the adapter dropped the arg)
# ---------------------------------------------------------------------------


def test_handler_forwards_attach_to_session_on_create():
    """The registered handler must forward attach_to_session to create_job."""
    with patch("tools.cronjob_tools.create_job") as m_create, \
         patch("tools.cronjob_tools._notify_provider_jobs_changed_safe"):
        m_create.return_value = {
            "id": "job-1",
            "name": "continuable canary",
            "prompt": "Reply exactly: canary",
            "schedule": {"kind": "interval", "seconds": 3600},
            "schedule_display": "1h",
            "repeat": {"times": 1},
            "deliver": "origin",
            "next_run_at": "2026-08-13T00:00:00Z",
        }
        out = json.loads(_handler()({
            "action": "create",
            "name": "continuable canary",
            "schedule": "1h",
            "repeat": 1,
            "deliver": "origin",
            "attach_to_session": True,
            "prompt": "Reply exactly: canary",
        }))

    assert out["success"] is True, out
    assert m_create.call_args.kwargs["attach_to_session"] is True


def test_handler_forwards_attach_to_session_on_update():
    """Update through the handler must persist attach_to_session — the exact
    ``No updates provided.`` repro from the issue must now succeed."""
    existing = {
        "id": "job-1",
        "name": "canary",
        "prompt": "hi",
        "schedule": {"kind": "interval", "seconds": 3600},
        "schedule_display": "1h",
        "repeat": {"times": 1},
        "deliver": "local",
    }
    with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(existing)), \
         patch("tools.cronjob_tools.update_job",
               return_value={**existing, "attach_to_session": True}) as m_update, \
         patch("tools.cronjob_tools._notify_provider_jobs_changed_safe"):
        out = json.loads(_handler()({
            "action": "update",
            "job_id": "job-1",
            "attach_to_session": True,
        }))

    assert out["success"] is True, out
    assert m_update.call_args.args[0] == "job-1"
    assert m_update.call_args.args[1]["attach_to_session"] is True


# ---------------------------------------------------------------------------
# End-to-end persistence through the registered handler
# ---------------------------------------------------------------------------


def test_create_persists_attach_to_session(cron_env):
    """create persists attach_to_session in the job store and surfaces a
    warning when the flag cannot take effect (local-only job)."""
    from cron.jobs import get_job

    out = json.loads(_handler()({
        "action": "create",
        "name": "continuable canary",
        "schedule": "1h",
        "repeat": 1,
        "deliver": "local",
        "attach_to_session": True,
        "prompt": "Reply exactly: canary",
    }))
    assert out["success"] is True, out

    job = get_job(out["job_id"])
    assert job["attach_to_session"] is True

    # Fail-closed: the flag is inert for a local-only job — the caller is told.
    assert "attach_to_session has no effect" in out["message"]


def test_update_persists_attach_to_session(cron_env):
    """update with ONLY attach_to_session works (no more 'No updates
    provided.'), persists the field in both directions, and warns when the
    flag cannot take effect."""
    from cron.jobs import get_job

    created = json.loads(_handler()({
        "action": "create",
        "name": "canary",
        "schedule": "1h",
        "repeat": 1,
        "deliver": "local",
        "prompt": "Reply exactly: canary",
    }))
    assert created["success"] is True, created
    job_id = created["job_id"]
    assert get_job(job_id).get("attach_to_session") is None

    out = json.loads(_handler()({
        "action": "update",
        "job_id": job_id,
        "attach_to_session": True,
    }))
    assert out["success"] is True, out
    assert out["job"]["attach_to_session"] is True
    assert get_job(job_id)["attach_to_session"] is True
    assert "attach_to_session has no effect" in out["warning"]

    out2 = json.loads(_handler()({
        "action": "update",
        "job_id": job_id,
        "attach_to_session": False,
    }))
    assert out2["success"] is True, out2
    assert out2["job"]["attach_to_session"] is False
    assert get_job(job_id)["attach_to_session"] is False


def test_no_warning_when_attach_to_session_applies(cron_env):
    """A job that resolves a delivery target gets no attach_to_session warning."""
    with patch("cron.scheduler._resolve_delivery_targets",
               return_value=[{"platform": "telegram", "chat_id": "1"}]):
        out = json.loads(_handler()({
            "action": "create",
            "name": "delivered canary",
            "schedule": "1h",
            "deliver": "telegram",
            "attach_to_session": True,
            "prompt": "hi",
        }))

    assert out["success"] is True, out
    assert "attach_to_session has no effect" not in out["message"]


def test_list_shows_attach_to_session(cron_env):
    """cronjob(action='list') surfaces the persisted attach_to_session flag."""
    created = json.loads(_handler()({
        "action": "create",
        "name": "canary",
        "schedule": "1h",
        "deliver": "local",
        "attach_to_session": True,
        "prompt": "hi",
    }))
    assert created["success"] is True, created

    listed = json.loads(_handler()({"action": "list"}))
    assert listed["success"] is True, listed
    job = next(j for j in listed["jobs"] if j["job_id"] == created["job_id"])
    assert job["attach_to_session"] is True
