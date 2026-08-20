"""Per-job Docker sandbox image isolation (the ``docker_image`` cron field).

Covers the field plumbing (create_job persistence, cronjob-tool update,
list/get echo), the deliberate exclusion from the agent-facing tool schema
(user-owned, like ``model`` / ``provider``), and the scheduler helper that
turns a pinned image into an isolated per-task sandbox.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with a temp HERMES_HOME + jobs store."""
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    return hermes_home


# --------------------------------------------------------------------------
# cron/jobs.py — create_job persists the field
# --------------------------------------------------------------------------

def test_create_job_persists_docker_image(cron_env):
    from cron.jobs import create_job, get_job

    job = create_job(prompt="check", schedule="every 1h", docker_image="my-sandbox:latest")
    assert job["docker_image"] == "my-sandbox:latest"
    assert get_job(job["id"])["docker_image"] == "my-sandbox:latest"


def test_create_job_docker_image_defaults_none(cron_env):
    from cron.jobs import create_job

    assert create_job(prompt="x", schedule="every 1h").get("docker_image") is None
    # blank / whitespace normalizes to None, never stored as ""
    blank = create_job(prompt="y", schedule="every 1h", docker_image="   ")
    assert blank.get("docker_image") is None


# --------------------------------------------------------------------------
# tools/cronjob_tools.py — update threads + clears; list/get echoes
# --------------------------------------------------------------------------

def test_cronjob_update_sets_and_clears_docker_image(cron_env):
    from cron.jobs import create_job, get_job
    from tools.cronjob_tools import cronjob

    jid = create_job(prompt="x", schedule="every 1h")["id"]

    res = json.loads(cronjob(action="update", job_id=jid, docker_image="img:1"))
    assert res.get("success") is True
    assert get_job(jid)["docker_image"] == "img:1"

    # empty string clears the pin -> back to the shared default sandbox
    res = json.loads(cronjob(action="update", job_id=jid, docker_image=""))
    assert res.get("success") is True
    assert get_job(jid).get("docker_image") is None


def test_format_job_echoes_docker_image():
    from tools.cronjob_tools import _format_job

    out = _format_job({"id": "abc", "name": "n", "docker_image": "img:2"})
    assert out.get("docker_image") == "img:2"
    # absent when the job has no pin
    assert "docker_image" not in _format_job({"id": "abc", "name": "n"})


def test_docker_image_absent_from_agent_tool_schema():
    """user-owned like model/provider: the agent's cronjob tool cannot set it."""
    from tools.cronjob_tools import CRONJOB_SCHEMA

    props = CRONJOB_SCHEMA["parameters"]["properties"]
    assert "docker_image" not in props
    assert "model" not in props and "provider" not in props  # same policy
    assert "workdir" in props  # contrast: workdir IS agent-settable


# --------------------------------------------------------------------------
# cron/scheduler.py — helper turns a pinned image into an isolated sandbox
# --------------------------------------------------------------------------

def test_resolve_cron_image_task_id_registers_and_isolates():
    from cron.scheduler import _resolve_cron_image_task_id
    from tools.terminal_tool import (
        _resolve_container_task_id,
        resolve_task_overrides,
        clear_task_env_overrides,
    )

    task_id = None
    try:
        task_id = _resolve_cron_image_task_id({"id": "job123", "docker_image": "iso-img:1"})
        assert task_id == "cronimg-job123"
        # the registered image override makes this task resolve to its OWN
        # container instead of collapsing to the shared "default" sandbox
        assert _resolve_container_task_id(task_id) == task_id
        assert resolve_task_overrides(task_id).get("docker_image") == "iso-img:1"
    finally:
        if task_id:
            clear_task_env_overrides(task_id)


def test_resolve_cron_image_task_id_none_when_unset():
    from cron.scheduler import _resolve_cron_image_task_id

    assert _resolve_cron_image_task_id({"id": "job123"}) is None
    assert _resolve_cron_image_task_id({"id": "job123", "docker_image": "   "}) is None
