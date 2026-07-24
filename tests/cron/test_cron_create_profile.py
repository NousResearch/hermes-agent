"""Regression: create_job() must accept ``profile=``.

The cronjob agent tool (tools/cronjob_tools.py) unconditionally passes
``profile=`` into ``cron.jobs.create_job()``. When create_job lacked the
parameter, every ``cronjob action='create'`` through the agent tool failed
with ``TypeError: create_job() got an unexpected keyword argument 'profile'``
— while list/update/read paths kept working, so the breakage only surfaced
on job creation.

These tests pin the contract: create_job accepts profile, stores it on the
job record, and defaults to None (inherit scheduler profile) when unset.
"""

from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so jobs don't leak."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)

    return home


def test_create_job_accepts_and_stores_profile(hermes_env):
    from cron.jobs import create_job

    job = create_job(
        prompt="say ok",
        schedule="0 9 * * 1",
        name="profile-pinned",
        profile="kimi-worker",
    )
    assert job["profile"] == "kimi-worker"


def test_create_job_profile_defaults_to_none(hermes_env):
    from cron.jobs import create_job

    job = create_job(prompt="say ok", schedule="0 9 * * 1", name="no-profile")
    assert job["profile"] is None


def test_create_job_profile_normalizes_empty(hermes_env):
    from cron.jobs import create_job

    job = create_job(
        prompt="say ok",
        schedule="0 9 * * 1",
        name="blank-profile",
        profile="   ",
    )
    assert job["profile"] is None


def test_cronjob_tool_create_with_profile_succeeds(hermes_env):
    """End-to-end: the agent tool path that previously raised TypeError.

    Only applicable when the tool layer advertises profile (forks carrying
    the profile-pinning tool changes). On upstream, where the tool has no
    profile surface, the create_job-level tests above pin the contract.
    """
    from tools import cronjob_tools

    _props = cronjob_tools.CRONJOB_SCHEMA.get("parameters", {}).get("properties", {})
    if "profile" not in _props:
        pytest.skip("tool layer does not expose profile on this tree")

    result = json.loads(
        cronjob_tools.cronjob(
            action="create",
            prompt="say ok",
            schedule="0 9 * * 1",
            name="tool-level-profile",
            profile="worker-a",
        )
    )
    assert result.get("success"), result
    assert result.get("job", {}).get("profile") == "worker-a"
