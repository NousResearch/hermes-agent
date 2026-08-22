"""Regression tests for empty-payload cron jobs (incident a5e29e688dc0).

A job whose effective runnable payload is nothing — blank/whitespace prompt,
no script, no skills — used to be creatable, updatable into existence, and
would wake the LLM with an empty instruction on every fire.

Covers:

* ``create_job`` / ``update_job`` rejecting the empty shape (merged record).
* Valid shapes staying valid: no_agent+script, agent script-only, prompt-only,
  skills-only.
* ``scheduler.run_job`` failing closed on a legacy/hand-edited record before
  constructing the agent, and pausing it so it stops being scheduled.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so jobs/scripts don't leak."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


# ---------------------------------------------------------------------------
# create_job validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", [None, "", "   ", "\n\t "])
def test_create_job_rejects_empty_payload(hermes_env, prompt):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="nothing to run"):
        create_job(prompt=prompt, schedule="every 5m")


def test_create_job_rejects_blank_script_and_blank_skills(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="nothing to run"):
        create_job(prompt="  ", schedule="every 5m", script="   ", skills=["", "  "])


def test_create_job_no_agent_error_still_wins(hermes_env):
    """no_agent=True without a script keeps its own, more specific message."""
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="no_agent=True requires a script"):
        create_job(prompt=None, schedule="every 5m", no_agent=True)


# ---------------------------------------------------------------------------
# Valid shapes stay valid
# ---------------------------------------------------------------------------


def test_valid_shapes_are_accepted(hermes_env):
    from cron.jobs import create_job

    (hermes_env / "scripts" / "w.sh").write_text("echo hi\n")

    no_agent_job = create_job(
        prompt=None, schedule="every 5m", script="w.sh", no_agent=True, deliver="local"
    )
    assert no_agent_job["no_agent"] is True

    script_only_agent_job = create_job(
        prompt=None, schedule="every 5m", script="w.sh", deliver="local"
    )
    assert script_only_agent_job["script"] == "w.sh"
    assert script_only_agent_job["no_agent"] is False

    prompt_job = create_job(prompt="check the news", schedule="every 5m", deliver="local")
    assert prompt_job["prompt"] == "check the news"

    skills_job = create_job(
        prompt=None, schedule="every 5m", skills=["daily-report"], deliver="local"
    )
    assert skills_job["skills"] == ["daily-report"]

    legacy_skill_job = create_job(
        prompt="  ", schedule="every 5m", skill="daily-report", deliver="local"
    )
    assert legacy_skill_job["skill"] == "daily-report"


# ---------------------------------------------------------------------------
# update_job validation — the MERGED record is what counts
# ---------------------------------------------------------------------------


def test_update_job_rejects_clearing_the_only_payload(hermes_env):
    from cron.jobs import create_job, get_job, update_job

    job = create_job(prompt="check the news", schedule="every 5m", deliver="local")

    with pytest.raises(ValueError, match="nothing to run"):
        update_job(job["id"], {"prompt": "   "})

    # Nothing was persisted.
    assert get_job(job["id"])["prompt"] == "check the news"


def test_update_job_rejects_dropping_last_skill_from_promptless_job(hermes_env):
    from cron.jobs import create_job, update_job

    job = create_job(prompt=None, schedule="every 5m", skills=["daily-report"], deliver="local")

    with pytest.raises(ValueError, match="nothing to run"):
        update_job(job["id"], {"skills": []})


def test_update_job_rejects_clearing_script_from_promptless_job(hermes_env):
    from cron.jobs import create_job, update_job

    (hermes_env / "scripts" / "w.sh").write_text("echo hi\n")
    job = create_job(prompt=None, schedule="every 5m", script="w.sh", deliver="local")

    with pytest.raises(ValueError, match="nothing to run"):
        update_job(job["id"], {"script": ""})


def test_update_job_rejects_toggling_no_agent_on_without_a_script(hermes_env):
    """create_job enforces no_agent ⇒ script; update_job must too."""
    from cron.jobs import create_job, get_job, update_job

    job = create_job(prompt="check the news", schedule="every 5m", deliver="local")

    with pytest.raises(ValueError, match="no_agent=True requires a script"):
        update_job(job["id"], {"no_agent": True})

    assert get_job(job["id"])["no_agent"] is False


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_update_job_rejects_removing_script_from_a_no_agent_job(hermes_env, blank):
    from cron.jobs import create_job, get_job, update_job

    (hermes_env / "scripts" / "w.sh").write_text("echo hi\n")
    job = create_job(
        prompt=None, schedule="every 5m", script="w.sh", no_agent=True, deliver="local"
    )

    with pytest.raises(ValueError, match="no_agent=True requires a script"):
        update_job(job["id"], {"script": blank})

    assert get_job(job["id"])["script"] == "w.sh"


def test_update_job_rejects_swapping_script_for_prompt_on_a_no_agent_job(hermes_env):
    """A prompt does not rescue no_agent — there is no agent to read it."""
    from cron.jobs import create_job, update_job

    (hermes_env / "scripts" / "w.sh").write_text("echo hi\n")
    job = create_job(
        prompt=None, schedule="every 5m", script="w.sh", no_agent=True, deliver="local"
    )

    with pytest.raises(ValueError, match="no_agent=True requires a script"):
        update_job(job["id"], {"script": "", "prompt": "do it yourself"})


def test_update_job_allows_dropping_script_when_no_agent_is_turned_off(hermes_env):
    """Both fields in one update: the merged record is a valid prompt job."""
    from cron.jobs import create_job, get_job, update_job

    (hermes_env / "scripts" / "w.sh").write_text("echo hi\n")
    job = create_job(
        prompt=None, schedule="every 5m", script="w.sh", no_agent=True, deliver="local"
    )

    update_job(job["id"], {"no_agent": False, "script": None, "prompt": "check the news"})

    stored = get_job(job["id"])
    assert stored["no_agent"] is False
    assert stored["prompt"] == "check the news"


def test_update_job_allows_swapping_the_script_of_a_no_agent_job(hermes_env):
    from cron.jobs import create_job, get_job, update_job

    (hermes_env / "scripts" / "w.sh").write_text("echo hi\n")
    job = create_job(
        prompt=None, schedule="every 5m", script="w.sh", no_agent=True, deliver="local"
    )

    update_job(job["id"], {"script": "other.sh"})
    assert get_job(job["id"])["script"] == "other.sh"


def test_update_job_allows_clearing_prompt_when_script_remains(hermes_env):
    """The merged record still has a script — that is a valid agent job."""
    from cron.jobs import create_job, get_job, update_job

    (hermes_env / "scripts" / "w.sh").write_text("echo hi\n")
    job = create_job(
        prompt="summarize this", schedule="every 5m", script="w.sh", deliver="local"
    )

    update_job(job["id"], {"prompt": ""})
    assert get_job(job["id"])["script"] == "w.sh"


def test_update_job_allows_swapping_prompt_for_skill(hermes_env):
    from cron.jobs import create_job, get_job, update_job

    job = create_job(prompt="check the news", schedule="every 5m", deliver="local")
    update_job(job["id"], {"prompt": "", "skills": ["daily-report"]})

    stored = get_job(job["id"])
    assert stored["skills"] == ["daily-report"]
    assert not (stored["prompt"] or "").strip()


def test_pause_job_still_works_on_an_already_empty_job(hermes_env):
    """Bookkeeping updates must not be blocked, or the fix can't pause the job."""
    from cron.jobs import get_job, pause_job

    job = _legacy_empty_job(hermes_env)

    paused = pause_job(job["id"], reason="empty payload")
    assert paused is not None

    stored = get_job(job["id"])
    assert stored["enabled"] is False
    assert stored["state"] == "paused"


# ---------------------------------------------------------------------------
# run_job runtime guard
# ---------------------------------------------------------------------------


def _legacy_empty_job(hermes_env):
    """Plant a jobs.json record predating the create/update guard.

    Built via ``create_job`` (so every derived field — schedule, repeat,
    snapshots — is realistic) then hand-blanked on disk, which is exactly how
    such a record comes into existence today.
    """
    from cron.jobs import create_job, load_jobs, save_jobs

    job = create_job(prompt="used to say something", schedule="every 5m", deliver="local")
    jobs = load_jobs()
    for stored in jobs:
        if stored["id"] == job["id"]:
            stored["prompt"] = "   "
    save_jobs(jobs)
    return dict(job, prompt="   ")


def test_run_job_fails_closed_and_never_builds_an_agent(hermes_env):
    import cron.scheduler as scheduler

    job = _legacy_empty_job(hermes_env)

    class _Boom:
        def __init__(self, *a, **kw):  # pragma: no cover - must never run
            raise AssertionError("AIAgent must not be constructed for an empty job")

    with patch("run_agent.AIAgent", _Boom):
        success, doc, final, error = scheduler.run_job(job)

    assert success is False
    assert "nothing to run" in error
    assert "auto-paused" in doc
    # Not silent: the user gets told why the job stopped.
    assert "auto-paused" in final


def test_run_job_pauses_the_job_on_disk(hermes_env):
    """Fail-closed isn't enough — the job must stop being scheduled."""
    import cron.scheduler as scheduler
    from cron.jobs import get_job

    job = _legacy_empty_job(hermes_env)

    scheduler.run_job(job)

    stored = get_job(job["id"])
    assert stored["enabled"] is False
    assert stored["state"] == "paused"
    assert "nothing to run" in (stored["paused_reason"] or "")
    assert stored["paused_at"]


def test_run_one_job_does_not_resurrect_the_paused_job(hermes_env):
    """The real caller runs post-run bookkeeping (mark_job_run) after run_job
    returns. That must not undo the pause, or the job re-fires every tick."""
    import cron.scheduler as scheduler
    from cron.jobs import get_due_jobs, get_job

    job = _legacy_empty_job(hermes_env)

    scheduler.run_one_job(job)

    stored = get_job(job["id"])
    assert stored["enabled"] is False
    assert stored["state"] == "paused"
    # The whole point: it must never come up as due again.
    assert [j["id"] for j in get_due_jobs()] == []


def _legacy_no_agent_scriptless_job(hermes_env, script_value=None):
    """Plant a no_agent job whose script went missing after creation."""
    from cron.jobs import create_job, load_jobs, save_jobs

    (hermes_env / "scripts" / "w.sh").write_text("echo hi\n")
    job = create_job(
        prompt=None, schedule="every 5m", script="w.sh", no_agent=True, deliver="local"
    )
    jobs = load_jobs()
    for stored in jobs:
        if stored["id"] == job["id"]:
            stored["script"] = script_value
    save_jobs(jobs)
    return dict(job, script=script_value)


@pytest.mark.parametrize("script_value", [None, "", "   "])
def test_run_job_pauses_a_legacy_no_agent_job_without_a_script(hermes_env, script_value):
    """Erroring alone left it enabled, so it re-fired every tick."""
    import cron.scheduler as scheduler
    from cron.jobs import get_job

    job = _legacy_no_agent_scriptless_job(hermes_env, script_value)

    success, doc, final, error = scheduler.run_job(job)

    assert success is False
    assert "no_agent=True requires a script" in error
    assert "auto-paused" in doc
    assert "auto-paused" in final

    stored = get_job(job["id"])
    assert stored["enabled"] is False
    assert stored["state"] == "paused"
    assert "no_agent=True requires a script" in (stored["paused_reason"] or "")


def test_run_one_job_does_not_resurrect_the_paused_no_agent_job(hermes_env):
    """Post-run bookkeeping must not put it back in the due queue."""
    import cron.scheduler as scheduler
    from cron.jobs import get_due_jobs, get_job

    job = _legacy_no_agent_scriptless_job(hermes_env)

    scheduler.run_one_job(job)

    stored = get_job(job["id"])
    assert stored["enabled"] is False
    assert stored["state"] == "paused"
    assert [j["id"] for j in get_due_jobs()] == []


def test_run_job_does_not_block_a_valid_no_agent_job(hermes_env):
    """The guard sits after the no_agent short-circuit, which must still run."""
    import cron.scheduler as scheduler

    script = hermes_env / "scripts" / "w.sh"
    script.write_text("echo hello\n")

    job = dict(_legacy_empty_job(hermes_env), script="w.sh", no_agent=True)
    success, doc, final, error = scheduler.run_job(job)

    assert success is True
    assert error is None
    assert "hello" in final


# ---------------------------------------------------------------------------
# The real destructive shape: incident t_36f4e9c8, 2026-08-03 10:47-10:53 KST,
# session 20260803_102050_e0c3dd74, messages 275947-276085.
#
# The model re-sent the ENTIRE cronjob schema with type-default empties while
# only meaning to change `deliver`. The update branch's `is not None` sentinel
# read every one of those empties as an explicit clear, so 43 jobs lost prompt,
# name, skills, script, enabled_toolsets, workdir and context_from in one pass.
#
# These tests replay that literal argument set through the tool boundary --
# not through update_job() -- because that is the layer the incident went
# through and the layer where the empties become "edits".
# ---------------------------------------------------------------------------

# Verbatim keys from message 275947 (job b80587becc0a), job_id substituted.
DESTRUCTIVE_UPDATE_ARGS = {
    "action": "update",
    "prompt": "",
    "schedule": "0 10 * * *",
    "name": "",
    "repeat": 0,
    "deliver": "whatsapp",
    "skills": [],
    "script": "",
    "no_agent": False,
    "context_from": [],
    "enabled_toolsets": [],
    "workdir": "",
    "attach_to_session": False,
}


def _cronjob(**kwargs):
    import json as _json
    from tools.cronjob_tools import cronjob

    return _json.loads(cronjob(**kwargs))


def test_tool_update_rejects_the_2026_08_03_destructive_shape(hermes_env):
    """The exact call that wiped 43 jobs must now fail closed."""
    from cron.jobs import create_job, get_job

    job = create_job(
        prompt="Check for agent-browser updates with a 14-day safety buffer.",
        schedule="0 10 * * *",
        name="agent-browser safe auto-update",
        skills=["agent-browser"],
        deliver="local",
    )
    before = dict(get_job(job["id"]))

    result = _cronjob(job_id=job["id"], **DESTRUCTIVE_UPDATE_ARGS)

    assert result["success"] is False
    assert "nothing to run" in result["error"]

    # Atomicity: a guard that raises after a partial save is worse than none.
    after = get_job(job["id"])
    for field in ("prompt", "name", "skills", "skill", "script",
                  "enabled_toolsets", "workdir", "context_from",
                  "schedule", "deliver", "enabled", "state"):
        assert after.get(field) == before.get(field), f"{field} was clobbered"


def test_tool_update_rejects_the_destructive_shape_on_a_script_job(hermes_env):
    """script:"" + prompt:"" + skills:[] empties an agent script job too."""
    from cron.jobs import create_job, get_job

    (hermes_env / "scripts" / "w.sh").write_text("echo hi\n")
    job = create_job(
        prompt=None, schedule="0 2 * * *", script="w.sh",
        name="Daily Wiki Backup", deliver="local",
    )
    before = dict(get_job(job["id"]))

    result = _cronjob(job_id=job["id"], **dict(DESTRUCTIVE_UPDATE_ARGS,
                                              schedule="0 2 * * *"))

    assert result["success"] is False
    assert "nothing to run" in result["error"]
    assert get_job(job["id"]) == before


def test_tool_update_rejects_the_destructive_shape_on_a_no_agent_job(hermes_env):
    """no_agent job: the more specific no_agent diagnosis reports first."""
    from cron.jobs import create_job, get_job

    (hermes_env / "scripts" / "w.sh").write_text("echo hi\n")
    job = create_job(
        prompt=None, schedule="0 8 * * *", script="w.sh", no_agent=True,
        name="Lil'Log RSS ingest watchdog", deliver="local",
    )
    before = dict(get_job(job["id"]))

    result = _cronjob(job_id=job["id"], **dict(DESTRUCTIVE_UPDATE_ARGS,
                                              schedule="0 8 * * *",
                                              no_agent=True))

    assert result["success"] is False
    assert "script" in result["error"]
    assert get_job(job["id"]) == before


def test_tool_update_blank_name_is_a_no_op(hermes_env):
    """`name` is identity, not payload — no empty-payload guard covers it.

    A job that keeps a script survives the payload guard, so without this the
    same bulk-empty update still silently erases the name (which is exactly
    what made 43 corrupted jobs unidentifiable in jobs.json).
    """
    from cron.jobs import create_job, get_job

    (hermes_env / "scripts" / "w.sh").write_text("echo hi\n")
    job = create_job(
        prompt=None, schedule="0 2 * * *", script="w.sh",
        name="Daily Wiki Backup", deliver="local",
    )

    result = _cronjob(job_id=job["id"], action="update", name="", deliver="local")

    assert result["success"] is True
    assert get_job(job["id"])["name"] == "Daily Wiki Backup"


def test_tool_update_still_renames_when_a_real_name_is_given(hermes_env):
    from cron.jobs import create_job, get_job

    job = create_job(prompt="check the news", schedule="every 5m",
                     name="old", deliver="local")

    result = _cronjob(job_id=job["id"], action="update", name="new")

    assert result["success"] is True
    assert get_job(job["id"])["name"] == "new"


def test_tool_update_blank_scalars_still_clear_workdir_and_context_from(hermes_env):
    """KNOWN HOLE, pinned deliberately — not an endorsement.

    ``workdir:""`` / ``context_from:[]`` / ``enabled_toolsets:[]`` are
    documented clears, so a bulk-empty update that survives the payload guard
    (because it keeps a script) still silently drops all three. Closing this
    means changing documented semantics; recorded as a finding in
    RECOVERY_REPORT.md instead. This test exists so the behaviour cannot change
    unnoticed.
    """
    from cron.jobs import create_job, get_job

    (hermes_env / "scripts" / "w.sh").write_text("echo hi\n")
    (hermes_env / "wd").mkdir()
    job = create_job(
        prompt=None, schedule="0 2 * * *", script="w.sh", name="keeper",
        enabled_toolsets=["file"], workdir=str(hermes_env / "wd"),
        deliver="local",
    )
    assert get_job(job["id"])["workdir"] == str(hermes_env / "wd")

    result = _cronjob(job_id=job["id"], action="update", script="w.sh",
                      workdir="", enabled_toolsets=[], context_from=[])

    assert result["success"] is True
    stored = get_job(job["id"])
    assert stored["name"] == "keeper"          # protected by the fix above
    assert stored["script"] == "w.sh"          # payload survives
    assert stored["workdir"] is None           # still clobbered
    assert stored["enabled_toolsets"] is None  # still clobbered
    assert stored["context_from"] is None      # still clobbered
