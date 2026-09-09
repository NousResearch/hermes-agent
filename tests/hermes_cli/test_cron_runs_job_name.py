"""Regression: `hermes cron runs <name>` must accept a job NAME, not just an ID.

Every other job-targeting cron command (pause, resume, remove, edit, run)
resolves a reference through cron.jobs.resolve_job_ref, so a name works.
`cron runs` passed its argument straight to the executions query as a literal
job_id, so a name matched no rows.

The failure mode is worse than "unsupported": the query legitimately returned
zero rows, and the command reports zero rows as "No cron execution attempts
recorded." — the same message a real, never-fired job produces. A user
checking whether their job ran is told it never did, while the database holds
hundreds of completed executions for it.
"""

import json

import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def cron_home(monkeypatch):
    """A temp HERMES_HOME with one job and one recorded execution."""
    home = Path(tempfile.mkdtemp())
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Drop ONLY the modules that resolve HERMES_HOME at import time, so they
    # re-read the temp home above.
    #
    # An earlier version purged every module matching "cron." or "hermes_cli."
    # and leaked into unrelated suites: tests/hermes_cli/test_web_server_cron_profiles.py
    # holds a monkeypatched `hermes_cli.profiles`, and evicting it mid-session
    # handed the next test a freshly imported copy without those patches. Two
    # of its cases then failed only when run in the same batch as this file —
    # a failure with no connection to the code under test, which is exactly
    # the kind of cross-test pollution that wastes an afternoon.
    for mod in ("cron.executions", "cron.jobs", "cron.notepad", "hermes_cli.cron"):
        sys.modules.pop(mod, None)

    (home / "cron").mkdir(parents=True, exist_ok=True)

    from cron import executions as ex
    from cron import jobs as cron_jobs

    job = {
        "id": "43f8195c7f94",
        "name": "brain-sync",
        "schedule": {"kind": "cron", "expr": "*/20 * * * *", "display": "*/20 * * * *"},
        "prompt": "",
        "enabled": True,
    }
    jobs_path = home / "cron" / "jobs.json"
    jobs_path.write_text(json.dumps({"jobs": [job]}), encoding="utf-8")

    execution = ex.create_execution("43f8195c7f94", source="builtin")
    ex.finish_execution(execution["id"], success=True)

    yield {"home": home, "job": job, "jobs": cron_jobs, "executions": ex}


def test_runs_accepts_job_name(cron_home, capsys):
    """The whole point: a name is a valid reference everywhere else."""
    from hermes_cli.cron import cron_runs

    cron_runs("brain-sync")
    out = capsys.readouterr().out

    assert "No cron execution attempts recorded" not in out, (
        "a name that resolves to a job with executions must not report an empty history"
    )
    assert "43f8195c7f94" in out


def test_runs_still_accepts_job_id(cron_home, capsys):
    """Resolution must not break the documented ID path."""
    from hermes_cli.cron import cron_runs

    cron_runs("43f8195c7f94")
    out = capsys.readouterr().out
    assert "43f8195c7f94" in out
    assert "No cron execution attempts recorded" not in out


def test_unknown_job_is_distinguished_from_empty_history(cron_home, capsys):
    """'No such job' and 'job never ran' are different facts.

    Collapsing them is what made this bug invisible: the user reads "never
    ran" and goes looking for a scheduler problem that does not exist.
    """
    from hermes_cli.cron import cron_runs

    cron_runs("job-que-nao-existe")
    out = capsys.readouterr().out.lower()

    assert "not found" in out or "nao encontrado" in out or "no such" in out, (
        f"an unknown reference must say so; got: {out!r}"
    )


def test_history_of_a_deleted_job_is_still_reachable(cron_home, capsys):
    """The ledger outlives the job, and that is the point.

    Executions are not deleted with their job, and forensics after a removal
    is exactly when this command earns its keep. Resolution must therefore
    enrich the lookup, never gate it: an id with rows in the ledger prints
    them even though no job answers to it any more.
    """
    from hermes_cli.cron import cron_runs

    ex = cron_home["executions"]
    row = ex.create_execution("job-ja-removido", source="builtin")
    ex.finish_execution(row["id"], success=False, error="boom")

    cron_runs("job-ja-removido")
    out = capsys.readouterr().out

    assert row["id"] in out, f"history of a deleted job must survive; got: {out!r}"
    assert "boom" in out


def test_known_job_without_executions_reports_empty_history(cron_home, capsys):
    """The converse: a real job that never fired still reports an empty history."""
    from cron.jobs import save_jobs, load_jobs
    from hermes_cli.cron import cron_runs

    jobs = load_jobs()
    jobs.append({
        "id": "aaaabbbbcccc",
        "name": "never-ran",
        "schedule": {"kind": "cron", "expr": "0 9 * * *", "display": "0 9 * * *"},
        "prompt": "",
        "enabled": True,
    })
    save_jobs(jobs)

    cron_runs("never-ran")
    out = capsys.readouterr().out

    assert "No cron execution attempts recorded" in out


def test_no_argument_lists_every_job(cron_home, capsys):
    """`hermes cron runs` with no argument keeps listing all executions."""
    from hermes_cli.cron import cron_runs

    cron_runs(None)
    out = capsys.readouterr().out
    assert "43f8195c7f94" in out


def test_ambiguous_name_lists_candidates(cron_home, capsys):
    """Two jobs sharing a name must not resolve silently to one of them."""
    from cron.jobs import load_jobs, save_jobs
    from hermes_cli.cron import cron_runs

    jobs = load_jobs()
    jobs.append({
        "id": "ddddeeeeffff",
        "name": "brain-sync",  # same name, different id
        "schedule": {"kind": "cron", "expr": "0 9 * * *", "display": "0 9 * * *"},
        "prompt": "",
        "enabled": True,
    })
    save_jobs(jobs)

    cron_runs("brain-sync")
    out = capsys.readouterr().out

    assert "43f8195c7f94" in out and "ddddeeeeffff" in out, (
        f"an ambiguous name must surface every candidate id; got: {out!r}"
    )


# --- notepad: same bug class, worse consequence -----------------------------
#
# The notepad is keyed by job id and the scheduler reads it back with
# job["id"] (cron/scheduler.py::render_notepad_section). The CLI used the raw
# argument, so `hermes cron notepad <name> set k v` wrote under the name and
# the scheduler — reading under the id — never saw it. Nothing errored; the
# note simply never reached the job it was written for.


class _Args:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_notepad_write_by_name_is_readable_by_the_scheduler(cron_home, capsys):
    from cron import notepad
    from hermes_cli.cron import cron_notepad

    rc = cron_notepad(_Args(
        job_id="brain-sync", notepad_action="set", key="ultimo", value="ok",
    ))
    assert rc == 0

    # The scheduler's view: keyed by the canonical id, never by the name.
    assert notepad.get_note("43f8195c7f94", "ultimo") == "ok", (
        "a note written by name must land under the id the scheduler reads"
    )


def test_notepad_unknown_job_keeps_working_by_raw_id(cron_home):
    """An unresolvable reference is used as-is, not refused.

    Notepad rows outlive the job, and a running cron agent writes to its own
    notepad by id through this CLI. Failing closed here would break that
    write path for any job the resolver cannot see (another profile's job, a
    job removed mid-run), turning a convenience into an outage.
    """
    from cron import notepad
    from hermes_cli.cron import cron_notepad

    rc = cron_notepad(_Args(
        job_id="job-de-outro-profile", notepad_action="set", key="k", value="v",
    ))

    assert rc == 0
    assert notepad.get_note("job-de-outro-profile", "k") == "v"


def test_notepad_still_accepts_the_canonical_id(cron_home):
    from cron import notepad
    from hermes_cli.cron import cron_notepad

    rc = cron_notepad(_Args(
        job_id="43f8195c7f94", notepad_action="set", key="via_id", value="1",
    ))
    assert rc == 0
    assert notepad.get_note("43f8195c7f94", "via_id") == "1"
