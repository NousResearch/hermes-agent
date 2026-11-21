"""`hermes cron runs` and `hermes cron notepad` accept a job NAME, not only its id.

The notepad case is not cosmetic: the scheduler reads a job's notepad with
``job["id"]`` (``render_notepad_section``), so a name stored verbatim wrote to a key
nothing ever reads — ``notepad <name> set cursor 42`` reported success and the
state was silently lost.

Resolution ENRICHES, never blocks: notepads and the execution ledger outlive their
job, so an argument that resolves to no job is still used as a raw id.
"""

from __future__ import annotations

import argparse

import pytest


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    import cron.notepad as notepad_mod

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    monkeypatch.setattr(notepad_mod, "NOTEPAD_FILE", hermes_home / "cron" / "notepad.db")
    return jobs_mod, notepad_mod


def _ns(**kw):
    base = dict(job_id=None, notepad_action="list", key=None, value=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_notepad_set_by_name_lands_on_the_key_the_scheduler_reads(cron_env):
    jobs_mod, notepad_mod = cron_env
    job = jobs_mod.create_job(prompt="Check the feed", schedule="every 1h", name="feed-check")
    from hermes_cli.cron import cron_notepad

    assert cron_notepad(_ns(job_id="feed-check", notepad_action="set", key="cursor", value="42")) == 0

    assert notepad_mod.get_note(job["id"], "cursor") == "42"
    assert notepad_mod.get_note("feed-check", "cursor") is None


def test_notepad_for_a_deleted_job_still_reads_by_raw_id(cron_env):
    """The notepad outlives its job; an unresolvable argument is used as the id."""
    _jobs_mod, notepad_mod = cron_env
    notepad_mod.set_note("job-gone", "cursor", "7")
    from hermes_cli.cron import cron_notepad

    assert cron_notepad(_ns(job_id="job-gone", notepad_action="get", key="cursor")) == 0


def test_runs_by_name_reads_the_ledger_the_id_indexes(cron_env, capsys):
    jobs_mod, _notepad_mod = cron_env
    job = jobs_mod.create_job(prompt="Check the feed", schedule="every 1h", name="feed-check")
    from cron import executions
    from hermes_cli.cron import cron_runs

    rec = executions.create_execution(job["id"], source="builtin")
    executions.finish_execution(rec["id"], success=True)

    cron_runs("feed-check")

    out = capsys.readouterr().out
    assert rec["id"] in out
    assert "No cron execution attempts recorded" not in out


def test_runs_names_a_wrong_argument_instead_of_reporting_no_history(cron_env, capsys):
    """Nothing resolved AND nothing in the ledger is a typo, not a job that never ran."""
    from hermes_cli.cron import cron_runs

    cron_runs("no-such-job")

    assert "Job not found: no-such-job" in capsys.readouterr().out
