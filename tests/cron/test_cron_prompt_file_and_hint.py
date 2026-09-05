"""Cron ``prompt_file`` and ``no_cron_hint`` (per-job opt-outs).

Two small, user-owned (CLI/programmatic) knobs:

* ``prompt_file`` -- point a job at a file whose contents ARE the prompt, read
  fresh at each fire so edits to the file are picked up (single source of
  truth for a routine that lives in a git-backed vault). Makes an
  otherwise-promptless job runnable.
* ``no_cron_hint`` -- suppress the always-prepended cron execution-guidance
  banner for a job that does not want it (e.g. one that always delivers via an
  attached session and never uses the [SILENT] convention).

Both follow the ``reasoning_effort`` precedent: conditional-persist (absent =
byte-identical to pre-feature jobs) and deliberately NOT model-settable.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
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
# prompt_file: create-time
# ---------------------------------------------------------------------------


def test_prompt_file_makes_a_promptless_job_valid(hermes_env):
    from cron.jobs import create_job

    routine = hermes_env / "routine.md"
    routine.write_text("Run the daily round.\n")

    job = create_job(
        prompt=None, schedule="every 5m", prompt_file=str(routine), deliver="local"
    )
    assert job["prompt_file"] == str(routine)


def test_prompt_file_missing_is_rejected_at_create(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="prompt_file"):
        create_job(
            prompt=None,
            schedule="every 5m",
            prompt_file=str(hermes_env / "does-not-exist.md"),
            deliver="local",
        )


def test_prompt_file_absent_when_unset(hermes_env):
    """Byte-identical to pre-feature jobs: no key when not provided."""
    from cron.jobs import create_job

    job = create_job(prompt="hi", schedule="every 5m", deliver="local")
    assert "prompt_file" not in job


def test_prompt_file_is_stored_as_an_absolute_path(hermes_env, monkeypatch):
    """A relative path validated at create must resolve to the same file at
    fire time (scheduler runs under its own cwd), so it is stored absolute."""
    from cron.jobs import create_job

    routine = hermes_env / "routine.md"
    routine.write_text("do it\n")
    # Create with a RELATIVE path from inside the file's directory.
    monkeypatch.chdir(hermes_env)
    job = create_job(
        prompt=None, schedule="every 5m", prompt_file="routine.md", deliver="local"
    )
    assert job["prompt_file"] == str(routine.resolve())


def test_prompt_file_only_job_is_not_treated_as_empty(hermes_env):
    """The run-time empty-payload guard (job_payload_is_empty) must count
    prompt_file as payload, or a prompt_file-only job gets auto-paused."""
    from cron.jobs import create_job, job_payload_is_empty

    routine = hermes_env / "routine.md"
    routine.write_text("run the round\n")
    job = create_job(
        prompt=None, schedule="every 5m", prompt_file=str(routine), deliver="local"
    )
    assert job_payload_is_empty(job) is False


# ---------------------------------------------------------------------------
# prompt_file: read at FIRE time (picks up edits)
# ---------------------------------------------------------------------------


def test_prompt_file_content_is_used_as_the_prompt(hermes_env):
    from cron.jobs import create_job
    from cron.scheduler import _build_job_prompt

    routine = hermes_env / "routine.md"
    routine.write_text("Execute step one.\n")
    job = create_job(
        prompt=None, schedule="every 5m", prompt_file=str(routine), deliver="local"
    )

    built = _build_job_prompt(job)
    assert "Execute step one." in built


def test_prompt_file_is_reread_each_fire(hermes_env):
    """Edits to the file land on the next run without re-creating the job."""
    from cron.jobs import create_job
    from cron.scheduler import _build_job_prompt

    routine = hermes_env / "routine.md"
    routine.write_text("first version\n")
    job = create_job(
        prompt=None, schedule="every 5m", prompt_file=str(routine), deliver="local"
    )
    assert "first version" in _build_job_prompt(job)

    routine.write_text("second version\n")
    assert "second version" in _build_job_prompt(job)


def test_inline_prompt_wins_over_prompt_file(hermes_env):
    """An explicit prompt takes precedence; prompt_file is the fallback."""
    from cron.jobs import create_job
    from cron.scheduler import _build_job_prompt

    routine = hermes_env / "routine.md"
    routine.write_text("from file\n")
    job = create_job(
        prompt="from inline",
        schedule="every 5m",
        prompt_file=str(routine),
        deliver="local",
    )

    built = _build_job_prompt(job)
    assert "from inline" in built
    assert "from file" not in built


# ---------------------------------------------------------------------------
# no_cron_hint: suppress the execution-guidance banner
# ---------------------------------------------------------------------------

_HINT_MARK = "running as a scheduled cron job"


def test_cron_hint_is_present_by_default(hermes_env):
    from cron.jobs import create_job
    from cron.scheduler import _build_job_prompt

    job = create_job(prompt="do the thing", schedule="every 5m", deliver="local")
    assert _HINT_MARK in _build_job_prompt(job)
    assert "prompt_file" not in job  # unrelated defaults untouched


def test_no_cron_hint_suppresses_the_banner(hermes_env):
    from cron.jobs import create_job
    from cron.scheduler import _build_job_prompt

    job = create_job(
        prompt="do the thing",
        schedule="every 5m",
        no_cron_hint=True,
        deliver="local",
    )
    assert job["no_cron_hint"] is True
    built = _build_job_prompt(job)
    assert _HINT_MARK not in built
    assert "do the thing" in built


def test_no_cron_hint_absent_when_unset(hermes_env):
    from cron.jobs import create_job

    job = create_job(prompt="hi", schedule="every 5m", deliver="local")
    assert "no_cron_hint" not in job


# ---------------------------------------------------------------------------
# Not model-settable (reasoning_effort precedent)
# ---------------------------------------------------------------------------


def test_prompt_file_and_no_cron_hint_are_not_in_the_model_schema():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    props = CRONJOB_SCHEMA["parameters"]["properties"]
    assert "prompt_file" not in props
    assert "no_cron_hint" not in props


# ---------------------------------------------------------------------------
# End-to-end through the tool boundary (the CLI create path)
# ---------------------------------------------------------------------------


def test_tool_create_with_prompt_file_and_no_cron_hint(hermes_env):
    import json
    from tools.cronjob_tools import cronjob
    from cron.scheduler import _build_job_prompt
    from cron.jobs import get_job

    routine = hermes_env / "routine.md"
    routine.write_text("Run the vault round.\n")

    result = json.loads(
        cronjob(
            action="create",
            schedule="every day at 08:00",
            prompt_file=str(routine),
            no_cron_hint=True,
            deliver="local",
        )
    )
    assert result["success"] is True

    job = get_job(result["job_id"])
    assert job["prompt_file"] == str(routine)
    assert job["no_cron_hint"] is True

    built = _build_job_prompt(job)
    assert "Run the vault round." in built
    assert _HINT_MARK not in built


def test_tool_create_rejects_missing_prompt_file(hermes_env):
    import json
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            prompt_file=str(hermes_env / "nope.md"),
            deliver="local",
        )
    )
    assert result["success"] is False
    assert "prompt_file" in result["error"]


def test_cli_create_parser_accepts_the_new_flags():
    """The argparse wiring exists and maps to the right dests."""
    import argparse
    from hermes_cli.subcommands.cron import build_cron_parser

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    build_cron_parser(sub, cmd_cron=lambda args: 0)

    args = parser.parse_args(
        ["cron", "create", "every 5m", "--prompt-file", "/tmp/r.md", "--no-cron-hint"]
    )
    assert args.prompt_file == "/tmp/r.md"
    assert args.no_cron_hint is True


def test_run_job_errors_when_prompt_file_vanishes_before_fire(hermes_env):
    """A prompt_file-only job whose file is deleted after create must fail the
    run loudly, not fire an empty prompt (wasted unattended model call)."""
    from unittest.mock import patch

    import cron.scheduler as scheduler
    from cron.jobs import create_job

    routine = hermes_env / "routine.md"
    routine.write_text("run the round\n")
    job = create_job(
        prompt=None, schedule="every 5m", prompt_file=str(routine), deliver="local"
    )
    routine.unlink()  # gone before the next fire

    class _Boom:
        def __init__(self, *a, **kw):  # pragma: no cover - must never run
            raise AssertionError("agent must not be constructed on unreadable prompt_file")

    with patch("run_agent.AIAgent", _Boom):
        success, doc, final, error = scheduler.run_job(job)

    assert success is False
    assert "prompt_file" in error
    assert "ERROR" in doc
