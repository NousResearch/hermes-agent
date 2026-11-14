"""Regression for #153042 — one transient sharing conflict on jobs.json lost a whole cron run.

On Windows the scheduler replaces ``jobs.json`` through mkstemp+rename while job processes read it,
and a reader whose ``open`` lands inside that window gets ``PermissionError`` (``winerror 32`` ->
EACCES). ``load_jobs`` had no retry: it re-raised as ``RuntimeError: Failed to read cron database``,
the execution was booked ``failed`` after 1.4 s and never re-attempted, while two jobs fired in the
same second completed and an immediate re-read was fine.

The read now retries a bounded set of transient errnos (``cron.jobs_read``) before the caller keeps
its "repair vs. bail" decision. ``test_load_jobs_survives_a_transient_permission_error`` and
``test_load_jobs_still_refuses_after_the_retries_run_out`` fail on ``upstream/main`` — the first
with the reported ``RuntimeError``, the second on the attempt count (1 there, 3 here).
"""

from __future__ import annotations

import builtins
import errno
import json

import pytest

from cron import jobs, jobs_read


class _JobsFileOpen:
    """``builtins.open`` stand-in that fails the jobs.json read with scripted errors, then delegates.

    ``errors`` is consumed one entry per jobs.json open: an exception instance is raised, ``None``
    means "let the real open through". Writes and every other path are unaffected, so the canonical
    save that builds the fixture keeps working.
    """

    def __init__(self, *errors: BaseException | None):
        self.errors = list(errors)
        self.attempts = 0
        # Bind the real opener now: ``builtins.open`` is what the fixture replaces, so looking it up
        # at call time would recurse into this stand-in.
        self._real_open = builtins.open

    def __call__(self, file, *args, **kwargs):
        if str(file).endswith("jobs.json"):
            self.attempts += 1
            error = self.errors.pop(0) if self.errors else None
            if error is not None:
                raise error
        return self._real_open(file, *args, **kwargs)


@pytest.fixture
def cron_job(tmp_path):
    """One real job in a scoped cron store, plus its id."""
    with jobs.use_cron_store(tmp_path / "home"):
        job = jobs.create_job(prompt="ping", schedule="every 2m", deliver="local")
        yield job["id"]


def test_load_jobs_survives_a_transient_permission_error(cron_job, monkeypatch):
    """A PermissionError on the first two puts is retried, so the run keeps its job records."""
    opener = _JobsFileOpen(
        PermissionError(errno.EACCES, "Permission denied"),
        PermissionError(errno.EACCES, "Permission denied"),
        None,
    )
    monkeypatch.setattr(builtins, "open", opener)
    loaded = jobs.load_jobs()
    assert [row["id"] for row in loaded] == [cron_job]
    assert opener.attempts == 3, "the read must retry, not bail on the first put"


def test_load_jobs_still_refuses_after_the_retries_run_out(cron_job, monkeypatch):
    """Bounded: a real permission problem still fails the load, after the attempts are spent."""
    opener = _JobsFileOpen(*[PermissionError(errno.EACCES, "Permission denied")] * 10)
    monkeypatch.setattr(builtins, "open", opener)
    with pytest.raises(RuntimeError, match="Failed to read cron database"):
        jobs.load_jobs()
    assert opener.attempts == jobs_read.READ_ATTEMPTS


def test_load_jobs_does_not_retry_an_unexpected_oserror(cron_job, monkeypatch):
    """A dead disk is not a sharing conflict: refuse at once instead of sleeping on it."""
    opener = _JobsFileOpen(OSError(errno.EIO, "Input/output error"))
    monkeypatch.setattr(builtins, "open", opener)
    with pytest.raises(RuntimeError, match="Failed to read cron database"):
        jobs.load_jobs()
    assert opener.attempts == 1


def test_parse_jobs_file_backs_off_between_attempts(tmp_path, monkeypatch):
    """The retry waits 50 ms then 100 ms — short enough that a real refusal is not stalled."""
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(json.dumps({"jobs": []}), encoding="utf-8")
    sleeps: list[float] = []
    opener = _JobsFileOpen(
        PermissionError(errno.EACCES, "Permission denied"),
        PermissionError(errno.EACCES, "Permission denied"),
        None,
    )
    monkeypatch.setattr("cron.jobs_read.open", opener, raising=False)
    data, strict = jobs_read.parse_jobs_file(jobs_file, sleep=sleeps.append)
    assert strict is False and data == {"jobs": []}
    assert opener.attempts == 3
    assert sleeps == [jobs_read.RETRY_BASE_SECONDS, jobs_read.RETRY_BASE_SECONDS * 2]


def test_the_tolerant_parse_still_absorbs_a_bom_and_a_stray_control_character(tmp_path):
    """The move must not drop the two repairs the old inline body performed."""
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_bytes(b"\xef\xbb\xbf" + b'{"jobs": [{"id": "a", "prompt": "tab\there"}]}')
    data, strict = jobs_read.parse_jobs_file(jobs_file)
    assert strict is True, "a raw control character in a string still needs the strict=False pass"
    assert data["jobs"][0]["id"] == "a"
