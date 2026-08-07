"""Per-job durable cron notepad: KV scratchpad surviving scheduled wake-ups.

Covers CRUD on the SQLite-backed store, size-cap enforcement, prompt
injection of non-empty notepads, byte-stable prompts for jobs that don't
use the notepad, and the `hermes cron notepad` CLI handler.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def notepad(monkeypatch, tmp_path):
    import cron.notepad as notepad_mod

    monkeypatch.setattr(
        notepad_mod, "NOTEPAD_FILE", tmp_path / "cron" / "notepad.db"
    )
    return notepad_mod


@pytest.fixture
def cron_env(tmp_path, monkeypatch, notepad):
    """Isolated cron environment with temp HERMES_HOME (mirrors test_cron_context_from)."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    return hermes_home


class TestNotepadCrud:
    def test_set_then_get_roundtrip(self, notepad):
        notepad.set_note("job-1", "cursor", "page=7")
        assert notepad.get_note("job-1", "cursor") == "page=7"

    def test_get_missing_key_returns_none(self, notepad):
        assert notepad.get_note("job-1", "nope") is None

    def test_set_overwrites_existing_value(self, notepad):
        notepad.set_note("job-1", "cursor", "page=1")
        notepad.set_note("job-1", "cursor", "page=2")
        assert notepad.get_note("job-1", "cursor") == "page=2"
        assert len(notepad.list_notes("job-1")) == 1

    def test_delete_removes_key(self, notepad):
        notepad.set_note("job-1", "cursor", "page=1")
        assert notepad.delete_note("job-1", "cursor") is True
        assert notepad.get_note("job-1", "cursor") is None

    def test_delete_missing_key_returns_false(self, notepad):
        assert notepad.delete_note("job-1", "ghost") is False

    def test_list_notes_sorted_by_key(self, notepad):
        notepad.set_note("job-1", "beta", "2")
        notepad.set_note("job-1", "alpha", "1")
        notes = notepad.list_notes("job-1")
        assert [n["key"] for n in notes] == ["alpha", "beta"]
        assert all(n["updated_at"] for n in notes)

    def test_jobs_are_isolated(self, notepad):
        notepad.set_note("job-1", "k", "one")
        notepad.set_note("job-2", "k", "two")
        assert notepad.get_note("job-1", "k") == "one"
        assert notepad.get_note("job-2", "k") == "two"
        assert len(notepad.list_notes("job-1")) == 1

    def test_survives_new_connection(self, notepad):
        """Durability: values persist across independent calls (fresh connections)."""
        notepad.set_note("job-1", "state", "persisted")
        # Every public call opens its own connection, so a second read
        # after the writer's connection closed proves on-disk durability.
        assert notepad.get_note("job-1", "state") == "persisted"
        assert notepad.list_notes("job-1")[0]["value"] == "persisted"

    def test_clear_notepad_removes_all_keys_for_job_only(self, notepad):
        notepad.set_note("job-1", "a", "1")
        notepad.set_note("job-1", "b", "2")
        notepad.set_note("job-2", "a", "keep")
        assert notepad.clear_notepad("job-1") == 2
        assert notepad.list_notes("job-1") == []
        assert notepad.get_note("job-2", "a") == "keep"


class TestNotepadCaps:
    def test_value_over_per_key_cap_rejected(self, notepad):
        big = "x" * (notepad.MAX_VALUE_BYTES + 1)
        with pytest.raises(ValueError, match="value too large"):
            notepad.set_note("job-1", "big", big)
        assert notepad.get_note("job-1", "big") is None

    def test_value_at_cap_accepted(self, notepad):
        exact = "x" * notepad.MAX_VALUE_BYTES
        notepad.set_note("job-1", "exact", exact)
        assert notepad.get_note("job-1", "exact") == exact

    def test_key_over_cap_rejected(self, notepad):
        with pytest.raises(ValueError, match="key too long"):
            notepad.set_note("job-1", "k" * (notepad.MAX_KEY_CHARS + 1), "v")

    def test_empty_key_rejected(self, notepad):
        with pytest.raises(ValueError, match="key must be non-empty"):
            notepad.set_note("job-1", "", "v")

    def test_per_job_total_cap_enforced(self, notepad, monkeypatch):
        monkeypatch.setattr(notepad, "MAX_JOB_TOTAL_BYTES", 100)
        notepad.set_note("job-1", "a", "x" * 60)
        with pytest.raises(ValueError, match="notepad full"):
            notepad.set_note("job-1", "b", "x" * 60)
        # Overwriting the existing key within budget still works.
        notepad.set_note("job-1", "a", "x" * 90)
        assert notepad.get_note("job-1", "a") == "x" * 90

    def test_total_cap_counts_only_this_job(self, notepad, monkeypatch):
        monkeypatch.setattr(notepad, "MAX_JOB_TOTAL_BYTES", 100)
        notepad.set_note("other-job", "a", "x" * 90)
        notepad.set_note("job-1", "a", "x" * 90)  # must not raise
        assert notepad.get_note("job-1", "a") == "x" * 90


class TestPromptInjection:
    def test_nonempty_notepad_rendered_into_prompt(self, cron_env, notepad):
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        job = create_job(prompt="Check the feed", schedule="every 1h")
        notepad.set_note(job["id"], "last_seen_id", "8842")
        notepad.set_note(job["id"], "watchlist", "alpha, beta")

        prompt = _build_job_prompt(job)
        assert "Job notepad (persistent across runs)" in prompt
        assert "last_seen_id" in prompt
        assert "8842" in prompt
        assert "watchlist" in prompt
        # The injected section documents the CLI write path for this job.
        assert f"hermes cron notepad {job['id']} set" in prompt

    def test_empty_notepad_prompt_byte_stable(self, cron_env, notepad):
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        job = create_job(prompt="Check the feed", schedule="every 1h")
        baseline = _build_job_prompt(job)

        # A different job's notes must not leak in; a set+delete cycle on
        # this job must leave the prompt byte-identical.
        notepad.set_note("ffffffffffff", "k", "other job data")
        notepad.set_note(job["id"], "tmp", "scratch")
        notepad.delete_note(job["id"], "tmp")

        assert _build_job_prompt(job) == baseline
        assert "notepad" not in baseline.lower()

    def test_notepad_read_failure_does_not_break_prompt(self, cron_env, notepad, monkeypatch):
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        job = create_job(prompt="Check the feed", schedule="every 1h")

        def _boom(_job_id):
            raise RuntimeError("db unavailable")

        monkeypatch.setattr(notepad, "list_notes", _boom)
        prompt = _build_job_prompt(job)
        assert "Check the feed" in prompt
        assert "notepad" not in prompt.lower()


class TestNotepadCli:
    def _ns(self, **kw):
        base = dict(job_id=None, notepad_action="list", key=None, value=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_cli_set_get_list_delete(self, notepad, capsys):
        from hermes_cli.cron import cron_notepad

        assert cron_notepad(self._ns(job_id="job-1", notepad_action="set", key="cursor", value="42")) == 0
        assert notepad.get_note("job-1", "cursor") == "42"

        assert cron_notepad(self._ns(job_id="job-1", notepad_action="get", key="cursor")) == 0
        assert "42" in capsys.readouterr().out

        assert cron_notepad(self._ns(job_id="job-1", notepad_action="list")) == 0
        assert "cursor" in capsys.readouterr().out

        assert cron_notepad(self._ns(job_id="job-1", notepad_action="delete", key="cursor")) == 0
        assert notepad.get_note("job-1", "cursor") is None

    def test_cli_get_missing_key_exits_nonzero(self, notepad, capsys):
        from hermes_cli.cron import cron_notepad

        assert cron_notepad(self._ns(job_id="job-1", notepad_action="get", key="ghost")) == 1

    def test_cli_set_requires_key_and_value(self, notepad, capsys):
        from hermes_cli.cron import cron_notepad

        assert cron_notepad(self._ns(job_id="job-1", notepad_action="set", key="k")) == 1
        assert cron_notepad(self._ns(job_id="job-1", notepad_action="set")) == 1

    def test_cli_set_oversized_value_reports_error(self, notepad, capsys):
        from hermes_cli.cron import cron_notepad

        big = "x" * (notepad.MAX_VALUE_BYTES + 1)
        assert cron_notepad(self._ns(job_id="job-1", notepad_action="set", key="k", value=big)) == 1
        assert "too large" in capsys.readouterr().out.lower()

    def test_cron_command_dispatches_notepad(self, notepad):
        from hermes_cli.cron import cron_command

        ns = self._ns(job_id="job-9", notepad_action="set", key="k", value="v")
        ns.cron_command = "notepad"
        assert cron_command(ns) == 0
        assert notepad.get_note("job-9", "k") == "v"
