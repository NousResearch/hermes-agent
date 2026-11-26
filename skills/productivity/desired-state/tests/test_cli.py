"""Tests for ds.py — the CLI dispatcher: exit codes, text + JSON surfaces."""

from __future__ import annotations

import json

import pytest

import ds


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point the store at a tmp HERMES_HOME for the whole CLI surface."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _define(extra=None):
    argv = [
        "define", "finance", "Hit 15% savings rate",
        "--direction", "increase", "--baseline", "9", "--current", "9",
        "--target", "15", "--unit", "%", "--start-date", "2026-01-01",
        "--target-date", "2026-12-31",
    ]
    return ds.main(argv + (extra or []))


class TestDefineAndTrack:
    def test_define_creates_file(self, home, capsys):
        assert _define() == 0
        out = capsys.readouterr().out
        assert "defined finance/hit-15-savings-rate" in out
        assert (home / "state" / "desired" / "finance" / "hit-15-savings-rate.md").exists()

    def test_define_duplicate_exits_2(self, home, capsys):
        assert _define() == 0
        assert _define() == 2
        assert "already exists" in capsys.readouterr().err

    def test_track_updates_and_reports(self, home, capsys):
        _define()
        capsys.readouterr()
        assert ds.main(["track", "finance", "hit-15-savings-rate", "12.4"]) == 0
        out = capsys.readouterr().out
        assert "tracked" in out and "%" in out


class TestReads:
    def test_gap_json_surface(self, home, capsys):
        _define()
        capsys.readouterr()
        rc = ds.main(["--json", "gap", "finance", "hit-15-savings-rate"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["slug"] == "hit-15-savings-rate"
        assert payload["gap"]["pace"] in ("ahead", "on_track", "behind", "met", "unknown")
        assert "progress" in payload["gap"]

    def test_list_and_report(self, home, capsys):
        _define()
        ds.main(["define", "health", "Cut HR", "--current", "70", "--target", "60"])
        capsys.readouterr()
        assert ds.main(["list"]) == 0
        listing = capsys.readouterr().out
        assert "finance/hit-15-savings-rate" in listing and "health/cut-hr" in listing
        assert ds.main(["report"]) == 0
        report = capsys.readouterr().out
        assert "finance" in report and "health" in report

    def test_gap_all_active_only(self, home, capsys):
        _define()
        ds.main(["archive", "finance", "hit-15-savings-rate", "--status", "achieved"])
        capsys.readouterr()
        ds.main(["gap"])
        assert "no active goals" in capsys.readouterr().out


def _write_malformed(home):
    d = home / "state" / "desired" / "finance"
    d.mkdir(parents=True, exist_ok=True)
    (d / "halfbaked.md").write_text("---\ndomain: finance\n---\n\nno goal field\n", encoding="utf-8")


class TestErrors:
    def test_track_missing_exits_2(self, home, capsys):
        assert ds.main(["track", "finance", "ghost", "5"]) == 2
        assert "no goal" in capsys.readouterr().err

    def test_edit_nothing_exits_2(self, home, capsys):
        _define()
        capsys.readouterr()
        assert ds.main(["edit", "finance", "hit-15-savings-rate"]) == 2

    def test_define_missing_body_file_exits_2(self, home, capsys):
        rc = ds.main(["define", "notes", "Needs body", "--body-file", str(home / "missing.md")])
        assert rc == 2
        assert "could not read body file" in capsys.readouterr().err

    def test_show_malformed_exits_2(self, home, capsys):
        # codex P2: direct reads must return clean exit 2, not a traceback.
        _write_malformed(home)
        assert ds.main(["show", "finance", "halfbaked"]) == 2
        assert "error" in capsys.readouterr().err.lower()

    def test_track_malformed_exits_2(self, home, capsys):
        _write_malformed(home)
        assert ds.main(["track", "finance", "halfbaked", "5"]) == 2

    def test_gap_single_malformed_exits_2(self, home, capsys):
        _write_malformed(home)
        assert ds.main(["gap", "finance", "halfbaked"]) == 2


class TestMilestoneCommand:
    def test_add_list_check(self, home, capsys):
        _define()
        capsys.readouterr()
        assert ds.main(["milestone", "finance", "hit-15-savings-rate", "--add", "Renegotiate rent"]) == 0
        assert "1. [ ] Renegotiate rent" in capsys.readouterr().out
        assert ds.main(["milestone", "finance", "hit-15-savings-rate", "--check", "1"]) == 0
        assert "1. [x] Renegotiate rent" in capsys.readouterr().out

    def test_milestone_shows_in_gap(self, home, capsys):
        _define()
        ds.main(["milestone", "finance", "hit-15-savings-rate", "--add", "step one"])
        capsys.readouterr()
        ds.main(["gap", "finance", "hit-15-savings-rate"])
        assert "0/1 milestones" in capsys.readouterr().out

    def test_bad_index_exits_2(self, home, capsys):
        _define()
        capsys.readouterr()
        assert ds.main(["milestone", "finance", "hit-15-savings-rate", "--check", "9"]) == 2
        assert "no milestone #9" in capsys.readouterr().err
