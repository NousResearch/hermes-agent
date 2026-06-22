#!/usr/bin/env python3
"""Regression test for the kanban_create assignee lint (card t_13660393).

Guards against ghost-assignee cards being created when the operator
typos a profile name. The dispatcher would silently self-heal to a
real-but-wrong profile at claim time; this lint catches the typo at
the CLI create boundary.

Structure mirrors ``tests/test_skills_profile_leak.py``:

  * Unit tests on ``_validate_assignee`` / ``_suggest_profile`` /
    ``_levenshtein`` — pure helpers, no DB or HOME dependencies.
  * Integration tests on the CLI path — runs the real
    ``hermes kanban create`` against a temp HERMES_HOME / kanban DB
    to exercise the argparse → DB write boundary.

Run: ``python3 tests/test_kanban_assignee_lint.py``
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# --- Test infrastructure ---------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
HERMES_HOME_REAL = Path.home() / ".hermes"
HERMES_BIN = Path("/Users/charlessectish/.hermes/hermes-agent/venv/bin/hermes")
WRAPPER = Path("/tmp/run_kanban.sh")


def _setup_temp_home(tmp: Path) -> None:
    """Mirror ``~/.hermes/profiles/`` into ``tmp/profiles/``."""
    profiles = tmp / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    real = HERMES_HOME_REAL / "profiles"
    if real.is_dir():
        for p in real.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                (profiles / p.name).mkdir(exist_ok=True)


def _run_hermes_kanban(tmp: Path, *args: str):
    """Run ``hermes kanban …`` with HERMES_HOME + HERMES_KANBAN_DB
    redirected to ``tmp``. Strips dispatcher-set overrides from the
    inherited env so the test exercises the lint in isolation."""
    if not WRAPPER.exists():
        raise unittest.SkipTest(
            f"Test wrapper {WRAPPER} missing — re-create before running"
        )
    env = os.environ.copy()
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_TASK",
    ):
        env.pop(var, None)
    env["TEST_HERMES_HOME"] = str(tmp)
    env["TEST_KANBAN_DB"] = str(tmp / "kanban.db")
    env["PATH"] = str(HERMES_BIN.parent) + ":" + env.get("PATH", "")
    r = subprocess.run(
        [str(WRAPPER), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    return r.returncode, r.stdout, r.stderr


def _db_count(tmp: Path) -> int:
    db = tmp / "kanban.db"
    if not db.exists():
        return 0
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    finally:
        conn.close()


# --- Unit tests on the pure helpers ----------------------------------------


class TestLevenshtein(unittest.TestCase):
    """Pure distance function — independent of roster."""

    def test_identical_strings(self):
        from hermes_cli.kanban import _levenshtein

        self.assertEqual(_levenshtein("code-craftsman", "code-craftsman"), 0)
        self.assertEqual(_levenshtein("", ""), 0)

    def test_one_char_difference(self):
        from hermes_cli.kanban import _levenshtein

        # One substitution
        self.assertEqual(_levenshtein("code", "coda"), 1)
        # One insertion / deletion
        self.assertEqual(_levenshtein("code", "coder"), 1)
        self.assertEqual(_levenshtein("code", "cod"), 1)

    def test_classic_kitten_sitting(self):
        from hermes_cli.kanban import _levenshtein

        self.assertEqual(_levenshtein("kitten", "sitting"), 3)

    def test_empty_inputs(self):
        from hermes_cli.kanban import _levenshtein

        self.assertEqual(_levenshtein("", "abc"), 3)
        self.assertEqual(_levenshtein("abc", ""), 3)

    def test_case_sensitive(self):
        """Levenshtein is case-sensitive — the caller lowercases first."""
        from hermes_cli.kanban import _levenshtein

        self.assertEqual(_levenshtein("Code", "code"), 1)


class TestSuggestProfile(unittest.TestCase):
    """Suggester logic — tested against the real roster on this host."""

    def test_levenshtein_within_2_suggests_closest(self):
        from hermes_cli.kanban import _suggest_profile

        # 'code-crafsman' is one deletion from 'code-craftsman'
        self.assertEqual(_suggest_profile("code-crafsman"), "code-craftsman")

    def test_levenshtein_within_2_typo_of_wags(self):
        from hermes_cli.kanban import _suggest_profile

        # 'wags-reviwer' is two deletions from 'wags-reviewer'
        self.assertEqual(_suggest_profile("wags-reviwer"), "wags-reviewer")

    def test_levenshtein_too_far_no_suggestion(self):
        from hermes_cli.kanban import _suggest_profile

        # 'code-craftsman' vs 'coderftzman' is distance 4 (>>2)
        self.assertIsNone(_suggest_profile("coderftzman"))

    def test_levenshtein_no_close_match_no_suggestion(self):
        from hermes_cli.kanban import _suggest_profile

        # 'architect' is not within distance 2 of any roster entry
        self.assertIsNone(_suggest_profile("architect"))

    def test_unique_prefix_length_3_suggests(self):
        from hermes_cli.kanban import _suggest_profile

        # 'wags' uniquely prefixes 'wags-reviewer' (no other profile starts with wags)
        self.assertEqual(_suggest_profile("wags"), "wags-reviewer")

    def test_unique_prefix_length_3_content(self):
        from hermes_cli.kanban import _suggest_profile

        # 'content-cur' uniquely prefixes 'content-curator'
        self.assertEqual(_suggest_profile("content-cur"), "content-curator")

    def test_ambiguous_prefix_no_suggestion(self):
        from hermes_cli.kanban import _suggest_profile

        # 'tot' ambiguously prefixes totum-build, totum-interface,
        # totum-operator, totum-orchestrator, totum-runner, totum-t
        self.assertIsNone(_suggest_profile("tot"))

    def test_short_prefix_no_suggestion(self):
        """Prefix length < 3 is below the suggestion threshold."""
        from hermes_cli.kanban import _suggest_profile

        self.assertIsNone(_suggest_profile("co"))  # length 2
        self.assertIsNone(_suggest_profile("c"))   # length 1

    def test_empty_name_no_suggestion(self):
        from hermes_cli.kanban import _suggest_profile

        self.assertIsNone(_suggest_profile(""))


class TestValidateAssignee(unittest.TestCase):
    """Pure logic — uses real roster via hermes_cli.profiles.profile_exists."""

    def test_none_returns_zero(self):
        from hermes_cli.kanban import _validate_assignee

        self.assertEqual(_validate_assignee(None), 0)

    def test_empty_returns_zero(self):
        from hermes_cli.kanban import _validate_assignee

        self.assertEqual(_validate_assignee(""), 0)

    def test_real_profile_returns_zero(self):
        from hermes_cli.kanban import _validate_assignee

        self.assertEqual(_validate_assignee("code-craftsman"), 0)

    def test_default_profile_returns_zero(self):
        """``default`` is a special pass-through alias."""
        from hermes_cli.kanban import _validate_assignee

        self.assertEqual(_validate_assignee("default"), 0)

    def test_mixed_case_normalizes(self):
        """profile_exists is case-insensitive for non-default profiles."""
        from hermes_cli.kanban import _validate_assignee

        self.assertEqual(_validate_assignee("Code-Craftsman"), 0)
        self.assertEqual(_validate_assignee("CODE-CRAFTSMAN"), 0)

    def test_ghost_returns_two(self):
        from hermes_cli.kanban import _validate_assignee

        rc = _validate_assignee("architect")
        self.assertEqual(rc, 2)

    def test_ghost_does_not_crash_when_roster_missing(self):
        """If HERMES_HOME is unreadable, refuse rather than silently route.

        Patch list_profiles to raise; _validate_assignee must catch
        and return 2 (refuse) rather than letting the exception
        propagate. Defensive against a corrupt HERMES_HOME where
        silently routing to a wrong profile would be worse than a
        loud refusal.
        """
        from hermes_cli import kanban

        with patch.object(kanban, "list_profiles", side_effect=OSError("disk full")):
            self.assertEqual(kanban._validate_assignee("anything"), 2)

    def test_ghost_message_printed_to_stderr(self):
        """Rejection message lands on stderr with the required format.

        Per the acceptance criteria: ``assignee <name> is not a
        registered Hermes profile. Run `hermes profile list` to see
        valid profiles.``
        """
        from hermes_cli import kanban

        with patch.object(kanban, "list_profiles", return_value=[]):
            import io
            from contextlib import redirect_stderr

            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = kanban._validate_assignee("ghost")
            self.assertEqual(rc, 2)
            err = buf.getvalue()
            self.assertIn("ghost", err)
            self.assertIn("is not a registered Hermes profile", err)
            self.assertIn("hermes profile list", err)

    def test_ghost_with_suggestion_prints_did_you_mean(self):
        """When a close match exists, ``Did you mean: <x>?`` appears."""
        from hermes_cli import kanban

        with patch.object(kanban, "list_profiles", return_value=[]), \
             patch.object(kanban, "profile_exists", return_value=False):
            import io
            from contextlib import redirect_stderr

            buf = io.StringIO()
            with patch.object(kanban, "_suggest_profile", return_value="code-craftsman"), \
                 redirect_stderr(buf):
                rc = kanban._validate_assignee("code-crafsman")
            self.assertEqual(rc, 2)
            err = buf.getvalue()
            self.assertIn("Did you mean: code-craftsman?", err)


# --- CLI integration tests (subprocess; require real roster) ----------------


class TestCliIntegration(unittest.TestCase):
    """End-to-end via the real ``hermes kanban create`` CLI.

    These tests run against a temp HERMES_HOME / kanban DB so they
    exercise the actual argparse → DB write boundary. They require:
      * The venv hermes binary at HERMES_BIN
      * The wrapper script at WRAPPER (sets HERMES_HOME +
        HERMES_KANBAN_DB redirect)

    Skipped if either is missing.
    """

    @classmethod
    def setUpClass(cls):
        if not HERMES_BIN.exists():
            raise unittest.SkipTest(f"hermes binary missing at {HERMES_BIN}")
        if not WRAPPER.exists():
            raise unittest.SkipTest(f"wrapper missing at {WRAPPER}")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        _setup_temp_home(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ghost_assignee_exits_2_with_message(self):
        """Acceptance criterion #1: rc=2, clear stderr, no DB row."""
        rc, out, err = _run_hermes_kanban(
            self.tmp, "create", "Test ghost", "--assignee", "architect",
            "--body", "smoke", "--created-by", "test",
        )
        self.assertEqual(rc, 2, f"expected rc=2, got {rc}; stdout={out!r}; stderr={err!r}")
        self.assertIn("architect", err)
        self.assertIn("not a registered Hermes profile", err)
        self.assertIn("hermes profile list", err)
        self.assertNotIn("Did you mean", err)  # no close match
        self.assertEqual(_db_count(self.tmp), 0)

    def test_levenshtein_typo_suggests_match(self):
        """Acceptance criterion #2: 'code-crafsman' suggests 'code-craftsman'."""
        rc, out, err = _run_hermes_kanban(
            self.tmp, "create", "Test typo", "--assignee", "code-crafsman",
            "--body", "smoke", "--created-by", "test",
        )
        self.assertEqual(rc, 2, f"expected rc=2, got {rc}")
        self.assertIn("Did you mean: code-craftsman?", err)
        self.assertEqual(_db_count(self.tmp), 0)

    def test_unique_prefix_suggests(self):
        """'wags' (prefix length 3, unique) suggests 'wags-reviewer'."""
        rc, out, err = _run_hermes_kanban(
            self.tmp, "create", "Test prefix", "--assignee", "wags",
            "--body", "smoke", "--created-by", "test",
        )
        self.assertEqual(rc, 2, f"expected rc=2, got {rc}")
        self.assertIn("Did you mean: wags-reviewer?", err)
        self.assertEqual(_db_count(self.tmp), 0)

    def test_ambiguous_prefix_no_suggestion(self):
        """'tot' is ambiguous across 6 totum-* profiles → no suggestion."""
        rc, out, err = _run_hermes_kanban(
            self.tmp, "create", "Test ambiguous", "--assignee", "tot",
            "--body", "smoke", "--created-by", "test",
        )
        self.assertEqual(rc, 2, f"expected rc=2, got {rc}")
        self.assertNotIn("Did you mean", err)
        self.assertEqual(_db_count(self.tmp), 0)

    def test_real_assignee_creates_task(self):
        """Acceptance criterion #3: valid name → task created, no behavior change."""
        rc, out, err = _run_hermes_kanban(
            self.tmp, "create", "Test real", "--assignee", "code-craftsman",
            "--body", "smoke", "--created-by", "test",
        )
        self.assertEqual(rc, 0, f"expected rc=0, got {rc}; stderr={err!r}")
        self.assertIn("Created", out)
        self.assertEqual(_db_count(self.tmp), 1)
        # Verify the assignee stored in the DB
        conn = sqlite3.connect(str(self.tmp / "kanban.db"))
        try:
            row = conn.execute(
                "SELECT assignee FROM tasks WHERE title = ?",
                ("Test real",),
            ).fetchone()
            self.assertEqual(row[0], "code-craftsman")
        finally:
            conn.close()

    def test_no_assignee_passes_through(self):
        """No --assignee → unassigned task (lint skipped)."""
        rc, out, err = _run_hermes_kanban(
            self.tmp, "create", "Test unassigned", "--body", "smoke",
            "--created-by", "test",
        )
        self.assertEqual(rc, 0, f"expected rc=0, got {rc}; stderr={err!r}")
        conn = sqlite3.connect(str(self.tmp / "kanban.db"))
        try:
            row = conn.execute(
                "SELECT assignee FROM tasks WHERE title = ?",
                ("Test unassigned",),
            ).fetchone()
            self.assertIsNone(row[0])
        finally:
            conn.close()

    def test_mixed_case_assignee_normalizes(self):
        """--assignee Code-Craftsman → stored as code-craftsman."""
        rc, out, err = _run_hermes_kanban(
            self.tmp, "create", "Test mixed case", "--assignee", "Code-Craftsman",
            "--body", "smoke", "--created-by", "test",
        )
        self.assertEqual(rc, 0, f"expected rc=0, got {rc}; stderr={err!r}")
        conn = sqlite3.connect(str(self.tmp / "kanban.db"))
        try:
            row = conn.execute(
                "SELECT assignee FROM tasks WHERE title = ?",
                ("Test mixed case",),
            ).fetchone()
            self.assertEqual(row[0], "code-craftsman")
        finally:
            conn.close()

    def test_default_assignee_passes(self):
        """'default' is a special alias and should pass the lint."""
        rc, out, err = _run_hermes_kanban(
            self.tmp, "create", "Test default", "--assignee", "default",
            "--body", "smoke", "--created-by", "test",
        )
        self.assertEqual(rc, 0, f"expected rc=0, got {rc}; stderr={err!r}")
        conn = sqlite3.connect(str(self.tmp / "kanban.db"))
        try:
            row = conn.execute(
                "SELECT assignee FROM tasks WHERE title = ?",
                ("Test default",),
            ).fetchone()
            self.assertEqual(row[0], "default")
        finally:
            conn.close()

    def test_help_exits_zero_no_lint(self):
        """--help must keep working — argparse exits before handler dispatch."""
        rc, out, err = _run_hermes_kanban(self.tmp, "create", "--help")
        self.assertEqual(rc, 0)
        self.assertIn("--assignee", out)


if __name__ == "__main__":
    unittest.main()
