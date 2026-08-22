"""Regression tests: a failed quarantine restore must never strand `hermes`.

On Windows the updater renames the live ``hermes*.exe`` shims aside
(``hermes.exe.old.<unix-ms>``) so uv can write replacements. Gaps in the
recovery path ended with ``hermes`` gone from PATH — and, because the command
that repairs it IS ``hermes update``, unrecoverable without a manual reinstall
(#75584):

1. Restoring a shim got a single attempt whose ``OSError`` was swallowed in
   silence, while the outbound quarantine rename already retried a lock.
2. The startup sweep unlinked every ``*.exe.old.*``. When the original shim was
   already missing, that .old file was the ONLY surviving copy — deleting it
   converted a one-rename recovery into a full reinstall. It also raced a
   concurrent in-flight update, destroying the quarantine that update's own
   restore was about to rename back.

These tests pin the hardened behavior: retry, rescue, report, order by parsed
stamp, and leave files we did not create alone.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import _early_recovery as er
from hermes_cli import _install_repair as ir
from hermes_cli import main as cli_main


def _make_scripts_dir(tmp_path: Path) -> Path:
    scripts = tmp_path / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    return scripts


def _stamp(ms_ago: int = 0) -> int:
    return int(time.time() * 1000) - ms_ago


def _run_cleanup(scripts: Path):
    """Drive the sweep with the Windows gate forced and the registry stubbed.

    ``_cleanup_pending_shim_renames`` reaches into PendingFileRenameOperations;
    it has its own tests and must not run here.
    """
    return patch.multiple(
        cli_main,
        _is_windows=lambda: True,
        _cleanup_pending_shim_renames=lambda _scripts_dir: 0,
    )


# ---------------------------------------------------------------------------
# orphan rescue
# ---------------------------------------------------------------------------


def test_cleanup_rescues_orphan_when_original_missing(tmp_path):
    """The .old file is the last copy of the shim — put it back, don't delete."""
    scripts = _make_scripts_dir(tmp_path)
    orphan = scripts / f"hermes.exe.old.{_stamp()}"
    orphan.write_bytes(b"MZ-orphan")

    with _run_cleanup(scripts):
        cli_main._cleanup_quarantined_exes(scripts)

    assert (scripts / "hermes.exe").read_bytes() == b"MZ-orphan"
    assert not orphan.exists()


def test_cleanup_rescue_survives_a_transient_lock(tmp_path, capsys):
    """The rescue rename retries a lock instead of stranding on first failure.

    This is the window the sweep runs in: the shim is ALREADY gone from PATH, so
    giving up here leaves the user stranded exactly as if the sweep had deleted
    the file.
    """
    scripts = _make_scripts_dir(tmp_path)
    orphan = scripts / f"hermes.exe.old.{_stamp()}"
    orphan.write_bytes(b"MZ-orphan")

    real_rename = os.rename
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(32, "being used by another process")
        return real_rename(src, dst)

    with _run_cleanup(scripts), patch.object(er.os, "rename", flaky):
        cli_main._cleanup_quarantined_exes(scripts)

    assert (scripts / "hermes.exe").read_bytes() == b"MZ-orphan"
    assert calls["n"] >= 2, "rescue must retry after a transient lock"
    assert capsys.readouterr().err == "", "a recovered rescue must stay quiet"


def test_cleanup_rescue_reports_when_it_cannot_recover(tmp_path, capsys):
    """A rescue that exhausts its retries must say so, not fail silently."""
    scripts = _make_scripts_dir(tmp_path)
    orphan = scripts / f"hermes.exe.old.{_stamp()}"
    orphan.write_bytes(b"MZ-orphan")

    def always_locked(src, dst):
        raise PermissionError(32, "being used by another process")

    with _run_cleanup(scripts), patch.object(er.os, "rename", always_locked):
        cli_main._cleanup_quarantined_exes(scripts)

    captured = capsys.readouterr()
    assert "FAILED to restore hermes.exe" in captured.err
    assert "move" in captured.err, "must print the literal recovery command"
    assert captured.out == "", "stdout must stay clean for JSON-RPC"
    assert orphan.exists(), "the last copy must survive a failed rescue"


def test_cleanup_rescue_is_quiet_when_another_process_wins(tmp_path, capsys):
    """Two sweeps, one orphan: the loser must no-op cleanly, not report failure."""
    scripts = _make_scripts_dir(tmp_path)
    orphan = scripts / f"hermes.exe.old.{_stamp()}"
    orphan.write_bytes(b"MZ-orphan")
    original = scripts / "hermes.exe"

    def loses_race(src, dst):
        # The "winner" lands the shim while our attempt is in flight.
        original.write_bytes(b"MZ-from-winner")
        raise PermissionError(32, "being used by another process")

    with _run_cleanup(scripts), patch.object(er.os, "rename", loses_race):
        cli_main._cleanup_quarantined_exes(scripts)

    captured = capsys.readouterr()
    assert original.read_bytes() == b"MZ-from-winner"
    assert captured.err == "", "losing a benign race is not a failure"
    assert captured.out == ""


# ---------------------------------------------------------------------------
# ordering and provenance
# ---------------------------------------------------------------------------


def test_cleanup_rescues_newest_by_parsed_stamp_not_lexicographic(tmp_path):
    """Mixed-width stamps: ordering must follow the parsed integer.

    ``sorted(reverse=True)`` over raw filenames puts ``.old.999`` above a
    13-digit epoch-ms stamp, which would rescue the wrong bytes onto the live
    shim name.
    """
    scripts = _make_scripts_dir(tmp_path)
    (scripts / "hermes.exe.old.999").write_bytes(b"MZ-stray-short-stamp")
    (scripts / f"hermes.exe.old.{_stamp(60_000)}").write_bytes(b"MZ-genuine")

    with _run_cleanup(scripts):
        cli_main._cleanup_quarantined_exes(scripts)

    assert (scripts / "hermes.exe").read_bytes() == b"MZ-genuine"


def test_cleanup_ignores_names_it_did_not_create(tmp_path):
    """An unparseable suffix is not ours: never rescued, never deleted."""
    scripts = _make_scripts_dir(tmp_path)
    (scripts / "hermes.exe").write_bytes(b"MZ-live")
    foreign = scripts / "hermes.exe.old.backup"
    foreign.write_bytes(b"MZ-someone-elses-file")

    with _run_cleanup(scripts):
        cli_main._cleanup_quarantined_exes(scripts)

    assert foreign.exists(), "the sweep must not delete files of unknown provenance"
    assert foreign.read_bytes() == b"MZ-someone-elses-file"
    assert (scripts / "hermes.exe").read_bytes() == b"MZ-live"


def test_cleanup_does_not_rescue_from_a_foreign_name(tmp_path):
    """Missing shim + only a foreign .old: leave it be rather than guess."""
    scripts = _make_scripts_dir(tmp_path)
    foreign = scripts / "hermes.exe.old.backup"
    foreign.write_bytes(b"MZ-someone-elses-file")

    with _run_cleanup(scripts):
        cli_main._cleanup_quarantined_exes(scripts)

    assert not (scripts / "hermes.exe").exists()
    assert foreign.exists()


# ---------------------------------------------------------------------------
# concurrency grace window
# ---------------------------------------------------------------------------


def test_cleanup_leaves_fresh_quarantine_for_concurrent_update(tmp_path):
    """A young .old may belong to an update in flight elsewhere — hands off."""
    scripts = _make_scripts_dir(tmp_path)
    (scripts / "hermes.exe").write_bytes(b"MZ-live")
    fresh = scripts / f"hermes.exe.old.{_stamp()}"
    fresh.write_bytes(b"MZ-inflight")

    with _run_cleanup(scripts):
        cli_main._cleanup_quarantined_exes(scripts)

    assert fresh.exists(), "a live quarantine must survive another process's sweep"


def test_cleanup_still_sweeps_genuinely_stale_quarantine(tmp_path):
    """Past the grace window, with the shim present, it's garbage — sweep it."""
    scripts = _make_scripts_dir(tmp_path)
    (scripts / "hermes.exe").write_bytes(b"MZ-live")
    ancient_ms = (cli_main._QUARANTINE_GRACE_SECONDS + 60) * 1000
    stale = scripts / f"hermes.exe.old.{_stamp(ancient_ms)}"
    stale.write_bytes(b"MZ-stale")

    with _run_cleanup(scripts):
        cli_main._cleanup_quarantined_exes(scripts)

    assert not stale.exists()
    assert (scripts / "hermes.exe").read_bytes() == b"MZ-live"


def test_cleanup_age_comes_from_filename_not_mtime(tmp_path):
    """rename() preserves mtime, so only the name records the quarantine time."""
    scripts = _make_scripts_dir(tmp_path)
    (scripts / "hermes.exe").write_bytes(b"MZ-live")
    fresh = scripts / f"hermes.exe.old.{_stamp()}"
    fresh.write_bytes(b"MZ-inflight")
    week_ago = time.time() - 7 * 24 * 3600
    os.utime(fresh, (week_ago, week_ago))

    with _run_cleanup(scripts):
        cli_main._cleanup_quarantined_exes(scripts)

    assert fresh.exists(), "grace window must key off the .old.<ms> stamp"


def test_quarantine_stamp_ms_parses_and_rejects():
    assert cli_main._quarantine_stamp_ms(Path("hermes.exe.old.1787020473885")) == 1787020473885
    assert cli_main._quarantine_stamp_ms(Path("hermes.exe.old.backup")) is None
    assert cli_main._quarantine_stamp_ms(Path("hermes.exe")) is None


# ---------------------------------------------------------------------------
# the shared restore helper
# ---------------------------------------------------------------------------


def test_helper_retries_then_succeeds(tmp_path):
    scripts = _make_scripts_dir(tmp_path)
    quarantined = scripts / "hermes.exe.old.123"
    quarantined.write_bytes(b"MZ-old-hermes")
    original = scripts / "hermes.exe"

    real_rename = os.rename
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(32, "being used by another process")
        return real_rename(src, dst)

    with patch.object(er.os, "rename", flaky):
        failed = er.restore_quarantined_shims([(original, quarantined)])

    assert failed == []
    assert original.read_bytes() == b"MZ-old-hermes"
    assert calls["n"] >= 2


def test_helper_reports_failure_and_returns_the_pair(tmp_path, capsys):
    scripts = _make_scripts_dir(tmp_path)
    quarantined = scripts / "hermes.exe.old.123"
    quarantined.write_bytes(b"MZ-old-hermes")
    original = scripts / "hermes.exe"

    def always_locked(src, dst):
        raise PermissionError(32, "being used by another process")

    with patch.object(er.os, "rename", always_locked):
        failed = er.restore_quarantined_shims([(original, quarantined)])

    captured = capsys.readouterr()
    assert failed == [(original, quarantined)]
    assert "FAILED to restore hermes.exe" in captured.err
    assert "hermes.exe.old.123" in captured.err
    assert "move" in captured.err
    assert captured.out == ""


def test_helper_is_a_noop_when_installer_wrote_a_fresh_shim(tmp_path, capsys):
    scripts = _make_scripts_dir(tmp_path)
    quarantined = scripts / "hermes.exe.old.123"
    quarantined.write_bytes(b"MZ-old")
    original = scripts / "hermes.exe"
    original.write_bytes(b"MZ-fresh")

    failed = er.restore_quarantined_shims([(original, quarantined)])

    assert failed == []
    assert original.read_bytes() == b"MZ-fresh", "must not clobber the fresh shim"
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# both call sites route through the helper
# ---------------------------------------------------------------------------


def test_main_restore_reports_on_stderr(tmp_path, capsys):
    scripts = _make_scripts_dir(tmp_path)
    quarantined = scripts / "hermes.exe.old.123"
    quarantined.write_bytes(b"MZ-old-hermes")
    original = scripts / "hermes.exe"

    def always_locked(src, dst):
        raise PermissionError(32, "being used by another process")

    with patch.object(er.os, "rename", always_locked):
        cli_main._restore_quarantined_exes([(original, quarantined)])

    captured = capsys.readouterr()
    assert "FAILED to restore hermes.exe" in captured.err
    assert captured.out == ""


def test_repair_restore_reports_on_stderr(tmp_path, capsys):
    """The early-recovery path must warn on stderr (acp speaks JSON-RPC on stdout)."""
    scripts = _make_scripts_dir(tmp_path)
    quarantined = scripts / "hermes.exe.old.123"
    quarantined.write_bytes(b"MZ-old-hermes")
    original = scripts / "hermes.exe"

    def always_locked(src, dst):
        raise PermissionError(32, "being used by another process")

    with patch.object(er.os, "rename", always_locked):
        ir._restore_quarantined_exes([(original, quarantined)])

    captured = capsys.readouterr()
    assert "FAILED to restore hermes.exe" in captured.err
    assert captured.out == "", "stdout must stay clean for JSON-RPC"
