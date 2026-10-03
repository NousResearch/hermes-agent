"""Scratch prune audit trail tests — #132401: a delete is never silent.

Pins the per-entry INFO audit record (name, kind, bytes, newest mtime), the
per-pid reap record (pid + cmdline captured BEFORE the kill), and the boot
caller's summary. Red on main (no records exist), green on the fix.
"""
from __future__ import annotations

import logging
import os
import time

import pytest

from hermes_constants import get_scratch_dir, prune_scratch_dir


def _age(path, hours=30.0):
    ancient = time.time() - hours * 3600
    os.utime(path, (ancient, ancient))


@pytest.fixture
def audit_records(caplog):
    """Collect INFO+ records from both logger names, root included."""
    caplog.set_level(logging.INFO)
    return caplog


def test_prune_writes_per_entry_audit_record(tmp_path, audit_records):
    """Each removed entry leaves an INFO line naming it — with size and newest mtime."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "lane-report-draft"
    entry.mkdir()
    (entry / "draft.md").write_text("multi-day work product\n", encoding="utf-8")
    _age(entry)
    _age(entry / "draft.md")

    assert prune_scratch_dir(scratch) == 1
    assert not entry.exists()

    audit = [r.message for r in audit_records.records if "scratch prune" in r.message]
    assert any("removed entry='lane-report-draft'" in m and "kind=dir" in m for m in audit), audit


def test_prune_audit_record_carries_bytes_and_mtime(tmp_path, audit_records):
    """The audit line prices the loss: bytes present, newest_mtime present."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "bench-suite"
    entry.mkdir()
    (entry / "results.json").write_text('{"latency_ms": 21}\n', encoding="utf-8")
    _age(entry)
    _age(entry / "results.json")

    assert prune_scratch_dir(scratch) == 1

    audit = [r.message for r in audit_records.records if "removed entry=" in r.message]
    assert audit, "no per-entry audit record"
    line = audit[0]
    assert "bytes=" in line and "bytes=0 " not in line, line
    assert "newest_mtime=" in line, line


def test_prune_of_file_entry_writes_audit_record(tmp_path, audit_records):
    """A top-level FILE entry (not a dir) is audited too — same fields, kind=file."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    artifact = scratch / "q3-report.pdf"
    artifact.write_bytes(b"%PDF-1.4 fake report bytes\n" * 40)
    _age(artifact)

    assert prune_scratch_dir(scratch) == 1

    audit = [r.message for r in audit_records.records if "removed entry=" in r.message]
    assert any("kind=file" in m and "q3-report.pdf" in m for m in audit), audit


def test_prune_of_live_tree_writes_no_audit_record(tmp_path, audit_records):
    """A live tree is not audited (and not removed): early-stop means no record."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    live = scratch / "live-lane"
    live.mkdir()
    (live / "log").write_text("still writing\n", encoding="utf-8")

    assert prune_scratch_dir(scratch) == 0

    audit = [r.message for r in audit_records.records if "removed entry=" in r.message]
    assert not audit, audit


@pytest.mark.platforms("posix")  # reap path uses psutil + POSIX process semantics
def test_prune_reap_writes_per_pid_record_with_cmdline(tmp_path, audit_records):
    """Each reaped pid leaves an INFO record naming pid + cmdline (captured pre-kill)."""
    import subprocess
    import sys

    scratch = get_scratch_dir(tmp_path, prune=False)
    idle = scratch / "idle-lane" / "wt"
    idle.mkdir(parents=True)
    ancient = time.time() - 30 * 3600
    for p in (idle.parent, idle):
        os.utime(p, (ancient, ancient))
    cmdline_sentinel = "audit-trail-sentinel-cmdline"
    sleeper = subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep(60); # {cmdline_sentinel}"],
        cwd=str(idle), stdin=subprocess.DEVNULL,
    )
    try:
        assert prune_scratch_dir(scratch) == 1
        assert sleeper.wait(timeout=10) is not None
        audit = [r.message for r in audit_records.records if "reaped pid=" in r.message]
        assert audit, "no per-pid reap record"
        assert any(str(sleeper.pid) in m for m in audit), audit
        # cmdline evidence is captured BEFORE the kill — the sentinel must survive.
        assert any(cmdline_sentinel in m for m in audit), audit
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait(timeout=10)


def test_prune_summary_logged_at_boot_caller(tmp_path, audit_records):
    """The boot caller surfaces the count — the summary no longer evaporates (#132401)."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    for name in ("alpha", "beta"):
        e = scratch / name
        e.mkdir()
        (e / "f").write_text("x", encoding="utf-8")
        _age(e)
        _age(e / "f")

    assert prune_scratch_dir(scratch) == 2
    summary = [r.message for r in audit_records.records if "removed 2 idle entry" in r.message]
    all_records = [r.message for r in audit_records.records]
    assert summary, all_records