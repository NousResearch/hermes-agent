"""Scratch prune audit trail tests — #132401: a delete is never silent.

Pins the per-entry INFO audit record (name, kind, bytes, newest mtime), the
per-pid reap record (pid + cmdline captured BEFORE the kill), and the boot
caller's summary. Red on main (no records exist), green on the fix.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

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


def test_midprune_resumed_write_is_rescued(tmp_path, audit_records, monkeypatch):
    """andrexibiza's C1/F1 (issue #132401): a writer (cwd outside scratch) resumes
    during the prune's reap window — after selection, before the deletion loop.
    The fresh write must survive: selection is a candidate list, never a verdict.
    Red on pre-fix code (entry deleted with the fresh write inside), green after."""
    import hermes_constants_scratch as scratch_mod

    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "resumed-lane"
    entry.mkdir()
    (entry / "old-output.md").write_text("stale content\n", encoding="utf-8")
    _age(entry)
    _age(entry / "old-output.md")
    fresh = entry / "resumed-work.md"

    def write_during_reap(root, doomed_list, *args, **kwargs):
        # The resumed writer lands its fresh write mid-prune: selection is done,
        # the deletion loop has not started. This is the F1 window.
        fresh.write_text("FRESH RESUMED WORK\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(scratch_mod, "reap_processes_rooted_in", write_during_reap)

    assert prune_scratch_dir(scratch) == 0  # rescued — not counted as removed
    assert fresh.exists(), "fresh mid-prune write was destroyed"
    assert entry.exists()
    all_records = [r.message for r in audit_records.records]
    rescue = [m for m in all_records if "rescued" in m and "resumed-lane" in m]
    assert rescue, all_records
    assert not any("removed entry='resumed-lane'" in m for m in all_records)


def test_prune_failed_removal_not_counted_and_logged(tmp_path, audit_records, monkeypatch):
    """andrexibiza's C2/F2 (issue #132401): a failed directory removal must not be
    reported as a removal — the count claims confirmed departures only, and the
    failure leaves its own record. Monkeypatched rmtree leaves the entry intact,
    mimicking an ignore_errors-suppressed partial failure (permissions, locked files)."""
    import hermes_constants_scratch as scratch_mod

    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "locked-lane"
    entry.mkdir()
    (entry / "payload").write_text("data\n", encoding="utf-8")
    _age(entry)
    _age(entry / "payload")

    real_rmtree = shutil.rmtree

    def failing_rmtree(path, *args, **kwargs):
        # Pure residue simulation: rmtree swallows every error (ignore_errors
        # semantics under a fully-locked tree) and the entry survives intact.
        return None

    monkeypatch.setattr(scratch_mod.shutil, "rmtree", failing_rmtree)
    assert prune_scratch_dir(scratch) == 0  # not counted — the entry still exists
    all_records = [r.message for r in audit_records.records]
    residue = [m for m in all_records if "removal left residue" in m and "locked-lane" in m]
    assert residue, all_records
    assert not any("removed entry='locked-lane'" in m for m in all_records)


def test_audit_records_reach_durable_sink_early_boot(tmp_path, audit_records, monkeypatch):
    """The early-boot sink gap (#132401, andrexibiza): the prune can run before
    ``setup_logging`` installs the file handler, so a bare logger.info is
    delivery-by-hope. The audit records must land somewhere durable regardless:
    with no INFO-capable root handler (the early-boot state), the record is
    appended to ``<home>/logs/scratch-prune.log`` on disk."""
    import logging as _logging

    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "early-boot-lane"
    entry.mkdir()
    (entry / "f").write_text("x", encoding="utf-8")
    _age(entry)
    _age(entry / "f")

    audit_home = tmp_path / "audit-home"
    monkeypatch.setenv("HERMES_HOME", str(audit_home))
    # Early-boot state, simulated the way the code sees it: the REAL root logger
    # has no INFO-capable handlers. Strip any pytest/caplog handlers for the call,
    # then restore — auditing audit_info's own check, not a patched module attr.
    real_root = _logging.getLogger()
    saved_handlers = list(real_root.handlers)
    saved_level = real_root.level
    real_root.handlers = []
    try:
        assert prune_scratch_dir(scratch) == 1
    finally:
        real_root.handlers = saved_handlers
        real_root.setLevel(saved_level)

    sink = audit_home / "logs" / "scratch-prune.log"
    assert sink.exists(), "audit record did not reach the durable sink"
    content = sink.read_text(encoding="utf-8-sig")
    assert "removed entry='early-boot-lane'" in content, content


def test_audit_no_double_write_when_log_sink_live(tmp_path, audit_records, monkeypatch):
    """When the normal file handler IS live (post-setup), records go only through
    the logger — no duplication into the durable sink file."""
    import logging as _logging

    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "normal-boot-lane"
    entry.mkdir()
    (entry / "f").write_text("x", encoding="utf-8")
    _age(entry)
    _age(entry / "f")

    audit_home = tmp_path / "audit-home2"
    monkeypatch.setenv("HERMES_HOME", str(audit_home))

    # Post-setup state: an INFO-capable handler exists at the root.
    live_root = _logging.getLogger("probe-live-sink")
    live_root.addHandler(_logging.StreamHandler())

    class _InfoHandler(_logging.NullHandler):
        level = _logging.INFO

    import hermes_constants_scratch as scratch_mod
    real_getlogger = _logging.getLogger

    def getlogger_with_live(name=None):
        lg = real_getlogger(name)
        if lg is real_getlogger():
            lg.addHandler(_InfoHandler())
        return lg

    monkeypatch.setattr(_logging, "getLogger", getlogger_with_live)

    assert prune_scratch_dir(scratch) == 1
    sink = audit_home / "logs" / "scratch-prune.log"
    assert not sink.exists(), "record duplicated into durable sink while handler was live"


def test_prune_oserror_failure_logged_not_counted(tmp_path, audit_records, monkeypatch):
    """An outright OSError on the delete path (vanished mid-run, permissions) is
    recorded as a failure and never counted as a removal."""
    import hermes_constants_scratch as scratch_mod

    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "doomed-lane"
    entry.mkdir()
    (entry / "f").write_text("x", encoding="utf-8")
    _age(entry)
    _age(entry / "f")

    def exploding_rmtree(path, *args, **kwargs):
        raise OSError("simulated permission denied")

    monkeypatch.setattr(scratch_mod.shutil, "rmtree", exploding_rmtree)
    assert prune_scratch_dir(scratch) == 0
    all_records = [r.message for r in audit_records.records]
    failed = [m for m in all_records if "removal failed" in m and "doomed-lane" in m]
    assert failed, all_records
    assert entry.exists()  # and the entry itself survives the failed attempt