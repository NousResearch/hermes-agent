"""The Windows crash cells' oracles, judged on this OS: a verdict the cells can only reach on a
Windows runner must still say pass exactly when the user's update was what the cell claims."""

import os
import subprocess
import time
from pathlib import Path

import psutil
import pytest

import tests.e2e.core.windows_update.test_crash_cells as crash


class _Machine:
    def __init__(self, logs: Path) -> None:
        self.logs = logs

    def evidence(self) -> str:
        return ""

    def kill_owned(self) -> None:
        pass


class _Proc:
    """An update process: ``rc`` None while it runs; a kill ends it."""

    pid = 4242
    transcript = Path("update.log")

    def __init__(self, rc: int | None) -> None:
        self.returncode = rc

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 1
        return self.returncode


@pytest.mark.parametrize("exit_rc,taskkill_rc,point_seen,killed_there", [
    (None, 0, True, True),  # running at its kill point: the crash the cell asserts on
    (0, 0, True, False),  # already exited 0 with HEAD at the target: never interrupted
    (None, 128, True, False),  # taskkill found no process: it exited between poll and kill
    (0, 0, False, False),  # exited before the point: harness verdict
])
def test_kill_point_counts_only_an_update_killed_while_running(
        tmp_path, monkeypatch, exit_rc, taskkill_rc, point_seen, killed_there):
    monkeypatch.setattr(crash, "taskkill_tree", lambda pid: subprocess.CompletedProcess([], taskkill_rc))
    proc = _Proc(exit_rc)

    def point(*_):
        return "checkout at target" if point_seen else None

    if killed_there:
        assert crash._kill_when(_Machine(tmp_path), proc, "cell", point, "t") == "checkout at target"
    else:
        with pytest.raises(AssertionError, match="cell: the update"):
            crash._kill_when(_Machine(tmp_path), proc, "cell", point, "t")


@pytest.mark.parametrize("age,with_ct,live", [
    (60, False, True),  # v1 marker inside update_lock's 20-minute ceiling
    (1260, False, False),  # v1 past the ceiling: the product reads it DEAD (the pid may be reused)
    (1260, True, True),  # a matching creation time is live at any age
])
def test_marker_oracle_reads_a_marker_live_exactly_when_update_lock_does(age, with_ct, live):
    pid = os.getpid()
    ct = f"ct:{psutil.Process(pid).create_time():.3f}\n" if with_ct else ""
    text = f"{pid}\n{time.time() - age}\n{ct}"
    assert crash._marker_live(text) == (f"owner {pid}" if live else None)


class _Journey:
    def __init__(self, machine, cells: dict) -> None:
        self.machine, self._cells = machine, cells

    def __getitem__(self, cell: str) -> dict:
        return self._cells[cell]


@pytest.mark.parametrize("rc,receipt,completed", [
    (0, None, True),  # exited 0 after logging completion
    (crash.PY_FINAL_FLUSH_FAILED, "success", True),  # only the flush into the dead script failed
    (crash.PY_FINAL_FLUSH_FAILED, "failed", False),  # 120 masked a failure exit
    (crash.PY_FINAL_FLUSH_FAILED, None, False),  # 120 and no receipt of this run
])
def test_orphan_cell_accepts_exit_120_only_with_a_success_receipt(tmp_path, monkeypatch, rc, receipt, completed):
    monkeypatch.delenv("HERMES_E2E_STRICT_ACCEPTANCE", raising=False)
    target = "a" * 40
    tree = {"head": target, "dirty": "", "diff_rc": 0, "diff_err": "", "target_file": True}
    ok = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    cell = {"orphan_finished": True, "orphan_reported_done": True, "orphan_rc": rc, "orphan_receipt": receipt,
            "target": target, "tree_after_orphan": tree, "tree_final": tree, "holders": ["delegate"],
            "turn": type("Turn", (), {"ok": True, "run": ok})(), "follow_up": ok, "marker_final": False,
            "dead_while_running": None, "marker_after_orphan": None}
    journey = _Journey(_Machine(tmp_path), {"orphaned_update": cell})
    if completed:
        crash.test_desktop_handoff_script_killed_alone_keeps_the_marker_live_until_its_update_ends(journey)
    else:
        with pytest.raises(AssertionError, match="did not finish the update"):
            crash.test_desktop_handoff_script_killed_alone_keeps_the_marker_live_until_its_update_ends(journey)


def test_orphan_receipt_is_the_newest_one_written_after_the_baseline(tmp_path):
    receipts = tmp_path / "logs" / "update_receipts"
    receipts.mkdir(parents=True)
    machine = type("M", (), {"hermes_home": tmp_path})()
    (receipts / "update_20261004_000000_1_a.json").write_text('{"outcome": "success"}', encoding="utf-8")
    before = crash._receipts(machine)
    assert crash._new_receipt_outcome(machine, before) is None  # an earlier run's success is not this run's
    (receipts / "update_20261004_000100_2_b.json").write_text('{"outcome": "failed"}', encoding="utf-8")
    assert crash._new_receipt_outcome(machine, before) == "failed"
