"""The Windows update E2E, judged on this OS: the workflow's selection runs every cell, and a
verdict the crash cells can only reach on a Windows runner says pass exactly when the user's
update was what the cell claims."""

import importlib
import inspect
import os
import shlex
import subprocess
import time
from pathlib import Path

import psutil
import pytest
from _pytest.mark.expression import Expression

import tests.e2e.core.windows_update.test_crash_cells as crash

_REPO = Path(__file__).resolve().parents[2]
_SUITE = "tests/e2e/core/windows_update"


def _workflow_selection() -> str:
    """The ``-m`` expression the Windows install + update E2E workflow runs its suite with."""
    yaml = pytest.importorskip("hermes_yaml")
    wf = yaml.safe_load((_REPO / ".github/workflows/windows-install-update-e2e.yml").read_text(encoding="utf-8-sig"))
    step = next(s for s in wf["jobs"]["install-update"]["steps"] if s.get("name") == "Run Windows install + update E2E")
    argv = shlex.split(step["run"].replace("\\\n", " "))
    assert f"{_SUITE}/" in argv, f"the workflow no longer runs {_SUITE}/: {argv}"
    return argv[argv.index("-m") + 1]


# run_tests.sh reports a file whose every test the -m expression deselects as passed with 0
# tests run, so a dropped platforms("windows") marker would silently run zero cells.
@pytest.mark.parametrize("path", sorted(p.name for p in (_REPO / _SUITE).glob("test_*.py")))
def test_windows_update_workflow_selects_every_test_of_the_suite(path):
    selection = Expression.compile(_workflow_selection())
    module = importlib.import_module(f"{_SUITE.replace('/', '.')}.{Path(path).stem}")
    tests = [fn for name, fn in inspect.getmembers(module, inspect.isfunction) if name.startswith("test")]
    assert tests, f"{path}: no module-level test functions to select"
    for fn in tests:
        marks = [getattr(m, "mark", m) for m in [*getattr(module, "pytestmark", []), *getattr(fn, "pytestmark", [])]]
        names = {m.name for m in marks}
        assert selection.evaluate(lambda name, **_: name in names), f"{path}::{fn.__name__}: the workflow deselects it"
        windows = any(m.name == "platforms" and "windows" in m.args for m in marks)
        assert windows, f"{path}::{fn.__name__}: not marked platforms('windows'), so the Windows runner skips it"


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


# Review CI1: a run-time xfail outlives its fix unless something checks the tree. PR CI never runs
# the orphan cell (Windows, opt-in) nor strict acceptance, so this file is the expiry.
def test_orphan_marker_wrapper_expires_once_its_fix_is_in_the_tree():
    missing = crash.orphan_marker_fix_missing()
    assert missing, ("the orphan-marker fix (#132354 + #132365) is in the tree: delete ORPHAN_MARKER_GAP, "
                     "ORPHAN_MARKER_FIX, _orphan_gap_excuse and these guards, and assert the marker plainly")
    # The halves already on this branch must still carry their footprint: a renamed judge would
    # otherwise read as "not landed" forever and keep the excuse alive.
    landed = {rel for rel, _ in crash.ORPHAN_MARKER_FIX} - {m.split(":")[0] for m in missing}
    assert "hermes_cli/update_lock.py" in landed, f"the Python delegate judge lost its footprint: {missing}"


def test_orphan_marker_fix_footprint_is_read_from_the_tree(tmp_path):
    for rel, footprint in crash.ORPHAN_MARKER_FIX:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(f"x\n{footprint}\n", encoding="utf-8")
    assert crash.orphan_marker_fix_missing(tmp_path) == []
    (tmp_path / "scripts/desktop-update/windows.ps1").unlink()
    assert crash.orphan_marker_fix_missing(tmp_path) == [
        "scripts/desktop-update/windows.ps1: " + crash.ORPHAN_MARKER_FIX[2][1]]


_GAP_FAILURE = "orphaned_update: .hermes-update-in-progress read DEAD 3s after the script died"


@pytest.mark.parametrize("fix_in_tree", [False, True])
def test_orphan_gap_is_excused_only_while_its_fix_is_absent(monkeypatch, fix_in_tree):
    monkeypatch.delenv("HERMES_E2E_STRICT_ACCEPTANCE", raising=False)
    monkeypatch.setattr(crash, "orphan_marker_fix_missing", lambda: [] if fix_in_tree else ["a half"])
    expected = AssertionError if fix_in_tree else pytest.xfail.Exception
    with pytest.raises(expected, match="read DEAD"):
        with crash._orphan_gap_excuse():
            raise AssertionError(_GAP_FAILURE)


def test_orphan_marker_guard_fires_once_the_fix_lands(monkeypatch):
    monkeypatch.setattr(crash, "orphan_marker_fix_missing", lambda: [])
    with pytest.raises(AssertionError, match="delete ORPHAN_MARKER_GAP"):
        test_orphan_marker_wrapper_expires_once_its_fix_is_in_the_tree()
