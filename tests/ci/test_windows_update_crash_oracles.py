"""The Windows update E2E, judged on this OS: the workflow's selection runs every cell, and a
verdict the crash cells can only reach on a Windows runner says pass exactly when the user's
update was what the cell claims."""

import importlib
import inspect
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from _pytest.mark.expression import Expression

import tests.e2e.core.windows_update.test_crash_cells as crash
from hermes_cli import update_lock

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


_ROOT_KILLED = b"SUCCESS: The process with PID 4242 (child process of PID 77) has been terminated.\r\n"
_CHILD_GONE = b'ERROR: The process with PID 5151 (child process of PID 4242) could not be terminated.\r\n'
_ROOT_GONE = b'ERROR: The process "4242" not found.\r\n'


@pytest.mark.parametrize("exit_rc,taskkill_rc,stdout,point_seen,killed_there", [
    (None, 0, _ROOT_KILLED, True, True),  # running at its kill point: the crash the cell asserts on
    (0, 0, b"", True, False),  # already exited 0 with HEAD at the target: never interrupted
    (None, 128, _ROOT_GONE, True, False),  # taskkill found no process: it exited between poll and kill
    # The update's job (kill-on-close) took git down with it mid-walk: the root WAS killed there.
    (None, 128, _CHILD_GONE + _ROOT_KILLED, True, True),
    (0, 0, b"", False, False),  # exited before the point: harness verdict
])
def test_kill_point_counts_only_an_update_killed_while_running(
        tmp_path, monkeypatch, exit_rc, taskkill_rc, stdout, point_seen, killed_there):
    monkeypatch.setattr(crash, "taskkill_tree",
                        lambda pid: subprocess.CompletedProcess([], taskkill_rc, stdout, b""))
    proc = _Proc(exit_rc)

    def point(*_):
        return "checkout at target" if point_seen else None

    if killed_there:
        assert crash._kill_when(_Machine(tmp_path), proc, "cell", point, "t") == "checkout at target"
    else:
        with pytest.raises(AssertionError, match="cell: the update"):
            crash._kill_when(_Machine(tmp_path), proc, "cell", point, "t")


@pytest.fixture(scope="module")
def other_process():
    """A live process that is not this one: the judge reads its own pid by exact incarnation."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    try:
        yield proc.pid, update_lock.process_create_time(proc.pid)
    finally:
        proc.kill()
        proc.wait(timeout=60)


# (lines after "<pid>\n<started_at>\n" given (pid, ct, now), the oracle's expected reading)
_MARKERS = {
    "v1-inside-ceiling": (lambda p, ct, now: f"{now - 60}\n", "owner"),
    "v1-past-ceiling": (lambda p, ct, now: f"{now - 1260}\n", None),  # the pid may be reused
    "ct-match-any-age": (lambda p, ct, now: f"{now - 1260}\nct:{ct:.3f}\n", "owner"),
    # Review K132346-oracle-judge: the three rows a hand copy of the judge got wrong.
    "float-started-at": (lambda p, ct, now: f"{now - 60}.5\nct:{ct:.3f}\n", None),  # malformed
    "ct-off-by-1.5s": (lambda p, ct, now: f"{now - 60}\nct:{ct + 1.5:.3f}\n", "owner"),
    "run-before-delegate": (lambda p, ct, now: f"{now - 60}\nct:{ct - 100:.3f}\nrun:abc\n"
                                               f"delegate:{p} ct:{ct:.3f}\n", "delegate"),
}


@pytest.mark.parametrize("case", list(_MARKERS))
def test_marker_oracle_reads_a_marker_live_exactly_when_update_lock_does(other_process, case):
    pid, ct = other_process
    assert ct is not None, "the product cannot read a creation time on this host"
    tail, expected = _MARKERS[case]
    text = f"{pid}\n" + tail(pid, ct, int(time.time()))
    verdict, owner, _ = update_lock.judge_marker(text.encode("utf-8"))
    assert crash._marker_live(text) == (f"{expected} {pid}" if expected else None)
    assert (verdict == "live", owner) == ((True, pid) if expected else (False, None)), verdict


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
    monkeypatch.setattr(crash, "orphan_marker_fix_missing", list)
    with pytest.raises(AssertionError, match="delete ORPHAN_MARKER_GAP"):
        test_orphan_marker_wrapper_expires_once_its_fix_is_in_the_tree()


# Review 5411223855 (F52): a runtime proof the crash file ran, beside the static selection check
# above: a lost opt-in env or any other zero-execution path also exits 0 per file.
def test_workflow_requires_the_crash_journey_manifest(tmp_path, monkeypatch):
    from tests.ci import workflow_steps

    yaml = pytest.importorskip("hermes_yaml")
    wf = yaml.safe_load((_REPO / ".github/workflows/windows-install-update-e2e.yml").read_text(encoding="utf-8-sig"))
    steps = wf["jobs"]["install-update"]["steps"]
    names = [s.get("name") for s in steps]
    run, check = steps[names.index("Run Windows install + update E2E")], steps[names.index("Every crash cell ran")]
    assert names.index("Every crash cell ran") > names.index("Run Windows install + update E2E")
    workflow_steps.required(check, {"inputs": {}})  # neither disabled nor advisory
    assert check["env"]["HERMES_E2E_ARTIFACTS"] == run["env"]["HERMES_E2E_ARTIFACTS"]
    assert f"'{crash.CELLS_RAN_MANIFEST}'" in check["run"] and "throw" in check["run"]

    j = crash.Journey(_Machine(tmp_path))
    j.step("install", lambda: "ok")
    j.step("mid_fetch", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.delenv("HERMES_E2E_ARTIFACTS", raising=False)
    crash._record_cells_ran(j)
    assert not list(tmp_path.rglob(crash.CELLS_RAN_MANIFEST))
    monkeypatch.setenv("HERMES_E2E_ARTIFACTS", str(tmp_path / "artifacts"))
    crash._record_cells_ran(j)
    assert (tmp_path / "artifacts" / crash.CELLS_RAN_MANIFEST).read_text(encoding="utf-8") == (
        "install: ok\nmid_fetch: failed\n")


# Review 5411223855 (crash-cell timing): while the hold filter blocks git, the op and index.lock
# cannot move, so a starved runner must get more wall time, never fewer looks, before the verdict.
def test_mid_git_hold_verdict_needs_looks_as_well_as_seconds(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(crash, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(crash, "_git_op", lambda proc, ops: None)  # the op is not visible yet
    monkeypatch.setattr(crash, "descendants", lambda proc: [])
    machine = SimpleNamespace(root=tmp_path, install_dir=tmp_path / "install", logs=tmp_path, evidence=lambda: "")
    hold = crash._GitHold(machine)
    hold.flags.mkdir(parents=True)
    (hold.flags / "held").write_text("1", encoding="utf-8")
    for _ in range(crash.HOLD_SETTLE_LOOKS - 1):  # each look takes 10 s on this runner
        assert hold.point(None, machine, "t") is None
        clock[0] += 10.0
    with pytest.raises(AssertionError, match="the hold filter ran but no merge/reset/checkout"):
        hold.point(None, machine, "t")


class _SharedMachine:
    """The crash journey's one machine: a checkout HEAD, a marker and an index.lock on disk."""

    def __init__(self, root: Path) -> None:
        self.logs, self.hermes_home, self.install_dir = root, root / "home", root / "install"
        (self.install_dir / ".git").mkdir(parents=True)
        self.hermes_home.mkdir()
        self.head, self.timings, self.killed = "pre", [], 0

    def evidence(self) -> str:
        return ""

    def kill_owned(self) -> None:
        self.killed += 1


def _cell_result(machine, label: str, target: str, rc: int = 0) -> dict:
    machine.head = target
    return {"label": label, "target": target, "follow_up": SimpleNamespace(returncode=rc)}


def test_a_cell_never_starts_on_the_machine_a_broken_cell_left(tmp_path, monkeypatch):
    m = _SharedMachine(tmp_path)
    monkeypatch.setattr(crash, "_tree", lambda machine, label: {
        "head": machine.head, "dirty": [], "diff_rc": 0, "diff_err": "", "target_file": True})
    monkeypatch.setattr(crash, "_marker_text", lambda machine: "dead-pid marker")

    def reset(*args, **_):
        assert args[2:5] == ("reset", "--quiet", "--hard"), args
        m.head = args[-1] if args[-1] != "HEAD" else m.head
    monkeypatch.setattr(crash, "harness_git", reset)

    seen = {}

    def sees(name):
        def run():
            seen[name] = (m.head, (m.hermes_home / crash.MARKER).exists(),
                          (m.install_dir / ".git" / "index.lock").exists())
            return _cell_result(m, name, f"{name}-target")
        return run

    def breaks():  # follow-up failed: torn tree, marker and lock left behind
        m.head = "torn"
        (m.hermes_home / crash.MARKER).write_text("1", encoding="utf-8")
        (m.install_dir / ".git" / "index.lock").write_text("", encoding="utf-8")
        return {"label": "b", "target": "b-target", "follow_up": SimpleNamespace(returncode=1)}

    def raises():
        m.head = "half"
        (m.hermes_home / crash.MARKER).write_text("1", encoding="utf-8")
        raise AssertionError("kill point not reached")

    j = crash.Journey(m)
    crash._run_cells(m, j, (("a", sees("a")), ("b", breaks), ("c", sees("c")), ("d", raises), ("e", sees("e"))))
    assert seen["a"] == ("pre", False, False)
    assert seen["c"] == ("b-target", False, False)  # reset to b's target, marker and lock gone
    assert seen["e"] == ("half", False, False)  # d raised: reset to the HEAD it left, marker gone
    assert j.ok("c") and j.ok("e") and m.killed == 2
    notes = [label for label, _ in m.timings]
    assert notes[0].startswith("(harness reset the checkout to b-target after b: its follow-up update exited rc=1")
    assert "dead-pid marker" in notes[0] and ".git/index.lock left" in notes[0]
    assert notes[1].startswith("(harness reset the checkout to HEAD after d: its step raised: kill point not reached")


def test_a_machine_the_harness_cannot_restore_fails_the_next_cell_naming_the_culprit(tmp_path, monkeypatch):
    m = _SharedMachine(tmp_path)
    monkeypatch.setattr(crash, "_tree", lambda machine, label: {
        "head": "torn", "dirty": ["M x"], "diff_rc": 1, "diff_err": "", "target_file": False})

    def reset(*args, **_):
        raise RuntimeError("fatal: index file corrupt")
    monkeypatch.setattr(crash, "harness_git", reset)
    j = crash.Journey(m)
    crash._run_cells(m, j, (("b", lambda: {"label": "b", "target": "t", "follow_up": None}),
                            ("c", lambda: pytest.fail("ran on an unsound machine"))))
    with pytest.raises(RuntimeError, match=r"not run: b left the shared machine unsound \(the checkout is not its "
                                           r"target t.*index file corrupt"):
        j["c"]
