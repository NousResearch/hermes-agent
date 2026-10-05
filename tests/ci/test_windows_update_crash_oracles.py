"""The Windows crash cells' oracles, judged on this OS: a verdict the cells can only reach on a
Windows runner must still say pass exactly when the user's update was what the cell claims."""

import subprocess
from pathlib import Path

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
