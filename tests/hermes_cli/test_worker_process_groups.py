"""Worker child-tree reclamation: a worker's children must not outlive the worker.

The failure this guards (2026-11-03, gateway cgroup incident): the terminal tool and the
``execute_code`` kernel open a new session per child, so a worker that exits, crashes, or is
SIGKILLed leaves its children re-parented to init and still running. A retried Kanban card then
runs a second copy of the same heavy job beside the first — 18.8 GB + 11.0 GB of frame decoding
pushed ``hermes-gateway.service`` past ``MemoryHigh`` and the owner stopped answering claims.

Every process here is one the test itself spawned; nothing touches an operator process.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools import worker_process_groups as wpg

pytestmark = [
    pytest.mark.skipif(os.name == "nt", reason="POSIX signals / process groups"),
    pytest.mark.live_system_guard_bypass,
]

REPO = Path(__file__).resolve().parents[2]

# A worker stand-in: arm exit-time reclaim, open one tool-style child in its own session, then
# idle. ``--marker`` puts a worker marker in the cmdline so the live-worker guard can be tested.
STANDIN = """
import os, subprocess, sys, time
sys.path.insert(0, {repo!r})
from tools.worker_process_groups import install, note_group
install(roots=[os.getcwd()])
child = subprocess.Popen(["/bin/sleep", "600"], start_new_session=True)
note_group(child.pid, os.getpgid(child.pid), label="standin child", roots=[os.getcwd()])
print("READY child=%d pgid=%d" % (child.pid, os.getpgid(child.pid)), flush=True)
while True:
    time.sleep(5)
"""


@pytest.fixture(autouse=True)
def _no_real_hermes_home(monkeypatch):
    """Never touch the real ~/.hermes: the repo's home_io_guard flags that I/O."""
    monkeypatch.setattr(wpg, "_default_roots", list)
    monkeypatch.setattr(wpg, "_ledger_dirs", lambda: [wpg.ledger_dir()])


@pytest.fixture()
def ledgers(tmp_path, monkeypatch):
    directory = tmp_path / "ledgers"
    directory.mkdir()
    monkeypatch.setenv(wpg.DIR_ENV, str(directory))
    return directory


def _alive(pid: int) -> bool:
    stat = wpg._read_stat(pid)
    return stat is not None and stat["state"] != "Z"


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def _start_standin(cwd: Path, marker: str = "") -> tuple[subprocess.Popen, int, int]:
    cwd.mkdir(parents=True, exist_ok=True)
    source = STANDIN.format(repo=str(REPO))
    if marker:
        source = f"# {marker}\n" + source
    proc = subprocess.Popen([sys.executable, "-c", source], cwd=str(cwd),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    assert line.startswith("READY"), line
    fields = dict(part.split("=", 1) for part in line.split()[1:])
    return proc, int(fields["child"]), int(fields["pgid"])


def test_ledger_is_written_before_any_child(ledgers, tmp_path):
    """An owner that loses the worker must still find a ledger naming it."""
    assert wpg.install(roots=[str(tmp_path)]) is True
    path = ledgers / f"worker-{os.getpid()}.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["worker"]["pid"] == os.getpid()
    assert str(tmp_path) in data["worker"]["roots"]


def test_sigterm_reclaims_child_group(ledgers, tmp_path):
    proc, child, pgid = _start_standin(tmp_path / "work")
    assert (ledgers / f"worker-{proc.pid}.json").exists()
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=30)
    assert _wait_gone(child), "child survived the worker's SIGTERM"
    assert not wpg._group_members(pgid)
    assert not (ledgers / f"worker-{proc.pid}.json").exists()


def test_sigkill_is_reclaimed_from_the_ledger(ledgers, tmp_path):
    """A SIGKILLed worker runs no code: only the ledger can name what it left behind."""
    proc, child, pgid = _start_standin(tmp_path / "work")
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)
    assert _alive(child), "child should outlive a SIGKILLed worker (that is the bug)"
    summary = wpg.reclaim_worker(proc.pid, grace=3.0, reason="test")
    assert summary["reclaimed"], summary
    assert _wait_gone(child)
    assert not wpg._group_members(pgid)
    assert not (ledgers / f"worker-{proc.pid}.json").exists()


def test_retry_never_runs_two_copies(ledgers, tmp_path):
    work = tmp_path / "workAE"
    first, first_child, _ = _start_standin(work)
    first.send_signal(signal.SIGKILL)
    first.wait(timeout=10)
    wpg.reclaim_worker(first.pid, grace=3.0, reason="retry")
    retry, retry_child, _ = _start_standin(work)
    try:
        assert not _alive(first_child)
        assert _alive(retry_child)
        assert [pid for pid in (first_child, retry_child) if _alive(pid)] == [retry_child]
    finally:
        retry.send_signal(signal.SIGTERM)
        retry.wait(timeout=30)


def test_scan_finds_ppid1_leak_and_sweep_kills_it(ledgers, tmp_path):
    root = tmp_path / "root"
    work = root / "workAE"
    work.mkdir(parents=True)
    # Double fork: the intermediate exits at once, so the sleeper is re-parented to init. The
    # sleeper's stdio must not inherit the pipe, or communicate() would wait for it to exit.
    out = subprocess.run(
        [sys.executable, "-c",
         "import subprocess;"
         f"p=subprocess.Popen(['/bin/sleep','600'],cwd={str(work)!r},start_new_session=True,"
         "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
         "print(p.pid)"],
        check=False, capture_output=True, text=True, timeout=30)
    leak = int(out.stdout.strip())
    try:
        rows = wpg.scan_orphans([str(root)], min_age=0.0)
        mine = [row for row in rows if row["pid"] == leak]
        # Reparenting lands on PID 1 only when nothing above us acts as a subreaper; under
        # systemd --user (and under WSL, where an ancestor sets PR_SET_CHILD_SUBREAPER) the orphan
        # is adopted by that process instead. Either way the finding is "leaked child in a
        # workspace tree", so accept both sources and assert the sweep kills it.
        assert mine and mine[0]["source"] in ("ppid1", "unowned"), rows
        # A subreaper-adopted orphan is classified "unowned" (see above), and sweep only signals
        # those when explicitly asked — so ask here. On a host where init adopts it, the row is
        # "ppid1" and this flag changes nothing.
        killed = wpg.sweep(mine, roots=[str(root)], grace=3.0, include_unowned=True)
        assert [row["pid"] for row in killed["killed"]] == [leak]
        assert _wait_gone(leak)
    finally:
        if _alive(leak):
            os.kill(leak, signal.SIGKILL)


def test_scan_skips_children_of_a_live_worker(ledgers, tmp_path):
    """A command a worker is running right now must never be swept."""
    root = tmp_path / "root"
    proc, child, _ = _start_standin(root / "work", marker="agent.managed_worker")
    try:
        rows = wpg.scan_orphans([str(root)], min_age=0.0)
        assert [row for row in rows if row["pid"] == child] == []
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)


def test_scan_does_not_report_the_live_worker_itself(ledgers, tmp_path):
    """A running worker has a workspace cwd and a long life; it must not read as an orphan."""
    root = tmp_path / "root"
    proc, child, _ = _start_standin(root / "work", marker="agent.managed_worker")
    try:
        rows = wpg.scan_orphans([str(root)], min_age=0.0)
        pids = [row["pid"] for row in rows]
        assert proc.pid not in pids and child not in pids, rows
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)


def test_scan_reports_ledger_only_group(ledgers, tmp_path):
    """cwd outside every root and leader not PPID=1: only the ledger can name the group."""
    outside = tmp_path / "outside"
    outside.mkdir()
    group = subprocess.Popen(["/bin/sleep", "600"], cwd=str(outside), start_new_session=True)
    dead = subprocess.Popen(["/bin/true"])
    dead.wait()
    ledger = ledgers / f"worker-{dead.pid}.json"
    ledger.write_text(json.dumps({
        "version": 1,
        "worker": {"pid": dead.pid, "birth": 0.0, "started": time.time(), "roots": [str(outside)]},
        "groups": [{"pgid": os.getpgid(group.pid), "leader": group.pid,
                    "leader_birth": (wpg._read_stat(group.pid) or {}).get("start", -1),
                    "label": "ledger-only", "persist": False}],
    }))
    try:
        rows = wpg.scan_orphans([str(tmp_path / "root")], min_age=0.0)
        mine = [row for row in rows if row["source"] == "ledger" and row["pgid"] == os.getpgid(group.pid)]
        assert mine, rows
        wpg.sweep(mine, roots=[str(tmp_path / "root")], grace=3.0)
        assert not wpg._group_members(os.getpgid(group.pid))
    finally:
        if _alive(group.pid):
            os.kill(group.pid, signal.SIGKILL)


def test_cli_json_is_machine_readable(ledgers, tmp_path, capsys):
    root = tmp_path / "root"
    (root / "workAE").mkdir(parents=True)
    assert wpg.main(["--json", "--root", str(root), "--min-age", "0"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
