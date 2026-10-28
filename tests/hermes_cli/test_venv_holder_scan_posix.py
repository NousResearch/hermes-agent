"""POSIX live-venv holder detection (#134218).

Regression scope: the detector the updater consults returned ``[]`` on every non-Windows
host, so a gateway still executing from the venv was invisible while the updater replaced
it. These tests pin the POSIX signal: holders are found by env, by cmdline and by exe,
the caller's own process chain never counts itself, and unreadable entries are skipped
instead of raising.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from hermes_cli import venv_holder_scan as vhs

pytestmark = pytest.mark.skipif(not Path("/proc").is_dir(), reason="POSIX /proc scan only")


def _make_venv(tmp_path: Path) -> Path:
    """A directory that looks like a venv to the scanner (a pyvenv.cfg is the marker)."""
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("version_info = 3.11.0\n", encoding="utf-8")
    return venv


def _spawn_holder(*argv: str, env_extra: dict[str, str] | None = None) -> subprocess.Popen:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.Popen([*argv, "-c", "import time; time.sleep(30)"], env=env)


def _wait_for(fn, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    return fn()


def test_holder_is_found_by_virtual_env(tmp_path):
    venv = _make_venv(tmp_path)
    proc = _spawn_holder(sys.executable, env_extra={"VIRTUAL_ENV": str(venv)})
    try:
        pids = _wait_for(lambda: [p for p in vhs.pids_holding_venv(venv) if p == proc.pid])
        assert pids, "a process whose VIRTUAL_ENV points at the venv must be reported as a holder"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_holder_is_found_by_cmdline(tmp_path):
    """A gateway runs ``<venv>/bin/python -m hermes_cli.main ...``: argv is the only clue."""
    venv = _make_venv(tmp_path)
    launcher = venv / "bin" / "python"
    try:
        launcher.symlink_to(sys.executable)
    except OSError:
        pytest.skip("no symlink privilege")
    proc = _spawn_holder(str(launcher))
    try:
        match = _wait_for(lambda: [p for p in vhs.pids_holding_venv(venv) if p == proc.pid])
        assert match, "a process launched from <venv>/bin/python must be reported as a holder"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_unrelated_process_is_not_a_holder(tmp_path):
    venv = _make_venv(tmp_path)
    proc = _spawn_holder(sys.executable)
    try:
        time.sleep(0.2)
        assert proc.pid not in vhs.pids_holding_venv(venv)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_own_process_chain_is_excluded_but_can_be_requested(tmp_path, monkeypatch):
    """The updater lives under the venv it replaces; counting itself would deadlock it."""
    venv = _make_venv(tmp_path)
    assert os.getpid() in vhs.self_and_ancestor_pids()  # the real chain includes us
    proc = _spawn_holder(sys.executable, env_extra={"VIRTUAL_ENV": str(venv)})
    monkeypatch.setattr(vhs, "self_and_ancestor_pids", lambda *a, **k: {proc.pid})
    try:
        _wait_for(lambda: vhs.pids_holding_venv(venv, include_ancestors=True))
        assert proc.pid not in vhs.pids_holding_venv(venv), "own chain must never be a holder"
        assert proc.pid in vhs.pids_holding_venv(venv, include_ancestors=True)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_exclude_pids_is_honoured(tmp_path):
    venv = _make_venv(tmp_path)
    proc = _spawn_holder(sys.executable, env_extra={"VIRTUAL_ENV": str(venv)})
    try:
        _wait_for(lambda: [p for p in vhs.pids_holding_venv(venv) if p == proc.pid])
        assert proc.pid not in vhs.pids_holding_venv(venv, exclude_pids={proc.pid})
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_missing_venv_dir_reports_no_holders(tmp_path):
    assert vhs.pids_holding_venv(tmp_path / "nope") == []


def test_scan_never_raises_on_vanished_or_unreadable_pids(tmp_path, monkeypatch):
    venv = _make_venv(tmp_path)
    real_listdir = os.listdir

    def fake_listdir(path):
        if str(path) == "/proc":
            return ["999999999", "self", "uptime", "12345"]
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", fake_listdir)
    assert vhs.pids_holding_venv(venv) == []  # bogus/vanished entries are skipped, not fatal


def _patch_detector_env(monkeypatch, tmp_path: Path, venv: Path) -> None:
    """Point the detector at a fake checkout.

    Patch ``update_cmd._m`` (the detector's own import site) instead of importing
    ``hermes_cli.main``: main's import touches the real hermes home, which the repo's
    ``home_io_guard`` rightly refuses inside tests.
    """
    from hermes_cli import update_cmd as ucmd

    stub = types.SimpleNamespace(_is_windows=lambda: False, PROJECT_ROOT=tmp_path)
    monkeypatch.setattr(ucmd, "_m", lambda: stub)
    monkeypatch.setattr("hermes_constants.project_venv_dir", lambda _root: venv)


def test_detector_reports_posix_holders(monkeypatch, tmp_path):
    """The signal the updater consumes must be non-empty on POSIX (#134218)."""
    from hermes_cli import update_cmd_windows as ucw

    venv = _make_venv(tmp_path)
    _patch_detector_env(monkeypatch, tmp_path, venv)
    proc = _spawn_holder(sys.executable, env_extra={"VIRTUAL_ENV": str(venv)})
    try:
        rows = _wait_for(lambda: [r for r in ucw._detect_venv_python_processes() if r[0] == proc.pid])
        assert rows, "off-Windows the detector must report live holders instead of an empty list"
        pid, name, cmdline = rows[0]
        assert pid == proc.pid and name and cmdline
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_detector_shape_matches_windows_contract(monkeypatch, tmp_path):
    """``(pid, name, cmdline)`` with a parseable full cmdline — callers look for 'gateway run'."""
    from hermes_cli import update_cmd_windows as ucw

    venv = _make_venv(tmp_path)
    _patch_detector_env(monkeypatch, tmp_path, venv)
    launcher = venv / "bin" / "python"
    try:
        launcher.symlink_to(sys.executable)
    except OSError:
        pytest.skip("no symlink privilege")
    proc = subprocess.Popen(
        [str(launcher), "-c", "import time; time.sleep(30)", "gateway", "run"],
        env=dict(os.environ))
    try:
        rows = _wait_for(lambda: [r for r in ucw._detect_venv_python_processes() if r[0] == proc.pid])
        assert rows, "holder must be reported"
        _, _, cmdline = rows[0]
        assert "gateway run" in cmdline
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _patch_chain(monkeypatch, venv: Path, *, gateway_cmd: str) -> tuple[int, int, dict]:
    """Fake a parent chain: one gateway-shaped ancestor, one interactive ancestor.

    Returns ``(gateway_pid, shell_pid, seen)`` where ``seen`` records what the scan asked
    its own helpers for. ``hermes_cli.venv_holder_scan`` is patched rather than the detector,
    because the detector imports these names at call time.
    """
    gateway_pid, shell_pid = 424242, 434343
    seen: dict = {}
    monkeypatch.setattr(
        vhs, "self_and_ancestor_pids", lambda pid=None: {os.getpid(), gateway_pid, shell_pid}
    )
    monkeypatch.setattr(
        vhs,
        "proc_cmdline",
        lambda pid: {gateway_pid: gateway_cmd, shell_pid: "/bin/bash -l"}.get(pid, ""),
    )
    monkeypatch.setattr(vhs, "proc_name", lambda pid: "python")

    def fake_pids_holding_venv(venv_dir, *, exclude_pids=None, include_ancestors=False):
        seen["exclude"] = set(exclude_pids or set())
        seen["include_ancestors"] = include_ancestors
        return [gateway_pid]

    monkeypatch.setattr(vhs, "pids_holding_venv", fake_pids_holding_venv)
    return gateway_pid, shell_pid, seen


def test_gateway_ancestor_stays_visible_on_posix(monkeypatch, tmp_path):
    """#87594: under /update the updater is the gateway's child.

    Blanket ancestor-exclusion would hide the very process the pause machinery must see —
    on POSIX that is how a live gateway ends up with its venv replaced underneath it.
    """
    from hermes_cli import update_cmd_windows as ucw

    venv = _make_venv(tmp_path)
    _patch_detector_env(monkeypatch, tmp_path, venv)
    gateway_pid, shell_pid, seen = _patch_chain(
        monkeypatch, venv, gateway_cmd=f"{venv}/bin/python -m hermes_cli.main gateway run"
    )

    rows = ucw._detect_venv_python_processes()

    assert [row[0] for row in rows] == [gateway_pid], "a gateway ancestor must remain a visible holder"
    assert os.getpid() in seen["exclude"], "the updater never nominates itself"
    assert shell_pid in seen["exclude"], "interactive ancestry is never a blocker"
    assert gateway_pid not in seen["exclude"], "gateway ancestry must not be blanket-excluded (#87594)"
    assert seen["include_ancestors"] is True, "the chain decision is made once, here — not twice"


def test_caller_exclude_pids_is_unioned_with_own_chain(monkeypatch, tmp_path):
    """An explicit ``exclude_pids`` from the caller survives the #87594 carve-out."""
    from hermes_cli import update_cmd_windows as ucw

    venv = _make_venv(tmp_path)
    _patch_detector_env(monkeypatch, tmp_path, venv)
    _, _, seen = _patch_chain(monkeypatch, venv, gateway_cmd="/bin/sh -c true")

    ucw._detect_venv_python_processes(exclude_pids={777})

    assert 777 in seen["exclude"] and os.getpid() in seen["exclude"]
