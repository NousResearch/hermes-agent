"""Tests for issue #26670 — concurrent hermes.exe detection and improved
quarantine retry / reboot-deferred fallback during `hermes update` on Windows.

These tests force ``_is_windows`` to return ``True`` via patching so the
Windows-specific code paths can be exercised on any host.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import main as cli_main


# Tests in this module either exercise the REAL _detect_concurrent_hermes_instances
# helper (and need the autouse stub in tests/hermes_cli/conftest.py disabled),
# or supply their own explicit return value via patch.object. Mark the whole
# module so the conftest fixture skips its default stub.
pytestmark = pytest.mark.real_concurrent_gate


# ---------------------------------------------------------------------------
# _detect_concurrent_hermes_instances
# ---------------------------------------------------------------------------


def _make_proc(pid: int, exe: str, name: str = "hermes.exe"):
    """Build a duck-typed psutil Process stand-in with the .info dict."""
    proc = MagicMock()
    proc.info = {"pid": pid, "exe": exe, "name": name}
    return proc


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_concurrent_returns_empty_when_no_other_processes(_winp, tmp_path):
    scripts_dir = tmp_path
    (scripts_dir / "hermes.exe").write_bytes(b"")
    (scripts_dir / "hermes-gateway.exe").write_bytes(b"")

    fake_psutil = types.SimpleNamespace(process_iter=lambda attrs: iter([]))
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_concurrent_hermes_instances(scripts_dir)

    assert result == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_concurrent_excludes_self_pid(_winp, tmp_path):
    scripts_dir = tmp_path
    shim = scripts_dir / "hermes.exe"
    shim.write_bytes(b"")
    my_pid = os.getpid()

    procs = [_make_proc(my_pid, str(shim), "hermes.exe")]
    fake_psutil = types.SimpleNamespace(process_iter=lambda attrs: iter(procs))
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_concurrent_hermes_instances(scripts_dir)

    assert result == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_concurrent_finds_other_hermes_process(_winp, tmp_path):
    scripts_dir = tmp_path
    shim = scripts_dir / "hermes.exe"
    shim.write_bytes(b"")

    other_pid = os.getpid() + 1
    procs = [
        _make_proc(other_pid, str(shim), "hermes.exe"),
        _make_proc(os.getpid() + 2, r"C:\\Windows\\System32\\notepad.exe", "notepad.exe"),
    ]
    fake_psutil = types.SimpleNamespace(process_iter=lambda attrs: iter(procs))
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_concurrent_hermes_instances(scripts_dir)

    assert result == [(other_pid, "hermes.exe")]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_concurrent_matches_case_insensitively(_winp, tmp_path):
    scripts_dir = tmp_path
    shim = scripts_dir / "hermes.exe"
    shim.write_bytes(b"")

    # Simulate the desktop spawning hermes.EXE (uppercase ext) from same path
    upper = str(shim).replace("hermes.exe", "HERMES.EXE")
    procs = [_make_proc(9999, upper, "HERMES.EXE")]
    fake_psutil = types.SimpleNamespace(process_iter=lambda attrs: iter(procs))
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_concurrent_hermes_instances(scripts_dir)

    assert result == [(9999, "HERMES.EXE")]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_concurrent_no_psutil_returns_empty(_winp, tmp_path):
    scripts_dir = tmp_path
    (scripts_dir / "hermes.exe").write_bytes(b"")

    # Block psutil import — simulate environment without it.
    with patch.dict(sys.modules, {"psutil": None}):
        result = cli_main._detect_concurrent_hermes_instances(scripts_dir)

    assert result == []


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_concurrent_is_noop_off_windows(_winp, tmp_path):
    """No process enumeration off-Windows; the file-lock issue is Windows-only."""
    assert cli_main._detect_concurrent_hermes_instances(tmp_path) == []


# ---------------------------------------------------------------------------
# Parent-chain exclusion (issue #30768 follow-up — the setuptools .exe
# launcher on Windows is a separate native process that spawns python.exe;
# excluding only ``os.getpid()`` flags the launcher as a concurrent instance.
# ---------------------------------------------------------------------------


def _fake_psutil_with_parent_chain(
    parent_chain: list[int],
    proc_iter_rows: list,
    *,
    ancestor_exe: str | None = None,
):
    """Build a psutil stand-in that has Process()/parents()/exe() AND process_iter().

    ``parent_chain`` is the ordered list of ancestor PIDs (closest first)
    returned by ``proc.parents()`` on the seed (``os.getpid()``).
    ``ancestor_exe`` is the executable path reported by each ancestor's
    ``.exe()``; when it matches one of our shim paths the ancestor is
    excluded (the launcher-shim case). Pass ``None`` to model an ancestor
    whose exe can't be read (psutil error) — it stays in the candidate set.
    """

    class _FakeProc:
        def __init__(self, pid: int, exe_path: str | None):
            self.pid = pid
            self._exe = exe_path

        def exe(self):
            if self._exe is None:
                raise OSError("exe unavailable")
            return self._exe

        def parents(self):
            return [_FakeProc(p, ancestor_exe) for p in parent_chain]

    class _NoSuchProcess(Exception):
        pass

    class _AccessDenied(Exception):
        pass

    def _process(pid=None):
        return _FakeProc(pid if pid is not None else os.getpid(), ancestor_exe)

    return types.SimpleNamespace(
        Process=_process,
        NoSuchProcess=_NoSuchProcess,
        AccessDenied=_AccessDenied,
        process_iter=lambda attrs: iter(proc_iter_rows),
    )


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_concurrent_excludes_parent_chain(_winp, tmp_path):
    """The .exe launcher (parent of os.getpid()) must NOT be flagged.

    Simulates the real Windows topology: hermes.exe launcher (PID L) spawns
    python.exe (PID os.getpid()). Both run from the same shim path. With the
    old single-PID exclusion, L would be reported as a concurrent instance.
    """
    scripts_dir = tmp_path
    shim = scripts_dir / "hermes.exe"
    shim.write_bytes(b"")
    me = os.getpid()
    launcher_pid = me + 100  # the .exe launcher — our parent

    rows = [
        _make_proc(me, str(shim), "python.exe"),
        _make_proc(launcher_pid, str(shim), "hermes.exe"),
    ]
    fake_psutil = _fake_psutil_with_parent_chain(
        parent_chain=[launcher_pid],
        proc_iter_rows=rows,
        ancestor_exe=str(shim),
    )
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_concurrent_hermes_instances(scripts_dir)

    # Both self AND the launcher are excluded; no false positive.
    assert result == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_concurrent_still_finds_unrelated_other_hermes(_winp, tmp_path):
    """A sibling hermes.exe outside our ancestor chain must still be reported."""
    scripts_dir = tmp_path
    shim = scripts_dir / "hermes.exe"
    shim.write_bytes(b"")
    me = os.getpid()
    launcher_pid = me + 100  # our .exe launcher (parent — must be excluded)
    sibling_pid = me + 200  # an UNRELATED hermes.exe (must still be reported)

    rows = [
        _make_proc(me, str(shim), "python.exe"),
        _make_proc(launcher_pid, str(shim), "hermes.exe"),
        _make_proc(sibling_pid, str(shim), "hermes.exe"),
    ]
    fake_psutil = _fake_psutil_with_parent_chain(
        parent_chain=[launcher_pid],
        proc_iter_rows=rows,
        ancestor_exe=str(shim),
    )
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_concurrent_hermes_instances(scripts_dir)

    assert result == [(sibling_pid, "hermes.exe")]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_concurrent_parent_chain_walks_deep(_winp, tmp_path):
    """Multi-level ancestry (shell → launcher → python) is fully excluded."""
    scripts_dir = tmp_path
    shim = scripts_dir / "hermes.exe"
    shim.write_bytes(b"")
    me = os.getpid()
    parent_pid = me + 1
    grandparent_pid = me + 2
    greatgrandparent_pid = me + 3

    rows = [
        _make_proc(me, str(shim), "python.exe"),
        _make_proc(parent_pid, str(shim), "hermes.exe"),
        _make_proc(grandparent_pid, str(shim), "hermes.exe"),
        _make_proc(greatgrandparent_pid, str(shim), "hermes.exe"),
    ]
    fake_psutil = _fake_psutil_with_parent_chain(
        parent_chain=[parent_pid, grandparent_pid, greatgrandparent_pid],
        proc_iter_rows=rows,
        ancestor_exe=str(shim),
    )
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_concurrent_hermes_instances(scripts_dir)

    assert result == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_concurrent_parents_call_robust_to_one_bad_hop(_winp, tmp_path):
    """The launcher shim is still excluded even when an ancestor exe is unreadable.

    Field regression (issues #29341, #34795): the old per-hop ``parent()``
    walk bailed on the FIRST psutil error, so an AccessDenied on any hop left
    the launcher shim in the candidate set and re-triggered the false
    positive. ``parents()`` returns the whole list at once; we evaluate each
    ancestor independently, so one unreadable hop never strands the launcher.
    """
    scripts_dir = tmp_path
    shim = scripts_dir / "hermes.exe"
    shim.write_bytes(b"")
    me = os.getpid()
    launcher_pid = me + 100

    rows = [
        _make_proc(me, str(shim), "python.exe"),
        _make_proc(launcher_pid, str(shim), "hermes.exe"),
    ]
    # ancestor_exe=None → every ancestor's .exe() raises OSError. The helper
    # must swallow it per-ancestor and not crash; the launcher won't be
    # excluded in this degenerate case, but a real run reads the shim exe.
    fake_psutil = _fake_psutil_with_parent_chain(
        parent_chain=[launcher_pid],
        proc_iter_rows=rows,
        ancestor_exe=None,
    )
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_concurrent_hermes_instances(scripts_dir)

    # No crash; helper completes. (Degenerate stub: launcher exe unreadable.)
    assert result == [(launcher_pid, "hermes.exe")]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_concurrent_parent_walk_handles_stub_without_process(_winp, tmp_path):
    """Partially-stubbed psutil (no Process attr) must NOT crash the helper.

    The function documents itself as "never raises"; a unit-test stub that
    only models ``process_iter`` must still complete cleanly with a sensible
    result rather than escape ``AttributeError`` to the caller.
    """
    scripts_dir = tmp_path
    shim = scripts_dir / "hermes.exe"
    shim.write_bytes(b"")
    me = os.getpid()
    other_pid = me + 1

    rows = [
        _make_proc(me, str(shim), "hermes.exe"),
        _make_proc(other_pid, str(shim), "hermes.exe"),
    ]
    # SimpleNamespace with ONLY process_iter — no Process / NoSuchProcess.
    fake_psutil = types.SimpleNamespace(process_iter=lambda attrs: iter(rows))
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_concurrent_hermes_instances(scripts_dir)

    # Parent-walk silently failed; self still excluded; other still reported.
    assert result == [(other_pid, "hermes.exe")]


# ---------------------------------------------------------------------------
# _format_concurrent_instances_message
# ---------------------------------------------------------------------------


def test_format_message_mentions_pids_and_remediation(tmp_path):
    matches = [(1234, "hermes.exe"), (5678, "hermes.exe")]
    msg = cli_main._format_concurrent_instances_message(matches, tmp_path)

    assert "1234" in msg
    assert "5678" in msg
    assert "hermes.exe" in msg
    assert "Hermes Desktop" in msg
    assert "--force" in msg
    # Mentions the file that would have been overwritten
    assert str(tmp_path / "hermes.exe") in msg
    # Self-service kill command targets the exact stale PIDs (issue #34795).
    assert "taskkill" in msg
    assert "/PID 1234" in msg
    assert "/PID 5678" in msg
    assert "/F" in msg


# ---------------------------------------------------------------------------
# _quarantine_running_hermes_exe — retry + reboot-deferred fallback
# ---------------------------------------------------------------------------


@patch.object(cli_main, "_is_windows", return_value=True)
def test_quarantine_succeeds_first_attempt(_winp, tmp_path):
    """When the rename works immediately, no warning, single rename pair returned."""
    shim = tmp_path / "hermes.exe"
    shim.write_bytes(b"old")

    pairs = cli_main._quarantine_running_hermes_exe(tmp_path)

    assert len(pairs) == 1
    orig, quarantine = pairs[0]
    assert orig == shim
    assert quarantine.name.startswith("hermes.exe.old.")
    assert quarantine.exists()
    assert not shim.exists()


@patch.object(cli_main, "_is_windows", return_value=True)
def test_quarantine_retries_then_succeeds(_winp, tmp_path, monkeypatch):
    """A transient OSError on the first attempt should not be fatal."""
    shim = tmp_path / "hermes.exe"
    shim.write_bytes(b"old")

    original_rename = Path.rename
    call_count = {"n": 0}

    def flaky_rename(self, target):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError(32, "share violation (simulated AV scan)")
        return original_rename(self, target)

    # Speed up the test: avoid actual sleeps in the backoff schedule.
    monkeypatch.setattr(cli_main, "_hermes_exe_shims", lambda d: [shim])
    with patch.object(Path, "rename", flaky_rename), patch(
        "time.sleep", lambda *_a, **_k: None
    ):
        pairs = cli_main._quarantine_running_hermes_exe(tmp_path)

    assert call_count["n"] >= 2
    assert len(pairs) == 1
    assert not shim.exists()


@patch.object(cli_main, "_is_windows", return_value=True)
def test_quarantine_falls_back_to_reboot_schedule(_winp, tmp_path, capsys, monkeypatch):
    """When every retry fails, we schedule via MoveFileEx and warn helpfully."""
    shim = tmp_path / "hermes.exe"
    shim.write_bytes(b"locked")

    def always_fails(self, target):
        raise OSError(32, "The process cannot access the file (simulated lock)")

    scheduled_calls: list[tuple[Path, Path]] = []

    def fake_schedule(s: Path, q: Path) -> bool:
        scheduled_calls.append((s, q))
        return True

    monkeypatch.setattr(cli_main, "_hermes_exe_shims", lambda d: [shim])
    with patch.object(Path, "rename", always_fails), patch.object(
        cli_main, "_schedule_replace_on_reboot", fake_schedule
    ), patch("time.sleep", lambda *_a, **_k: None):
        pairs = cli_main._quarantine_running_hermes_exe(tmp_path)

    captured = capsys.readouterr().out

    # The reboot-deferred path was used.
    assert scheduled_calls and scheduled_calls[0][0] == shim
    # It is NOT added to the returned roll-back list (the issue calls this
    # out — don't undo a deferred operation).
    assert pairs == []
    # The user got a clear message, not raw [WinError 32].
    assert "scheduled" in captured.lower()
    assert "reboot" in captured.lower()


@patch.object(cli_main, "_is_windows", return_value=True)
def test_quarantine_actionable_warning_when_everything_fails(
    _winp, tmp_path, capsys, monkeypatch
):
    """When even MoveFileEx fails we should print remediation hints, not a bare error."""
    shim = tmp_path / "hermes.exe"
    shim.write_bytes(b"locked")

    def always_fails(self, target):
        raise OSError(32, "share violation")

    monkeypatch.setattr(cli_main, "_hermes_exe_shims", lambda d: [shim])
    with patch.object(Path, "rename", always_fails), patch.object(
        cli_main, "_schedule_replace_on_reboot", lambda *_a, **_k: False
    ), patch("time.sleep", lambda *_a, **_k: None):
        pairs = cli_main._quarantine_running_hermes_exe(tmp_path)

    captured = capsys.readouterr().out
    assert pairs == []
    # New message format: no raw "[WinError 32]" dump; instead names the cause
    # and tells the user what to do.
    assert "another process" in captured.lower()
    assert "Hermes Desktop" in captured or "gateway" in captured.lower()


# ---------------------------------------------------------------------------
# cmd_update integration — concurrent-instance gate
# ---------------------------------------------------------------------------


@patch.object(cli_main, "_is_windows", return_value=True)
def test_cmd_update_aborts_on_concurrent_instance(_winp, tmp_path, capsys):
    """If another hermes.exe is running, the update bails out before
    touching the working tree (exit code 2)."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()

    args = SimpleNamespace(
        check=False,
        gateway=False,
        yes=False,
        force=False,
        backup=False,
        no_backup=True,
    )

    with patch.object(
        cli_main, "_venv_scripts_dir", return_value=scripts_dir
    ), patch.object(
        cli_main,
        "_detect_concurrent_hermes_instances",
        return_value=[(4242, "hermes.exe")],
    ), patch.object(
        cli_main, "_run_pre_update_backup"
    ) as mock_backup, patch.object(
        cli_main, "_install_hangup_protection", return_value={}
    ), patch.object(
        cli_main, "_finalize_update_output"
    ):
        with pytest.raises(SystemExit) as excinfo:
            cli_main.cmd_update(args)

    assert excinfo.value.code == 2
    # The pre-update backup runs AFTER the concurrent check; should not have
    # been invoked.
    mock_backup.assert_not_called()

    captured = capsys.readouterr().out
    assert "4242" in captured
    assert "--force" in captured


@patch.object(cli_main, "_is_windows", return_value=True)
def test_cmd_update_force_bypasses_concurrent_check(_winp, tmp_path):
    """--force lets the update proceed past the concurrent-instance gate
    (subsequent steps are mocked so we only verify the gate is skipped)."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()

    args = SimpleNamespace(
        check=False,
        gateway=False,
        yes=False,
        force=True,  # ← the bypass
        backup=False,
        no_backup=True,
    )

    detect = MagicMock(return_value=[(9, "hermes.exe")])

    # Short-circuit out of _cmd_update_impl via a sentinel raise immediately
    # AFTER the gate. _run_pre_update_backup is the first call after the gate.
    sentinel = RuntimeError("reached post-gate body")
    with patch.object(
        cli_main, "_venv_scripts_dir", return_value=scripts_dir
    ), patch.object(
        cli_main, "_detect_concurrent_hermes_instances", detect
    ), patch.object(
        cli_main, "_run_pre_update_backup", side_effect=sentinel
    ), patch.object(
        cli_main, "_install_hangup_protection", return_value={}
    ), patch.object(
        cli_main, "_finalize_update_output"
    ):
        with pytest.raises(RuntimeError, match="reached post-gate body"):
            cli_main.cmd_update(args)

    # When --force is set, we should not have even consulted psutil.
    detect.assert_not_called()


# ---------------------------------------------------------------------------
# _classify_concurrent_instance / _filter_non_gateway_concurrent_instances
#
# These helpers were added so the pre-update concurrent-instance gate can
# let the update proceed when the only concurrent hermes.exe is a gateway
# process — the post-update kill+restart block at the bottom of cmd_update
# already handles gateway PIDs by command-line match, so refusing the
# update just to make the user kill the gateway manually is friction
# without benefit.
# ---------------------------------------------------------------------------


def _make_psutil_proc_with_cmdline(pid: int, cmdline: list[str]):
    """Mock a psutil.Process with a .cmdline() method returning ``cmdline``."""
    proc = MagicMock()
    proc.cmdline = MagicMock(return_value=list(cmdline))
    return proc


@patch.object(cli_main, "_is_windows", return_value=True)
def test_classify_concurrent_instance_recognises_hermes_gateway(_winp):
    """A process whose cmdline contains the ``gateway`` keyword classifies
    as ``"gateway"`` regardless of the launcher (python -m, hermes.exe
    shim, hermes-gateway.exe shim, or direct gateway/run.py invocation)."""
    from hermes_cli.main import _classify_concurrent_instance

    cases = [
        # python -m hermes_cli.main gateway run
        ["C:/venv/Scripts/python.exe", "-m", "hermes_cli.main", "gateway", "run"],
        # hermes.exe gateway run (Windows console-script shim)
        ["C:/venv/Scripts/hermes.exe", "gateway", "run"],
        # hermes-gateway.exe direct
        ["C:/venv/Scripts/hermes-gateway.exe"],
        # direct python gateway/run.py (development launch)
        ["C:/venv/Scripts/python.exe", "gateway/run.py"],
        # case-insensitive: uppercase GATEWAY
        ["hermes.exe", "GATEWAY", "run"],
    ]
    for cmdline in cases:
        fake = types.SimpleNamespace(
            Process=lambda pid: _make_psutil_proc_with_cmdline(pid, cmdline)
        )
        with patch.dict(sys.modules, {"psutil": fake}):
            result = _classify_concurrent_instance(1234)
        assert result == "gateway", f"expected gateway for cmdline={cmdline!r}, got {result!r}"


@patch.object(cli_main, "_is_windows", return_value=True)
def test_classify_concurrent_instance_recognises_non_gateway(_winp):
    """Processes without ``gateway`` in their cmdline classify as
    ``"non-gateway"``. This is the bucket that should still trigger the
    pre-update abort (TUI shells, Desktop backend, plain REPLs)."""
    from hermes_cli.main import _classify_concurrent_instance

    cases = [
        # Interactive REPL — no gateway subcommand
        ["C:/venv/Scripts/hermes.exe"],
        # Dashboard
        ["C:/venv/Scripts/hermes.exe", "dashboard"],
        # Direct python -m hermes_cli.main (no subcommand)
        ["python", "-m", "hermes_cli.main"],
        # Empty cmdline
        [],
    ]
    for cmdline in cases:
        fake = types.SimpleNamespace(
            Process=lambda pid: _make_psutil_proc_with_cmdline(pid, cmdline)
        )
        with patch.dict(sys.modules, {"psutil": fake}):
            result = _classify_concurrent_instance(1234)
        assert result == "non-gateway", f"expected non-gateway for cmdline={cmdline!r}, got {result!r}"


@patch.object(cli_main, "_is_windows", return_value=True)
def test_classify_concurrent_instance_returns_unknown_on_psutil_error(_winp):
    """If psutil can't read the cmdline (process gone, AccessDenied on a
    session boundary, etc.) the helper returns ``"unknown"``. The pre-update
    gate treats ``"unknown"`` as non-gateway — we'd rather block an update
    we could have completed than kill something we couldn't positively
    identify as a gateway."""
    from hermes_cli.main import _classify_concurrent_instance

    class _BrokenProc:
        def cmdline(self):  # noqa: D401 — psutil Process duck type
            raise psutil.AccessDenied(pid=1234)

    fake = types.SimpleNamespace(Process=lambda pid: _BrokenProc())
    import psutil as _real_psutil
    with patch.dict(sys.modules, {"psutil": _real_psutil}):
        result = _classify_concurrent_instance(1234)
    assert result == "unknown"


def test_classify_concurrent_instance_returns_unknown_without_psutil():
    """If psutil isn't importable at all (very rare but possible in slim
    installs), the helper returns ``"unknown"`` rather than crashing."""
    from hermes_cli.main import _classify_concurrent_instance

    with patch.dict(sys.modules, {"psutil": None}):
        result = _classify_concurrent_instance(1234)
    assert result == "unknown"


@patch.object(cli_main, "_is_windows", return_value=True)
def test_filter_non_gateway_concurrent_instances_splits_correctly(_winp):
    """The filter drops PIDs whose cmdline looks like a gateway and keeps
    the rest. The pre-update gate then aborts only on the kept list."""
    from hermes_cli.main import _filter_non_gateway_concurrent_instances

    # Three PIDs: one gateway (drop), one TUI/REPL (keep), one Desktop
    # backend child (keep). The gate should only block on the two
    # non-gateway entries; the gateway is left for the post-update sweep.
    def _make_filter_psutil():
        def _proc_factory(pid: int):
            if pid == 100:
                return _make_psutil_proc_with_cmdline(
                    pid, ["hermes.exe", "gateway", "run"]
                )
            if pid == 200:
                return _make_psutil_proc_with_cmdline(
                    pid, ["hermes.exe"]  # interactive REPL
                )
            if pid == 300:
                return _make_psutil_proc_with_cmdline(
                    pid, ["hermes.exe", "dashboard"]
                )
            raise AssertionError(f"unexpected pid {pid}")

        return types.SimpleNamespace(Process=_proc_factory)

    matches = [(100, "hermes.exe"), (200, "hermes.exe"), (300, "hermes.exe")]

    with patch.dict(sys.modules, {"psutil": _make_filter_psutil()}):
        kept = _filter_non_gateway_concurrent_instances(matches)

    assert kept == [(200, "hermes.exe"), (300, "hermes.exe")]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_filter_non_gateway_concurrent_instances_drops_gateway_only(_winp):
    """If every concurrent instance is a gateway, the filter returns an
    empty list. The pre-update gate sees no non-gateway matches and lets
    the update proceed — the gateway PID will be reaped by the post-update
    kill+restart block."""
    from hermes_cli.main import _filter_non_gateway_concurrent_instances

    def _proc_factory(pid: int):
        return _make_psutil_proc_with_cmdline(
            pid, ["hermes.exe", "gateway", "run"]
        )

    matches = [(111, "hermes.exe"), (222, "hermes-gateway.exe")]

    with patch.dict(sys.modules, {"psutil": types.SimpleNamespace(Process=_proc_factory)}):
        kept = _filter_non_gateway_concurrent_instances(matches)

    assert kept == []


# ---------------------------------------------------------------------------
# cmd_update integration with the relaxed pre-update gate
# ---------------------------------------------------------------------------


@patch.object(cli_main, "_is_windows", return_value=True)
def test_cmd_update_skips_abort_when_only_concurrent_is_gateway(
    _winp, tmp_path, capsys
):
    """Regression test for the user-reported issue: when the gateway is
    the only other ``hermes.exe`` running, ``hermes update`` used to abort
    with exit code 2, forcing the user to kill all hermes.exe manually
    before retrying. The post-update kill+restart block already handles
    gateway PIDs cleanly, so the gate should now let the update proceed
    instead."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()

    args = SimpleNamespace(
        check=False,
        gateway=False,
        yes=False,
        force=False,
        backup=False,
        no_backup=True,
    )

    # Two concurrent matches, BOTH gateway — filter should return [].
    # Short-circuit at the first call after the gate so we don't have to
    # mock the entire update pipeline.
    with patch.object(
        cli_main, "_venv_scripts_dir", return_value=scripts_dir
    ), patch.object(
        cli_main,
        "_detect_concurrent_hermes_instances",
        return_value=[(1000, "hermes.exe"), (2000, "hermes-gateway.exe")],
    ), patch.object(
        cli_main, "_filter_non_gateway_concurrent_instances", return_value=[]
    ), patch.object(
        cli_main, "_run_pre_update_backup"
    ) as mock_backup, patch.object(
        cli_main, "_install_hangup_protection", return_value={}
    ), patch.object(
        cli_main, "_finalize_update_output"
    ):
        # Should NOT raise SystemExit. The sentinel that fires at the next
        # real step (pre-update backup) lets us assert we got past the
        # gate without mocking the whole pipeline.
        mock_backup.side_effect = RuntimeError("reached post-gate body")
        with pytest.raises(RuntimeError, match="reached post-gate body"):
            cli_main.cmd_update(args)

    # Pre-update backup was reached — gate let us through.
    mock_backup.assert_called_once()

    captured = capsys.readouterr().out
    assert "Another hermes.exe is running" not in captured
    assert "--force" not in captured  # the abort-message override hint


@patch.object(cli_main, "_is_windows", return_value=True)
def test_cmd_update_still_aborts_on_non_gateway_concurrent(
    _winp, tmp_path, capsys
):
    """The gate must STILL abort when an unrelated concurrent hermes.exe
    is in the list (TUI, Desktop backend, plain REPL). Only gateway-only
    matches are exempted — see test_cmd_update_skips_abort_when_only_*
    for the relaxed case."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()

    args = SimpleNamespace(
        check=False,
        gateway=False,
        yes=False,
        force=False,
        backup=False,
        no_backup=True,
    )

    with patch.object(
        cli_main, "_venv_scripts_dir", return_value=scripts_dir
    ), patch.object(
        cli_main,
        "_detect_concurrent_hermes_instances",
        return_value=[
            (1000, "hermes.exe"),  # would be classified as gateway
            (3000, "hermes.exe"),  # would be classified as non-gateway
        ],
    ), patch.object(
        cli_main,
        "_filter_non_gateway_concurrent_instances",
        return_value=[(3000, "hermes.exe")],  # simulated filter result
    ), patch.object(
        cli_main, "_run_pre_update_backup"
    ) as mock_backup, patch.object(
        cli_main, "_install_hangup_protection", return_value={}
    ), patch.object(
        cli_main, "_finalize_update_output"
    ):
        with pytest.raises(SystemExit) as excinfo:
            cli_main.cmd_update(args)

    assert excinfo.value.code == 2
    mock_backup.assert_not_called()

    captured = capsys.readouterr().out
    assert "3000" in captured
    assert "--force" in captured
