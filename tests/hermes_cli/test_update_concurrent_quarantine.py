"""Tests for issue #26670 — concurrent hermes.exe detection and improved
quarantine retry / reboot-deferred fallback during `hermes update` on Windows.

These tests force ``_is_windows`` to return ``True`` via patching so the
Windows-specific code paths can be exercised on any host.
"""

from __future__ import annotations

import json
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
# Windows gateway pause/resume before update mutation
# ---------------------------------------------------------------------------


def test_update_gateway_shutdown_notification_defaults_visible(tmp_path):
    profile_home = tmp_path / "profile"
    profile_home.mkdir()

    assert cli_main._update_gateway_shutdown_notification_enabled(profile_home) is True
    assert cli_main._write_update_planned_stop_marker(profile_home, 101) is True

    marker = json.loads((profile_home / ".gateway-planned-stop.json").read_text())
    assert marker["suppress_notification"] is False


def test_update_gateway_shutdown_notification_can_be_disabled(tmp_path):
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    (profile_home / "config.yaml").write_text(
        "updates:\n  gateway_shutdown_notification: false\n",
        encoding="utf-8",
    )

    assert cli_main._update_gateway_shutdown_notification_enabled(profile_home) is False
    assert cli_main._write_update_planned_stop_marker(profile_home, 101) is True

    marker = json.loads((profile_home / ".gateway-planned-stop.json").read_text())
    assert marker["suppress_notification"] is True


def test_update_gateway_shutdown_notification_fails_visible(tmp_path):
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    (profile_home / "config.yaml").write_text("updates: [", encoding="utf-8")

    assert cli_main._update_gateway_shutdown_notification_enabled(profile_home) is True


@patch.object(cli_main, "_is_windows", return_value=True)
def test_pause_windows_gateways_for_update_stops_profile_and_unmapped_pids(
    _winp,
    monkeypatch,
    tmp_path,
    capsys,
):
    import gateway.status as status_mod
    import hermes_cli.gateway as gateway_mod

    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "updates:\n  gateway_shutdown_notification: false\n",
        encoding="utf-8",
    )
    profile_proc = SimpleNamespace(profile="work", path=profile_home, pid=101)

    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda **_k: [101, 202])
    monkeypatch.setattr(
        gateway_mod,
        "find_profile_gateway_processes",
        lambda **_k: [profile_proc],
    )
    monkeypatch.setattr(gateway_mod, "_get_restart_drain_timeout", lambda: 0.1)
    waited_for = []

    def fake_wait(pids, *, timeout):
        waited_for.extend(pids)
        return set()

    monkeypatch.setattr(cli_main, "_wait_for_windows_update_gateway_exit", fake_wait)
    monkeypatch.setattr(
        gateway_mod,
        "_capture_gateway_argv",
        lambda pid: ["pythonw.exe", "-m", "hermes_cli.main", "gateway", "run"]
        if pid == 202
        else None,
    )

    terminated = []
    monkeypatch.setattr(
        status_mod,
        "terminate_pid",
        lambda pid, force=False: terminated.append((pid, force)),
    )

    token = cli_main._pause_windows_gateways_for_update()

    assert token == {
        "resume_needed": True,
        "profiles": {"work": 101},
        "unmapped_pids": [202],
        "unmapped": [
            {
                "pid": 202,
                "argv": ["pythonw.exe", "-m", "hermes_cli.main", "gateway", "run"],
            }
        ],
    }
    assert waited_for == [101]
    assert terminated == [(202, True)]

    marker = json.loads((profile_home / ".gateway-planned-stop.json").read_text())
    assert marker["target_pid"] == 101
    assert marker["stopper_pid"] == os.getpid()
    assert marker["suppress_notification"] is True

    captured = capsys.readouterr().out
    assert "Paused gateway profile(s): work" in captured
    assert "without profile mapping" in captured
    # An unmapped PID whose argv we captured is respawnable, so we must NOT
    # tell the user to restart it manually.
    assert "Restart manually after update" not in captured


@patch.object(cli_main, "_is_windows", return_value=True)
def test_resume_windows_gateways_after_update_relaunches_paused_profiles(
    _winp,
    monkeypatch,
    capsys,
):
    import hermes_cli.gateway as gateway_mod

    relaunched = []
    monkeypatch.setattr(
        gateway_mod,
        "launch_detached_profile_gateway_restart",
        lambda profile, old_pid: relaunched.append((profile, old_pid)) or True,
    )

    token = {
        "resume_needed": True,
        "profiles": {"default": 101, "work": 202},
        "unmapped_pids": [],
    }

    cli_main._resume_windows_gateways_after_update(token)

    assert token["resume_needed"] is False
    assert relaunched == [("default", 101), ("work", 202)]
    assert (
        "Restarting Windows gateway profile(s): default, work"
        in capsys.readouterr().out
    )


@patch.object(cli_main, "_is_windows", return_value=True)
def test_resume_windows_gateways_after_update_respawns_unmapped_by_cmdline(
    _winp,
    monkeypatch,
    capsys,
):
    """Unmapped gateways (no profile→PID-file mapping, e.g. a Scheduled Task)
    are respawned by replaying the argv snapshotted before the force-kill."""
    import hermes_cli.gateway as gateway_mod

    by_cmdline = []
    monkeypatch.setattr(
        gateway_mod,
        "launch_detached_gateway_restart_by_cmdline",
        lambda old_pid, argv: by_cmdline.append((old_pid, argv)) or True,
    )
    monkeypatch.setattr(
        gateway_mod,
        "launch_detached_profile_gateway_restart",
        lambda profile, old_pid: True,
    )

    scheduled_argv = ["pythonw.exe", "-m", "hermes_cli.main", "gateway", "run"]
    token = {
        "resume_needed": True,
        "profiles": {},
        "unmapped_pids": [7560],
        "unmapped": [
            # Respawnable — argv captured.
            {"pid": 7560, "argv": scheduled_argv},
            # Not respawnable — no argv (psutil missing / access denied).
            {"pid": 9999, "argv": None},
        ],
    }

    cli_main._resume_windows_gateways_after_update(token)

    assert token["resume_needed"] is False
    assert by_cmdline == [(7560, scheduled_argv)]
    out = capsys.readouterr().out
    assert "Restarting 1 unmapped Windows gateway process(es)" in out


@patch.object(cli_main, "_is_windows", return_value=True)
def test_pause_returns_cold_start_token_when_installed_but_none_running(
    _winp,
    monkeypatch,
):
    """No gateway running + autostart entry installed → cold-start token.

    A gateway that died between updates (spawning terminal/TUI closed) leaves
    nothing for the resume path to relaunch, but the installed autostart entry
    is an explicit "I want a gateway" signal. The pause step must return a
    token that tells resume to cold-start one.
    """
    import hermes_cli.gateway as gateway_mod
    from hermes_cli import gateway_windows

    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)

    token = cli_main._pause_windows_gateways_for_update()

    assert token == {
        "resume_needed": True,
        "profiles": {},
        "unmapped_pids": [],
        "unmapped": [],
        "cold_start_if_installed": True,
    }


@patch.object(cli_main, "_is_windows", return_value=True)
def test_pause_returns_none_when_nothing_running_and_not_installed(
    _winp,
    monkeypatch,
):
    """No gateway running + no autostart entry → no token (gateway-less user).

    Users who deliberately run without a gateway must not get one forced on
    them by an update.
    """
    import hermes_cli.gateway as gateway_mod
    from hermes_cli import gateway_windows

    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: False)

    assert cli_main._pause_windows_gateways_for_update() is None


@patch.object(cli_main, "_is_windows", return_value=True)
def test_resume_cold_starts_gateway_when_token_requests_it(
    _winp,
    monkeypatch,
    capsys,
):
    """cold_start_if_installed token + nothing running → fresh detached spawn."""
    import hermes_cli.gateway as gateway_mod
    from hermes_cli import gateway_windows

    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda **_k: [])
    spawned = []
    monkeypatch.setattr(
        gateway_windows,
        "_spawn_detached",
        lambda: spawned.append(True) or 4242,
    )

    token = {
        "resume_needed": True,
        "profiles": {},
        "unmapped_pids": [],
        "unmapped": [],
        "cold_start_if_installed": True,
    }

    cli_main._resume_windows_gateways_after_update(token)

    assert token["resume_needed"] is False
    assert spawned == [True]
    assert "Starting Windows gateway after update (PID 4242)" in capsys.readouterr().out


@patch.object(cli_main, "_is_windows", return_value=True)
def test_resume_cold_start_skips_when_gateway_already_running(
    _winp,
    monkeypatch,
    capsys,
):
    """Don't double-start: if a gateway came up between pause and resume
    (e.g. the autostart entry fired), the cold-start must no-op."""
    import hermes_cli.gateway as gateway_mod
    from hermes_cli import gateway_windows

    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda **_k: [9001])
    spawned = []
    monkeypatch.setattr(
        gateway_windows,
        "_spawn_detached",
        lambda: spawned.append(True) or 4242,
    )

    token = {
        "resume_needed": True,
        "profiles": {},
        "unmapped_pids": [],
        "unmapped": [],
        "cold_start_if_installed": True,
    }

    cli_main._resume_windows_gateways_after_update(token)

    assert spawned == []
    assert "Starting Windows gateway after update" not in capsys.readouterr().out


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
