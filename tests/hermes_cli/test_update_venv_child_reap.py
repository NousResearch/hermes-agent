"""Tests for the Windows venv-lock wedge fix (July 2026 follow-up).

When ``hermes update`` pauses the gateway, the gateway's venv-resident helper
children (stdio MCP servers, ``hermes-perfmon``) are NOT reaped on Windows —
they survive as orphans holding native ``.pyd`` files locked, so the
venv-process guard refuses the update forever. This covers the fix:

1. ``_process_holds_install_venv`` — the shared venv-holder predicate.
2. ``_snapshot_gateway_venv_children`` — capture the gateway's venv children
   before the gateway is stopped (excluding the supervised desktop backend).
3. ``_terminate_gateway_venv_children`` — reap the captured helpers.
4. ``_sweep_stale_update_marker`` / ``_pid_is_alive`` — clear a stale
   ``.hermes-update-in-progress`` marker left by a killed ``hermes-setup.exe``.

All Windows-specific paths are exercised via ``_is_windows`` patching so they
run on any host (same approach as ``test_update_venv_health``).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import main as cli_main


# ---------------------------------------------------------------------------
# _process_holds_install_venv (shared predicate)
# ---------------------------------------------------------------------------


def _prefixes(tmp_path):
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path):
        return cli_main._venv_lock_prefixes()


def test_predicate_flags_exe_under_venv(tmp_path):
    venv_prefix, root_prefix = _prefixes(tmp_path)
    exe = str(tmp_path / "venv" / "Scripts" / "python.exe")
    assert cli_main._process_holds_install_venv(exe, "python.exe x", "", venv_prefix, root_prefix)


def test_predicate_flags_trampoline_by_cmdline(tmp_path):
    venv_prefix, root_prefix = _prefixes(tmp_path)
    base_py = "C:\\Python311\\python.exe"
    venv_path = str(tmp_path / "venv" / "Scripts" / "python.exe")
    assert cli_main._process_holds_install_venv(
        base_py, f"{base_py} {venv_path} -m x", "", venv_prefix, root_prefix
    )


def test_predicate_ignores_unrelated_python(tmp_path):
    venv_prefix, root_prefix = _prefixes(tmp_path)
    assert not cli_main._process_holds_install_venv(
        "C:\\Python311\\python.exe", "python.exe somescript.py", "C:\\other", venv_prefix, root_prefix
    )


def test_predicate_empty_exe_is_false(tmp_path):
    venv_prefix, root_prefix = _prefixes(tmp_path)
    assert not cli_main._process_holds_install_venv(None, "whatever", "", venv_prefix, root_prefix)


# ---------------------------------------------------------------------------
# _snapshot_gateway_venv_children
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, pid, *, exe="", cmdline=None, cwd="", name="python.exe", children=None):
        self.pid = pid
        self._exe = exe
        self._cmdline = cmdline or []
        self._cwd = cwd
        self._name = name
        self._children = children or []

    def exe(self):
        return self._exe

    def cmdline(self):
        return self._cmdline

    def cwd(self):
        return self._cwd

    def name(self):
        return self._name

    def parents(self):
        return []

    def children(self, recursive=False):
        return self._children


def _fake_psutil(by_pid, *, pid_exists=None):
    def Process(pid=None):
        if pid is None:
            return _FakeProc(-1)  # "current" process: parents() -> []
        proc = by_pid.get(int(pid))
        if proc is None:
            raise RuntimeError("no such process")
        return proc

    ns = types.SimpleNamespace(Process=Process)
    if pid_exists is not None:
        ns.pid_exists = pid_exists
    return ns


def test_snapshot_off_windows_is_empty():
    with patch.object(cli_main, "_is_windows", return_value=False):
        assert cli_main._snapshot_gateway_venv_children([1]) == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_snapshot_captures_venv_children_excludes_others_and_backend(_win, tmp_path):
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    other_py = "C:\\Python311\\python.exe"

    mcp = _FakeProc(2000, exe=venv_py, cmdline=[venv_py, "client_lookup_mcp.py"], name="python.exe")
    perfmon = _FakeProc(2001, exe=venv_py, cmdline=[venv_py, "hermes-perfmon.py"], name="python.exe")
    unrelated = _FakeProc(2002, exe=other_py, cmdline=[other_py, "vs-code.py"], cwd="C:\\code")
    # A desktop backend that (wrongly) appears under the gateway must NOT be reaped.
    backend = _FakeProc(2003, exe=venv_py, cmdline=[venv_py, "-m", "hermes_cli.main", "serve"])

    gateway = _FakeProc(1000, children=[mcp, perfmon, unrelated, backend])
    fake = _fake_psutil({1000: gateway})

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(sys.modules, {"psutil": fake}):
        captured = cli_main._snapshot_gateway_venv_children([1000])

    assert sorted(pid for pid, _n, _c in captured) == [2000, 2001]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_snapshot_captures_helpers_with_serve_in_their_name(_win, tmp_path):
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    mcp = _FakeProc(2000, exe=venv_py, cmdline=[venv_py, "-m", "mcp-server-time"])
    server = _FakeProc(2001, exe=venv_py, cmdline=[venv_py, "server.py"])
    gateway = _FakeProc(1000, children=[mcp, server])
    fake = _fake_psutil({1000: gateway})

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(sys.modules, {"psutil": fake}):
        captured = cli_main._snapshot_gateway_venv_children([1000])

    assert sorted(pid for pid, _n, _c in captured) == [2000, 2001]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_snapshot_dedups_across_gateways(_win, tmp_path):
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    shared = _FakeProc(2000, exe=venv_py, cmdline=[venv_py, "mcp.py"])
    g1 = _FakeProc(1000, children=[shared])
    g2 = _FakeProc(1001, children=[shared])
    fake = _fake_psutil({1000: g1, 1001: g2})

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(sys.modules, {"psutil": fake}):
        captured = cli_main._snapshot_gateway_venv_children([1000, 1001])

    assert [pid for pid, _n, _c in captured] == [2000]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_snapshot_survives_dead_gateway(_win, tmp_path):
    """A gateway PID that no longer exists must not raise, just be skipped."""
    fake = _fake_psutil({})  # Process(1000) raises
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(sys.modules, {"psutil": fake}):
        assert cli_main._snapshot_gateway_venv_children([1000]) == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_snapshot_no_psutil_is_empty(_win, tmp_path):
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(sys.modules, {"psutil": None}):
        assert cli_main._snapshot_gateway_venv_children([1000]) == []


# ---------------------------------------------------------------------------
# _terminate_gateway_venv_children
# ---------------------------------------------------------------------------


def test_terminate_off_windows_is_empty():
    with patch.object(cli_main, "_is_windows", return_value=False):
        assert cli_main._terminate_gateway_venv_children([(1, "p", "c")]) == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_terminate_empty_is_empty(_win):
    assert cli_main._terminate_gateway_venv_children([]) == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_terminate_force_kills_each_child(_win):
    fake_term = MagicMock()
    with patch("gateway.status.terminate_pid", fake_term):
        asked = cli_main._terminate_gateway_venv_children(
            [(2000, "python.exe", "mcp"), (2001, "python.exe", "perfmon")],
            wait_timeout=0.0,
        )
    assert sorted(asked) == [2000, 2001]
    killed = sorted(call.args[0] for call in fake_term.call_args_list)
    assert killed == [2000, 2001]
    assert all(call.kwargs.get("force") is True for call in fake_term.call_args_list)


@patch.object(cli_main, "_is_windows", return_value=True)
def test_terminate_is_best_effort_on_error(_win):
    """A kill that raises must not abort the reap of the remaining children."""
    def boom(pid, force=False):
        if int(pid) == 2000:
            raise OSError("access denied")

    with patch("gateway.status.terminate_pid", side_effect=boom):
        asked = cli_main._terminate_gateway_venv_children(
            [(2000, "python.exe", "a"), (2001, "python.exe", "b")],
            wait_timeout=0.0,
        )
    # 2000 raised -> not counted; 2001 still reaped.
    assert asked == [2001]


# ---------------------------------------------------------------------------
# _pid_is_alive
# ---------------------------------------------------------------------------


def test_pid_is_alive_uses_psutil():
    fake = types.SimpleNamespace(pid_exists=lambda p: int(p) == 42)
    with patch.dict(sys.modules, {"psutil": fake}):
        assert cli_main._pid_is_alive(42) is True
        assert cli_main._pid_is_alive(7) is False


def test_pid_is_alive_conservative_without_psutil():
    with patch.dict(sys.modules, {"psutil": None}):
        assert cli_main._pid_is_alive(12345) is True


# ---------------------------------------------------------------------------
# _sweep_stale_update_marker
# ---------------------------------------------------------------------------


def _write_marker(home, body):
    (home / ".hermes-update-in-progress").write_text(body, encoding="utf-8")


@patch.object(cli_main, "_is_windows", return_value=True)
def test_sweep_removes_dead_owner_marker(_win, tmp_path):
    _write_marker(tmp_path, "999999\n1700000000\n")
    with patch("hermes_constants.get_hermes_home", return_value=tmp_path), patch.object(
        cli_main, "_pid_is_alive", return_value=False
    ):
        cli_main._sweep_stale_update_marker()
    assert not (tmp_path / ".hermes-update-in-progress").exists()


@patch.object(cli_main, "_is_windows", return_value=True)
def test_sweep_preserves_live_owner_marker(_win, tmp_path):
    _write_marker(tmp_path, "4321\n1700000000\n")
    with patch("hermes_constants.get_hermes_home", return_value=tmp_path), patch.object(
        cli_main, "_pid_is_alive", return_value=True
    ):
        cli_main._sweep_stale_update_marker()
    assert (tmp_path / ".hermes-update-in-progress").exists()


@patch.object(cli_main, "_is_windows", return_value=True)
def test_sweep_removes_malformed_marker(_win, tmp_path):
    """An unparseable pid line can't name a live owner -> treat as stale."""
    _write_marker(tmp_path, "not-a-pid\n")
    with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        cli_main._sweep_stale_update_marker()
    assert not (tmp_path / ".hermes-update-in-progress").exists()


@patch.object(cli_main, "_is_windows", return_value=True)
def test_sweep_absent_marker_is_noop(_win, tmp_path):
    with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        cli_main._sweep_stale_update_marker()  # must not raise
    assert not (tmp_path / ".hermes-update-in-progress").exists()


@patch.object(cli_main, "_is_windows", return_value=False)
def test_sweep_off_windows_is_noop(_win, tmp_path):
    _write_marker(tmp_path, "999999\n1\n")
    with patch("hermes_constants.get_hermes_home", return_value=tmp_path), patch.object(
        cli_main, "_pid_is_alive", return_value=False
    ):
        cli_main._sweep_stale_update_marker()
    # POSIX behavior unchanged: the marker is left exactly as-is.
    assert (tmp_path / ".hermes-update-in-progress").exists()
