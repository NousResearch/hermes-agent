"""Tests for the orphaned-Desktop-backend reap in the venv-holder guard.

The GUI-updater handoff race (ryanc's 2026-08-09 failures): the Desktop app
fires SIGTERM + app.quit() and spawns hermes-setup, but its Python backend
(``python.exe -m hermes_cli.main serve``) survives the teardown race. The
Desktop is gone — nothing will respawn that backend — yet the venv-holder
guard refused on it and the update dead-ended with "Hermes is still running"
while the user had zero windows open.

``_orphaned_desktop_backend_pids`` classifies holders: a ``serve``/
``dashboard`` backend whose supervising parent is provably dead is safe to
reap (with its full child tree — the managed .hermes-runtime interpreter
child included, #70026); anything else keeps the refusal.

All paths run on any host via a fake psutil module (same approach as
test_update_venv_health.py).
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli import main as cli_main
from hermes_cli import update_cmd


class _FakeNoSuchProcess(Exception):
    pass


def _fake_psutil(procs: dict[int, MagicMock]):
    """Build a psutil stand-in whose Process(pid) serves from *procs*."""

    def _process(pid: int):
        if pid not in procs:
            raise _FakeNoSuchProcess(pid)
        return procs[pid]

    return types.SimpleNamespace(
        Process=_process, NoSuchProcess=_FakeNoSuchProcess
    )


def _proc(
    pid: int,
    cmdline: list[str],
    *,
    ppid: int = 0,
    create_time: float = 100.0,
    parents: list[MagicMock] | None = None,
):
    proc = MagicMock()
    proc.pid = pid
    proc.cmdline.return_value = cmdline
    proc.ppid.return_value = ppid
    proc.create_time.return_value = create_time
    proc.is_running.return_value = True
    proc.parents.return_value = parents or []
    return proc


_SERVE_ARGV = [
    "C:\\hermes\\venv\\Scripts\\python.exe",
    "-m",
    "hermes_cli.main",
    "serve",
    "--host",
    "127.0.0.1",
]


def _holders(pid=200, cmdline="python.exe -m hermes_cli.main serve"):
    return [(pid, "python.exe", cmdline)]


# ---------------------------------------------------------------------------
# _orphaned_desktop_backend_pids classification
# ---------------------------------------------------------------------------


def test_orphan_backend_dead_parent_qualifies():
    backend = _proc(200, _SERVE_ARGV, ppid=999)  # 999 not in table → dead
    fake = _fake_psutil({200: backend})
    with patch.dict(sys.modules, {"psutil": fake}):
        assert cli_main._orphaned_desktop_backend_pids(_holders()) == [(200, 10000)]


def test_backend_with_live_parent_keeps_refusal():
    parent = _proc(50, ["Hermes.exe"], create_time=10.0)
    backend = _proc(200, _SERVE_ARGV, ppid=50, create_time=100.0)
    fake = _fake_psutil({50: parent, 200: backend})
    with patch.dict(sys.modules, {"psutil": fake}):
        assert cli_main._orphaned_desktop_backend_pids(_holders()) is None


def test_recycled_parent_pid_counts_as_orphan():
    # "Parent" created AFTER the child = PID reuse; real supervisor is dead.
    recycled = _proc(50, ["notepad.exe"], create_time=500.0)
    backend = _proc(200, _SERVE_ARGV, ppid=50, create_time=100.0)
    fake = _fake_psutil({50: recycled, 200: backend})
    with patch.dict(sys.modules, {"psutil": fake}):
        assert cli_main._orphaned_desktop_backend_pids(_holders()) == [(200, 10000)]


def test_non_backend_holder_keeps_refusal():
    repl = _proc(300, ["python.exe", "-m", "hermes_cli.main", "chat"], ppid=999)
    fake = _fake_psutil({300: repl})
    with patch.dict(sys.modules, {"psutil": fake}):
        holders = _holders(pid=300, cmdline="python.exe -m hermes_cli.main chat")
        assert cli_main._orphaned_desktop_backend_pids(holders) is None


def test_mixed_holders_keep_refusal():
    # One orphan backend + one operator REPL → the whole set is refused.
    backend = _proc(200, _SERVE_ARGV, ppid=999)
    repl = _proc(300, ["python.exe", "some_script.py"], ppid=998)
    fake = _fake_psutil({200: backend, 300: repl})
    with patch.dict(sys.modules, {"psutil": fake}):
        holders = _holders() + [(300, "python.exe", "python.exe some_script.py")]
        assert cli_main._orphaned_desktop_backend_pids(holders) is None


def test_orphan_root_plus_managed_runtime_descendant_qualifies():
    # helix4u's review case (#82179): the scanner returns BOTH the orphaned
    # serve root and its .hermes-runtime interpreter child. The child's live
    # parent IS the orphan root, so the set is safe — only the root is
    # returned (taskkill /T reaps the descendant with it).
    backend = _proc(200, _SERVE_ARGV, ppid=999)
    child_argv = [
        "C:\\hermes\\.hermes-runtime\\python\\generation-1\\python.exe",
        "worker.py",
    ]
    child = _proc(210, child_argv, ppid=200, parents=[backend])
    fake = _fake_psutil({200: backend, 210: child})
    with patch.dict(sys.modules, {"psutil": fake}):
        holders = _holders() + [(210, "python.exe", " ".join(child_argv))]
        assert cli_main._orphaned_desktop_backend_pids(holders) == [(200, 10000)]


def test_descendant_of_grandchild_depth_qualifies():
    # Descendant two hops below the orphan root (root → child → grandchild):
    # psutil.parents() walks the full chain, so ancestry still matches.
    backend = _proc(200, _SERVE_ARGV, ppid=999)
    mid = _proc(210, ["python.exe", "mid.py"], ppid=200, parents=[backend])
    grand = _proc(
        220, ["python.exe", "leaf.py"], ppid=210, parents=[mid, backend]
    )
    fake = _fake_psutil({200: backend, 210: mid, 220: grand})
    with patch.dict(sys.modules, {"psutil": fake}):
        holders = _holders() + [(220, "python.exe", "python.exe leaf.py")]
        assert cli_main._orphaned_desktop_backend_pids(holders) == [(200, 10000)]


def test_non_descendant_alongside_orphan_root_keeps_refusal():
    # A stray process that is NOT under the orphan root disqualifies the set
    # even though an orphan root exists.
    backend = _proc(200, _SERVE_ARGV, ppid=999)
    unrelated_parent = _proc(50, ["explorer.exe"])
    stray = _proc(
        300, ["python.exe", "stray.py"], ppid=50, parents=[unrelated_parent]
    )
    fake = _fake_psutil({50: unrelated_parent, 200: backend, 300: stray})
    with patch.dict(sys.modules, {"psutil": fake}):
        holders = _holders() + [(300, "python.exe", "python.exe stray.py")]
        assert cli_main._orphaned_desktop_backend_pids(holders) is None


def test_descendant_exited_between_scan_and_classify_is_skipped():
    backend = _proc(200, _SERVE_ARGV, ppid=999)
    fake = _fake_psutil({200: backend})  # descendant 210 already gone
    with patch.dict(sys.modules, {"psutil": fake}):
        holders = _holders() + [(210, "python.exe", "python.exe worker.py")]
        assert cli_main._orphaned_desktop_backend_pids(holders) == [(200, 10000)]


def test_holder_gone_between_scan_and_classify_is_skipped():
    fake = _fake_psutil({})  # PID vanished entirely
    with patch.dict(sys.modules, {"psutil": fake}):
        assert cli_main._orphaned_desktop_backend_pids(_holders()) == []


def test_missing_psutil_keeps_refusal():
    import builtins

    real_import = builtins.__import__

    def _no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    with patch.dict(sys.modules, {"psutil": None}), patch.object(
        builtins, "__import__", _no_psutil
    ):
        assert cli_main._orphaned_desktop_backend_pids(_holders()) is None


# ---------------------------------------------------------------------------
# _stop_process_trees
# ---------------------------------------------------------------------------


def test_stop_process_trees_kills_full_tree():
    from hermes_cli import update_cmd

    with patch("gateway.status.get_process_start_time", return_value=123), patch(
        "hermes_cli._subprocess_compat.pid_is_hermes", return_value=True
    ), patch.object(update_cmd.subprocess, "run") as run:
        cli_main._stop_process_trees([111, 222])
    calls = [c.args[0] for c in run.call_args_list]
    assert calls == [
        ["taskkill", "/PID", "111", "/T", "/F"],
        ["taskkill", "/PID", "222", "/T", "/F"],
    ]


def test_stop_process_trees_never_raises():
    from hermes_cli import update_cmd

    with patch.object(
        update_cmd.subprocess, "run", side_effect=OSError("no taskkill")
    ):
        cli_main._stop_process_trees([111])  # must not raise


# ---------------------------------------------------------------------------
# Guard integration: orphan reap clears the dead-end
# ---------------------------------------------------------------------------


def _update_args(**overrides):
    defaults = dict(
        gateway=False,
        check=False,
        no_backup=True,
        backup=False,
        yes=True,
        branch=None,
        force=False,
        force_venv=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _run_guard(detect_side_effect, orphan_return):
    """Drive _cmd_update_impl to the venv-holder guard (harness mirrors
    test_update_venv_health.py)."""

    class _PastGuard(Exception):
        pass

    class _RootSentinel:
        def __truediv__(self, _other):
            raise _PastGuard

    killed: list[list[int]] = []

    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(cli_main, "_run_pre_update_backup"), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main, "_resume_windows_gateways_after_update"
    ), patch.object(
        cli_main, "_detect_venv_python_processes", side_effect=detect_side_effect
    ), patch.object(
        cli_main, "_leftover_pausable_gateway_pids", return_value=None
    ), patch.object(
        cli_main, "_orphaned_desktop_backend_pids", return_value=orphan_return
    ), patch.object(
        cli_main, "_stop_process_trees", side_effect=killed.append
    ), patch.object(
        cli_main, "PROJECT_ROOT", _RootSentinel()
    ), patch(
        "time.sleep"
    ):
        try:
            update_cmd._cmd_update_impl(_update_args(), gateway_mode=False)
        except _PastGuard:
            return "past_guard", killed
        except SystemExit as exc:
            return f"exit_{exc.code}", killed
    return "returned", killed


def test_guard_reaps_orphan_backend_and_proceeds():
    holders = _holders()
    # 1st scan: backend present; 2nd (post-reap) scan: clear.
    result, killed = _run_guard(
        detect_side_effect=[holders, []], orphan_return=[200]
    )
    assert result == "past_guard"
    assert killed == [[200]]


def test_guard_still_refuses_when_not_orphaned():
    holders = _holders()
    result, killed = _run_guard(
        detect_side_effect=[holders, holders], orphan_return=None
    )
    assert result == "exit_2"
    assert killed == []


def test_guard_refuses_when_reap_does_not_clear_holders():
    holders = _holders()
    # Reap runs but a holder survives (unkillable child) → refuse.
    result, killed = _run_guard(
        detect_side_effect=[holders, holders], orphan_return=[200]
    )
    assert result == "exit_2"
    assert killed == [[200]]
