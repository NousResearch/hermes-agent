"""Tests for the dashboard chat/kernel reap in the venv-holder guard.

``hermes dashboard`` hosts its embedded chat through ``python -m
tui_gateway.entry`` and each agent's tool kernel through
``<tmpdir>/hermes_kernel``, both on the install venv's interpreter. The
Status page that starts an update lives in that same app, so the guard's
refusal ("close Hermes Desktop / other Hermes terminals") named a process
the user could not close from where they were standing: pressing Update on
the dashboard dead-ended with exit 2 every time, with its own chat pane on
the list of blockers.

``_respawnable_dashboard_holder_pids`` classifies those holders: a holder
whose cmdline carries ``tui_gateway`` or ``hermes_kernel`` is re-created on
demand (the chat pane reconnects on reload, the next tool call spawns a
fresh kernel), so it is stopped and the scan re-runs instead of refusing.
Every other holder class keeps the refusal — the rung is a strict addition
to the existing gateway / ledger / orphan / serve / hand-off rungs.

All paths run on any host via a fake psutil module (same approach as
test_update_orphan_backend_reap.py).
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli import main as cli_main
from hermes_cli import update_cmd
from hermes_cli import update_cmd_windows


class _FakeNoSuchProcess(Exception):
    pass


def _fake_psutil(procs: dict[int, MagicMock]):
    """Build a psutil stand-in whose Process(pid) serves from *procs*."""

    def _process(pid: int):
        if pid not in procs:
            raise _FakeNoSuchProcess(pid)
        return procs[pid]

    return types.SimpleNamespace(Process=_process, NoSuchProcess=_FakeNoSuchProcess)


def _proc(pid: int, cmdline: str):
    proc = MagicMock()
    proc.pid = pid
    proc.cmdline.return_value = cmdline.split()
    return proc


def _rows(*cmdlines, first_pid=700):
    """Holder rows the way the scan returns them: (pid, name, cmdline)."""
    return [
        (first_pid + 10 * index, "python.exe", cmdline)
        for index, cmdline in enumerate(cmdlines)
    ]


_CHAT_BACKEND = r"C:\hermes\venv\Scripts\python.exe -m tui_gateway.entry"
_TOOL_KERNEL = (
    r"C:\hermes\venv\Scripts\python.exe"
    r" C:\Users\operator\AppData\Local\Temp\hermes_kernel\kernel\main.py"
)
_GATEWAY = r"C:\hermes\venv\Scripts\python.exe -m hermes_cli.main gateway run --profile deepseek"
_DASHBOARD_SERVER = r"C:\hermes\venv\Scripts\python.exe -m hermes_cli.main -p default dashboard --skip-build"
_USER_SESSION = r"C:\hermes\venv\Scripts\python.exe -m hermes_cli.main --profile deepseek"


def _classify(rows, *, live_pids=None):
    """Classify *rows* with a fake psutil; *live_pids* limits what is still running."""
    if live_pids is None:
        live_pids = {pid for pid, _name, _cmdline in rows}
    table = {pid: _proc(pid, c) for pid, _n, c in rows if pid in live_pids}
    with patch.dict(sys.modules, {"psutil": _fake_psutil(table)}):
        return update_cmd_windows._respawnable_dashboard_holder_pids(rows)


# ---------------------------------------------------------------------------
# _respawnable_dashboard_holder_pids classification
# ---------------------------------------------------------------------------


def test_chat_backend_holder_is_respawnable():
    assert _classify(_rows(_CHAT_BACKEND)) == [700]


def test_tool_kernel_holder_is_respawnable():
    assert _classify(_rows(_TOOL_KERNEL)) == [700]


def test_both_kinds_are_returned_in_scan_order():
    assert _classify(_rows(_CHAT_BACKEND, _TOOL_KERNEL, _CHAT_BACKEND)) == [
        700,
        710,
        720,
    ]


def test_match_is_case_insensitive():
    # Windows cmdlines come back in whatever case the launcher used; the
    # rung classifies the live argv, lower-cased.
    rows = _rows(
        r"C:\hermes\venv\Scripts\python.exe -m TUI_Gateway.Entry",
        r"C:\hermes\venv\Scripts\python.exe C:\Temp\Hermes_Kernel\main.py",
    )
    assert _classify(rows) == [700, 710]


def test_gateway_and_user_holders_are_left_alone():
    # The rung must not widen the set of holders the guard is willing to stop.
    assert _classify(_rows(_GATEWAY, _DASHBOARD_SERVER, _USER_SESSION)) == []


def test_empty_scan_returns_empty():
    assert _classify([]) == []


def test_exited_holder_is_skipped():
    # Classified on the live argv: a row whose process is already gone is not
    # returned (nothing to reap), and it does not take the others down.
    rows = _rows(_CHAT_BACKEND, _TOOL_KERNEL)
    assert _classify(rows, live_pids={710}) == [710]


def test_unparseable_pid_row_is_skipped():
    # Defensive: the scan is typed for int pids, but a row that cannot be
    # converted must not take the whole rung down with it.
    rows = [(object(), "python.exe", _CHAT_BACKEND)]
    assert _classify(rows) == []


def test_missing_psutil_keeps_refusal():
    with patch.dict(sys.modules, {"psutil": None}):
        assert update_cmd_windows._respawnable_dashboard_holder_pids(
            _rows(_CHAT_BACKEND)
        ) == []


# ---------------------------------------------------------------------------
# Guard integration: the dashboard reap clears the dead-end
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


def _run_guard(detect_side_effect, live_rows):
    """Drive _cmd_update_impl to the venv-holder guard (harness mirrors
    test_update_orphan_backend_reap.py / test_update_venv_health.py).

    Every other classifier rung is pinned inert so the only rung that can
    fire is the dashboard one, which runs for real against the fake psutil.
    """

    class _PastGuard(Exception):
        pass

    class _RootSentinel:
        def __truediv__(self, _other):
            raise _PastGuard

    killed: list[list[int]] = []
    table = {pid: _proc(pid, c) for pid, _n, c in live_rows}

    with patch.dict(sys.modules, {"psutil": _fake_psutil(table)}), patch.object(
        cli_main, "_is_windows", return_value=True
    ), patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(
        cli_main, "_run_pre_update_backup"
    ), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main, "_resume_windows_gateways_after_update"
    ), patch.object(
        cli_main, "_detect_venv_python_processes", side_effect=detect_side_effect
    ), patch.object(
        cli_main, "_leftover_pausable_gateway_pids", return_value=None
    ), patch.object(
        cli_main, "_ledger_reapable_backend_pids", return_value=None
    ), patch.object(
        cli_main, "_orphaned_desktop_backend_pids", return_value=None
    ), patch.object(
        cli_main, "_ledger_manual_serve_holders", return_value=None
    ), patch.object(
        cli_main, "_handoff_reapable_backend_pids", return_value=None
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


def test_guard_reaps_dashboard_holders_and_proceeds():
    # The bug: holders = [chat backend] → exit 2, forever, from the page that
    # asked for the update. Now the rung stops them and the update continues.
    rows = _rows(_CHAT_BACKEND, _TOOL_KERNEL)
    result, killed = _run_guard([rows, []], rows)
    assert result == "past_guard"
    assert killed == [[700, 710]]


def test_guard_refuses_when_a_dashboard_holder_survives():
    # Reap runs but the holder is still there (unkillable child) → refuse.
    rows = _rows(_CHAT_BACKEND)
    result, killed = _run_guard([rows, rows], rows)
    assert result == "exit_2"
    assert killed == [[700]]


def test_guard_still_refuses_a_user_venv_session():
    # Unchanged behaviour: a plain venv session the user owns keeps the
    # refusal, and nothing is stopped on the way to it.
    rows = _rows(_USER_SESSION, _DASHBOARD_SERVER)
    result, killed = _run_guard([rows, rows], rows)
    assert result == "exit_2"
    assert killed == []


def test_guard_reaps_only_dashboard_holders_from_a_mixed_set():
    # A dashboard holder alongside a user session: only the dashboard one is
    # stopped, and the survivor still produces the refusal.
    rows = _rows(_CHAT_BACKEND, _USER_SESSION)
    result, killed = _run_guard([rows, rows[1:]], rows)
    assert result == "exit_2"
    assert killed == [[700]]
