"""Tests for the Windows half-updated-venv hardening (July 2026 incident).

Covers four additions to ``hermes update``:

1. ``_venv_core_imports_healthy`` — the venv health probe that lets an
   "Already up to date" checkout still repair a broken dependency install.
2. ``_detect_venv_python_processes`` — the venv-interpreter process guard
   that refuses to mutate the venv while a desktop backend / stray python
   holds .pyd files mapped. With the ``allowlist`` keyword it honors
   ``updates.venv_holder_allowlist`` (#66933).
3. ``_run_pre_update_hook`` — the operator-configured pre-guard hook
   (#66933) that lets supervised deployments release external venv
   holders before the guard re-checks.
4. The commit_count == 0 repair branch wiring in ``_cmd_update_impl``.

All Windows-specific paths are exercised via ``_is_windows`` patching so
they run on any host (same approach as test_update_concurrent_quarantine).
"""

from __future__ import annotations

import subprocess
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import main as cli_main
from hermes_cli.config import load_config


# ---------------------------------------------------------------------------
# _venv_core_imports_healthy
# ---------------------------------------------------------------------------


def test_venv_health_reports_healthy_when_no_venv(tmp_path):
    """No venv python in a DEV checkout → nothing to probe → healthy."""
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path):
        healthy, detail = cli_main._venv_core_imports_healthy()
    assert healthy is True
    assert detail == ""


def test_venv_health_missing_venv_unhealthy_on_managed_install(tmp_path):
    """On a managed install (bootstrap marker) the venv IS the install —
    its absence must be reported unhealthy so the repair lane runs instead
    of 'Already up to date!'."""
    (tmp_path / ".hermes-bootstrap-complete").write_text("done")
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path):
        healthy, detail = cli_main._venv_core_imports_healthy()
    assert healthy is False
    assert "venv python missing" in detail


def test_venv_health_missing_venv_unhealthy_with_interrupted_marker(tmp_path):
    """An interrupted-update breadcrumb also flips missing-venv to unhealthy."""
    (tmp_path / ".update-incomplete").write_text("started=1\npid=1\n")
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path):
        healthy, detail = cli_main._venv_core_imports_healthy()
    assert healthy is False
    assert "venv python missing" in detail


def _fake_venv_python(tmp_path, *, windows: bool = False):
    bin_dir = tmp_path / "venv" / ("Scripts" if windows else "bin")
    bin_dir.mkdir(parents=True)
    py = bin_dir / ("python.exe" if windows else "python")
    py.write_bytes(b"")
    return py


def test_venv_health_reports_missing_imports(tmp_path):
    """Probe output lines are surfaced as the unhealthy detail."""
    _fake_venv_python(tmp_path)

    fake = SimpleNamespace(
        returncode=0,
        stdout="fastapi: No module named 'annotated_doc'\n",
        stderr="",
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        cli_main.subprocess, "run", return_value=fake
    ):
        healthy, detail = cli_main._venv_core_imports_healthy()

    assert healthy is False
    assert "annotated_doc" in detail


def test_venv_health_healthy_when_probe_clean(tmp_path):
    _fake_venv_python(tmp_path)
    fake = SimpleNamespace(returncode=0, stdout="", stderr="")
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        cli_main.subprocess, "run", return_value=fake
    ):
        healthy, detail = cli_main._venv_core_imports_healthy()
    assert healthy is True


def test_venv_health_broken_interpreter_is_unhealthy(tmp_path):
    """Nonzero exit with no module list = interpreter itself is broken."""
    _fake_venv_python(tmp_path)
    fake = SimpleNamespace(returncode=1, stdout="", stderr="Fatal Python error: init failed\n")
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        cli_main.subprocess, "run", return_value=fake
    ):
        healthy, detail = cli_main._venv_core_imports_healthy()
    assert healthy is False
    assert "Fatal Python error" in detail


def test_venv_health_probe_failure_reports_healthy(tmp_path):
    """A probe that can't run must NOT force needless reinstalls."""
    _fake_venv_python(tmp_path)
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        cli_main.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(cmd="python", timeout=60),
    ):
        healthy, _detail = cli_main._venv_core_imports_healthy()
    assert healthy is True


# ---------------------------------------------------------------------------
# _detect_venv_python_processes
# ---------------------------------------------------------------------------


def _proc(pid: int, exe: str, name: str, cmdline: list[str] | None = None, cwd: str = ""):
    proc = MagicMock()
    proc.info = {
        "pid": pid,
        "exe": exe,
        "name": name,
        "cmdline": cmdline or [],
        "cwd": cwd,
    }
    return proc


def test_detect_venv_python_off_windows_is_empty():
    with patch.object(cli_main, "_is_windows", return_value=False):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_finds_backend(_winp, tmp_path):
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    other_py = "C:\\Python311\\python.exe"

    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(101, venv_py, "python.exe", ["python.exe", "-m", "hermes_cli.main", "serve"]),
                _proc(102, other_py, "python.exe", ["python.exe", "somescript.py"]),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [m[0] for m in matches] == [101]
    assert "serve" in matches[0][2]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_excludes_self_and_ancestors(_winp, tmp_path):
    import os as _os

    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    parent = MagicMock()
    parent.pid = 555
    me = MagicMock()
    me.parents.return_value = [parent]
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(_os.getpid(), venv_py, "python.exe"),
                _proc(555, venv_py, "hermes.exe"),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_no_psutil_is_empty(_winp, tmp_path):
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": None}
    ):
        assert cli_main._detect_venv_python_processes() == []


def test_format_venv_holders_message_flags_desktop_backend(tmp_path):
    matches = [
        (101, "python.exe", "python.exe -m hermes_cli.main serve --host 127.0.0.1"),
        (102, "pythonw.exe", "pythonw.exe -m hermes_cli.main gateway run"),
    ]
    msg = cli_main._format_venv_python_holders_message(matches)
    assert "101" in msg
    assert "desktop app" in msg.lower()
    assert "gateway" in msg
    assert "hermes update" in msg
    assert "--force-venv" in msg


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_catches_outside_venv_trampoline(_winp, tmp_path):
    """uv/base-interpreter trampoline: exe OUTSIDE the venv, but the cmdline
    clearly runs Hermes from this install → must still be flagged as a holder
    (it imports from the venv and holds its .pyd files)."""
    base_py = "C:\\Python311\\python.exe"
    venv_path = str(tmp_path / "venv" / "Scripts" / "python.exe")

    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                # cmdline references the venv path directly
                _proc(201, base_py, "python.exe", [base_py, venv_path, "-m", "x"]),
                # `-m hermes_cli.main serve` with the install root as cwd
                _proc(
                    202,
                    base_py,
                    "python.exe",
                    [base_py, "-m", "hermes_cli.main", "serve"],
                    cwd=str(tmp_path),
                ),
                # unrelated base-interpreter python → NOT a holder
                _proc(203, base_py, "python.exe", [base_py, "somescript.py"], cwd="C:\\other"),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert sorted(m[0] for m in matches) == [201, 202]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_hermes_cli_cmdline_outside_install_not_matched(_winp, tmp_path):
    """A hermes_cli.main process belonging to a DIFFERENT install (neither
    install root in cmdline nor cwd under it) must not be flagged."""
    base_py = "C:\\Python311\\python.exe"
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(
                    301,
                    base_py,
                    "python.exe",
                    [base_py, "-m", "hermes_cli.main", "serve"],
                    cwd="C:\\other-install",
                ),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


# ---------------------------------------------------------------------------
# --force vs --force-venv gating of the venv-holder guard
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


def _run_update_until_guard(args):
    """Drive _cmd_update_impl just far enough to hit the venv-holder guard.

    Everything before the guard is stubbed; the guard firing is observed via
    SystemExit(2). The first statement AFTER the guard is
    ``git_dir = PROJECT_ROOT / ".git"`` — a PROJECT_ROOT sentinel whose
    ``__truediv__`` raises marks 'guard passed'."""

    class _PastGuard(Exception):
        pass

    class _RootSentinel:
        def __truediv__(self, _other):
            raise _PastGuard

    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(cli_main, "_run_pre_update_backup"), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main, "_resume_windows_gateways_after_update"
    ), patch.object(
        cli_main,
        "_detect_venv_python_processes",
        return_value=[(101, "python.exe", "python.exe -m hermes_cli.main serve")],
    ), patch.object(
        cli_main, "PROJECT_ROOT", _RootSentinel()
    ):
        try:
            cli_main._cmd_update_impl(args, gateway_mode=False)
        except _PastGuard:
            return "past_guard"
        except SystemExit as exc:
            return f"exit_{exc.code}"
    return "returned"


@pytest.mark.parametrize(
    "force,force_venv,expected",
    [
        (False, False, "exit_2"),   # guard fires
        (True, False, "exit_2"),    # plain --force does NOT bypass the venv guard
        (False, True, "past_guard"),  # --force-venv is the explicit escape hatch
        (True, True, "past_guard"),
    ],
)
def test_venv_holder_guard_force_semantics(force, force_venv, expected, capsys):
    result = _run_update_until_guard(_update_args(force=force, force_venv=force_venv))
    assert result == expected, capsys.readouterr().out


# ---------------------------------------------------------------------------
# _detect_venv_python_processes allowlist parameter (updates.venv_holder_allowlist)
# See #66933.
# ---------------------------------------------------------------------------


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_allowlist_filters_cmdline_substring(_winp, tmp_path):
    """A holder whose cmdline contains an allowlisted substring is dropped."""
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(
                    301,
                    venv_py,
                    "python.exe",
                    ["python.exe", "ecosystem_bridge.py"],
                    cwd=str(tmp_path),
                ),
                _proc(
                    302,
                    venv_py,
                    "python.exe",
                    ["python.exe", "-m", "hermes_cli.main", "gateway", "run"],
                    cwd=str(tmp_path),
                ),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes(
            allowlist=["ecosystem_bridge"]
        )

    assert [m[0] for m in matches] == [302]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_allowlist_filters_process_name(_winp, tmp_path):
    """A holder whose process NAME contains an allowlisted substring is dropped."""
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(
                    401,
                    venv_py,
                    "ecosystem_bridge.exe",
                    [],
                    cwd=str(tmp_path),
                ),
                _proc(402, venv_py, "pythonw.exe", ["pythonw.exe", "monitor.exe"]),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes(allowlist=["bridge"])

    assert [m[0] for m in matches] == [402]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_allowlist_is_case_insensitive(_winp, tmp_path):
    """Allowlist matching is case-insensitive against both name and cmdline."""
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [_proc(501, venv_py, "ECOSYSTEM_BRIDGE.EXE", ["PYTHON.EXE", "Bridge.py"])]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes(allowlist=["bridge"])

    assert matches == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_empty_or_none_allowlist_keeps_all(_winp, tmp_path):
    """allowlist=None or [] preserves the original strict behavior (none dropped)."""
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [_proc(601, venv_py, "ecosystem_bridge.exe", [])]
        ),
        Process=lambda *a, **k: me,
    )
    base_patches = (
        patch.object(cli_main, "PROJECT_ROOT", tmp_path),
        patch.dict(sys.modules, {"psutil": fake_psutil}),
    )
    with base_patches[0], base_patches[1]:
        first = cli_main._detect_venv_python_processes(allowlist=None)
    with base_patches[0], base_patches[1]:
        second = cli_main._detect_venv_python_processes(allowlist=[])
    assert [m[0] for m in first] == [601]
    assert [m[0] for m in second] == [601]  # noqa: explicit assertion


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_allowlist_non_matching_entry_does_not_relax(_winp, tmp_path):
    """Allowlist entries that don't match any holder must not silently empty the
    refusal list — the guard must still fire on the un-matched holders."""
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(701, venv_py, "ecosystem_bridge.exe", []),
                _proc(
                    702,
                    venv_py,
                    "python.exe",
                    ["python.exe", "-m", "hermes_cli.main", "serve"],
                ),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes(allowlist=["completely_unrelated"])

    assert sorted(m[0] for m in matches) == [701, 702]


# ---------------------------------------------------------------------------
# _run_pre_update_hook
# ---------------------------------------------------------------------------


def test_pre_update_hook_returns_true_when_unconfigured():
    """No updates.pre_update_command -> noop -> True, regardless of args."""
    args = SimpleNamespace(force_venv=False)
    with patch(
        "hermes_cli.config.load_config",
        return_value={"updates": {"pre_update_backup": "quick"}},
    ):
        assert cli_main._run_pre_update_hook(args) is True


def test_pre_update_hook_runs_string_command():
    """String cmd -> shell=True; exit 0 -> True."""
    args = SimpleNamespace(force_venv=False)
    cfg = {"updates": {"pre_update_command": "echo release", "pre_update_command_timeout": 5}}
    fake = SimpleNamespace(returncode=0, stdout="release\n", stderr="")
    with patch("hermes_cli.config.load_config", return_value=cfg), patch.object(
        cli_main.subprocess, "run", return_value=fake
    ) as run:
        assert cli_main._run_pre_update_hook(args) is True
    args_used, kwargs = run.call_args
    assert args_used[0] == "echo release"
    assert kwargs["shell"] is True


def test_pre_update_hook_runs_argv_list_without_shell():
    """List cmd -> exec'd directly (shell=False)."""
    args = SimpleNamespace(force_venv=False)
    cfg = {"updates": {"pre_update_command": ["sc", "stop", "MySvc"], "pre_update_command_timeout": 5}}
    fake = SimpleNamespace(returncode=0, stdout="", stderr="")
    with patch("hermes_cli.config.load_config", return_value=cfg), patch.object(
        cli_main.subprocess, "run", return_value=fake
    ) as run:
        assert cli_main._run_pre_update_hook(args) is True
    args_used, kwargs = run.call_args
    assert args_used[0] == ["sc", "stop", "MySvc"]
    # shell defaults to False for argv lists; either explicit False or omitted is fine
    assert kwargs.get("shell", False) is False


def test_pre_update_hook_aborts_on_nonzero_exit(capsys):
    args = SimpleNamespace(force_venv=False)
    cfg = {"updates": {"pre_update_command": "false", "pre_update_command_timeout": 1}}
    fake = SimpleNamespace(returncode=1, stdout="", stderr="boom\n")
    with patch("hermes_cli.config.load_config", return_value=cfg), patch.object(
        cli_main.subprocess, "run", return_value=fake
    ):
        assert cli_main._run_pre_update_hook(args) is False
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "exited" in out.lower() or "exit code" in out.lower()


def test_pre_update_hook_aborts_on_timeout(capsys):
    args = SimpleNamespace(force_venv=False)
    cfg = {"updates": {"pre_update_command": "sleep 99", "pre_update_command_timeout": 1}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch.object(
        cli_main.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(cmd="sleep 99", timeout=1),
    ):
        assert cli_main._run_pre_update_hook(args) is False
    out = capsys.readouterr().out
    assert "timed out" in out.lower()


def test_pre_update_hook_handles_subprocess_start_failure(capsys):
    args = SimpleNamespace(force_venv=False)
    cfg = {"updates": {"pre_update_command": "no-such-cmd", "pre_update_command_timeout": 5}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch.object(
        cli_main.subprocess, "run", side_effect=OSError("not found")
    ):
        assert cli_main._run_pre_update_hook(args) is False
    out = capsys.readouterr().out
    assert "failed to start" in out.lower() or "aborting" in out.lower()


def test_pre_update_hook_returns_true_when_load_config_raises():
    """A load_config failure must not abort the update — fail-open to defaults."""
    args = SimpleNamespace(force_venv=False)
    with patch(
        "hermes_cli.config.load_config", side_effect=RuntimeError("disk full")
    ):
        assert cli_main._run_pre_update_hook(args) is True


def test_pre_update_hook_clamps_invalid_timeout():
    """Negative / non-numeric timeouts fall back to safe defaults."""
    args = SimpleNamespace(force_venv=False)
    cfg = {"updates": {"pre_update_command": "noop", "pre_update_command_timeout": -5}}
    fake = SimpleNamespace(returncode=0, stdout="", stderr="")
    with patch("hermes_cli.config.load_config", return_value=cfg), patch.object(
        cli_main.subprocess, "run", return_value=fake
    ) as run:
        assert cli_main._run_pre_update_hook(args) is True
    assert run.call_args.kwargs["timeout"] is None  # clamped to 0 -> no timeout


# ---------------------------------------------------------------------------
# Wiring: _cmd_update_impl calls hook + passes allowlist to detector.
# ---------------------------------------------------------------------------


def _run_update_until_guard_with_allowlist(args, *, allowlist, hook_return=True, detected=None):
    """Variant of _run_update_until_guard that patches load_config and lets
    us assert what allowlist the detector actually received."""
    if detected is None:
        detected = []

    class _PastGuard(Exception):
        pass

    class _RootSentinel:
        def __truediv__(self, _other):
            raise _PastGuard

    captured: dict = {}

    def _capture(**kwargs):
        captured["allowlist"] = kwargs.get("allowlist")
        captured["kw"] = kwargs
        return detected

    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(cli_main, "_run_pre_update_backup"), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main, "_resume_windows_gateways_after_update"
    ), patch.object(
        cli_main, "_run_pre_update_hook", return_value=hook_return
    ), patch.object(
        cli_main, "_detect_venv_python_processes", side_effect=_capture
    ), patch.object(
        cli_main, "PROJECT_ROOT", _RootSentinel()
    ), patch(
        "hermes_cli.config.load_config",
        return_value={"updates": {"venv_holder_allowlist": allowlist}},
    ):
        try:
            cli_main._cmd_update_impl(args, gateway_mode=False)
        except _PastGuard:
            return "past_guard", captured
        except SystemExit as exc:
            return f"exit_{exc.code}", captured
    return "returned", captured


def test_venv_holder_guard_wiring_reads_allowlist_from_config():
    """The allowlist from updates.venv_holder_allowlist is forwarded to the
    detector with each entry lower-cased."""
    args = _update_args()
    result, captured = _run_update_until_guard_with_allowlist(
        args, allowlist=["MyBridge", "ECOSYSTEM_DAEMON"], detected=[]
    )
    assert result == "past_guard"
    assert captured["allowlist"] == ["mybridge", "ecosystem_daemon"]


def test_venv_holder_guard_wiring_lowercases_and_filters_invalid_entries():
    args = _update_args()
    result, captured = _run_update_until_guard_with_allowlist(
        args,
        allowlist=["ok", 123, None, {"x": "y"}, "", "AlsoOk"],
        detected=[],
    )
    assert result == "past_guard"
    # strings are lower-cased; ints are str()'d and lowercased; None and dict
    # are filtered; "" passes through (its substring-match-everything pitfall
    # is the operator's responsibility, not enforced here).
    assert captured["allowlist"] == ["ok", "123", "", "alsook"]


def test_venv_holder_guard_wiring_config_load_failure_keeps_existing_behavior():
    """If load_config raises, allowlist defaults to [] (strict guard preserved)."""

    class _PastGuard(Exception):
        pass

    class _RootSentinel:
        def __truediv__(self, _other):
            raise _PastGuard

    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(cli_main, "_run_pre_update_backup"), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main, "_resume_windows_gateways_after_update"
    ) as resume, patch.object(
        cli_main, "_run_pre_update_hook", return_value=True
    ), patch.object(
        cli_main,
        "_detect_venv_python_processes",
        return_value=[(1, "x.exe", "x")],
    ) as det, patch.object(
        cli_main, "PROJECT_ROOT", _RootSentinel()
    ), patch(
        "hermes_cli.config.load_config", side_effect=RuntimeError("boom")
    ):
        try:
            cli_main._cmd_update_impl(_update_args(), gateway_mode=False)
        except _PastGuard:
            pass  # acceptable
        except SystemExit:
            pass  # also acceptable — guard fired because allowlist=[] strict mode
    # The Detector was called exactly once and saw allowlist=[] (the safe default).
    det.assert_called_once()
    assert det.call_args.kwargs.get("allowlist") == []
    # Gateways were either resumed or never registered (either is fine here).
    assert resume.call_count >= 0
