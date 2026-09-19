"""Tests for the Windows half-updated-venv hardening (July 2026 incident).

Covers three additions to ``hermes update``:

1. ``_venv_core_imports_healthy`` — the venv health probe that lets an
   "Already up to date" checkout still repair a broken dependency install.
2. ``_detect_venv_python_processes`` — the venv-interpreter process guard
   that refuses to mutate the venv while a desktop backend / stray python
   holds .pyd files mapped.
3. The commit_count == 0 repair branch wiring in ``_cmd_update_impl``.

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
from hermes_cli import update_cmd


# ---------------------------------------------------------------------------
# _venv_core_imports_healthy
# ---------------------------------------------------------------------------




def _fake_venv_python(tmp_path, *, windows: bool = False):
    bin_dir = tmp_path / "venv" / ("Scripts" if windows else "bin")
    bin_dir.mkdir(parents=True)
    py = bin_dir / ("python.exe" if windows else "python")
    py.write_bytes(b"")
    return py




# ---------------------------------------------------------------------------
# _detect_venv_python_processes
# ---------------------------------------------------------------------------


def _proc(pid: int, exe: str, name: str, cmdline: list[str] | None = None, cwd: str = ""):
    proc = MagicMock()
    proc.info = {
        "pid": pid,
        "exe": exe,
        "name": name,
    }
    proc.cmdline.return_value = cmdline or []
    proc.cwd.return_value = cwd
    return proc




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
def test_detect_venv_python_prefetches_only_cheap_process_fields(_winp, tmp_path):
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    holder = _proc(101, venv_py, "python.exe", [venv_py, "-m", "hermes_cli.main", "serve"])
    unrelated = _proc(102, r"C:\Program Files\Browser\browser.exe", "browser.exe")
    unrelated.cmdline.side_effect = AssertionError("unrelated cmdline must stay lazy")
    unrelated.cwd.side_effect = AssertionError("unrelated cwd must stay lazy")
    attrs_seen = []
    me = MagicMock()
    me.parents.return_value = []

    def process_iter(attrs):
        attrs_seen.append(attrs)
        return iter([unrelated, holder])

    fake_psutil = types.SimpleNamespace(
        process_iter=process_iter,
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert attrs_seen == [["pid", "exe", "name"]]
    assert [match[0] for match in matches] == [101]
    holder.cmdline.assert_called_once_with()
    holder.cwd.assert_not_called()
    unrelated.cmdline.assert_not_called()
    unrelated.cwd.assert_not_called()


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_matches_uv_default_dotvenv(_winp, tmp_path):
    """#112958: the venv-prefix arm must see a uv-default ``.venv`` interpreter. A kernel-runner child has
    no ``hermes_cli.main`` in its cmdline, so only that arm can match it — the guard was blind to it."""
    venv_py = str(tmp_path / ".venv" / "Scripts" / "python.exe")
    holder = _proc(104, venv_py, "python.exe", [venv_py, str(tmp_path / "tools" / "hermes_kernel_runner.py")])
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([holder]),
        Process=lambda *a, **k: me,
    )
    (tmp_path / ".venv").mkdir()

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(sys.modules, {"psutil": fake_psutil}):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [104]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_keeps_external_interpreter_fallback(_winp, tmp_path):
    external = _proc(
        103,
        r"C:\Python311\python.exe",
        "python.exe",
        ["python.exe", "-m", "hermes_cli.main", "serve"],
        str(tmp_path),
    )
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([external]),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [103]
    external.cmdline.assert_called_once_with()
    external.cwd.assert_called_once_with()




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
        # Pin the orphan classifier: this test exercises --force/--force-venv
        # gating, not orphan detection (covered in
        # test_update_orphan_backend_reap.py). None = "not provably orphaned"
        # → the guard refuses exactly as before the orphan-reap addition.
        cli_main, "_orphaned_desktop_backend_pids", return_value=None
    ), patch.object(
        cli_main, "PROJECT_ROOT", _RootSentinel()
    ):
        try:
            update_cmd._cmd_update_impl(args, gateway_mode=False)
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
# "Already up to date" must not hide a venv the last pull never re-synced (#97208)
# ---------------------------------------------------------------------------


def test_venv_dependency_set_stale_compares_installed_distribution_with_checkout(tmp_path):
    """Reporter's state: git current at 0.21.2, venv metadata still 0.20.6 (the hand-off child
    refused the sync); core imports pass, so only the distribution version reveals the drift."""
    from hermes_cli import update_cmd_deps

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "hermes-agent"\nversion = "0.21.2"\n', encoding="utf-8")
    venv_python = _fake_venv_python(tmp_path)
    probes = []

    def fake_run(cmd, **kwargs):
        probes.append(cmd)
        return SimpleNamespace(returncode=0, stdout=f"{fake_run.installed}\n", stderr="")

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(cli_main, "_is_windows", return_value=False), \
            patch.object(update_cmd_deps.subprocess, "run", fake_run):
        fake_run.installed = "0.20.6"
        assert update_cmd._venv_dependency_set_stale() == (True, "installed hermes-agent 0.20.6, checkout is 0.21.2")
        fake_run.installed = "0.21.2"
        assert update_cmd._venv_dependency_set_stale() == (False, "")
    # Asked in the venv's own interpreter (the updater may run under another Python).
    assert {cmd[0] for cmd in probes} == {str(venv_python)}


def test_current_checkout_with_stale_dependency_set_runs_the_sync(monkeypatch, capsys):
    """Drift on the commit_count == 0 path triggers the same repair as an unhealthy venv instead
    of ``✓ Already up to date!``; a synced venv keeps the cheap node-only path."""
    from hermes_cli import main as hm

    calls = []
    monkeypatch.setattr(update_cmd, "_venv_core_imports_healthy", lambda: (True, ""))
    monkeypatch.setattr(hm, "_is_windows", lambda: False)
    monkeypatch.delenv(hm._UPDATE_REEXEC_ENV, raising=False)
    monkeypatch.setattr("hermes_cli.managed_uv.update_managed_uv", lambda **kwargs: None)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda **kwargs: "uv")
    monkeypatch.setattr(update_cmd, "_repair_venv_on_current_checkout",
                        lambda **kwargs: calls.append("sync") or True)
    monkeypatch.setattr(update_cmd, "_repair_node_deps_on_current_checkout",
                        lambda *a, **kwargs: calls.append(kwargs["completion_message"]) or True)

    def run(stale):
        monkeypatch.setattr(update_cmd, "_venv_dependency_set_stale", lambda: stale)
        return update_cmd._repair_current_checkout(
            assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
            had_desktop_app_before_update=False, active_lazy_features=[], active_tool_dependencies=[],
            upstream_checked=True, _windows_gateway_resume=None)

    assert run((True, "installed hermes-agent 0.20.6, checkout is 0.21.2")) is True
    assert calls == ["sync"]
    assert "never synced after the last pull" in capsys.readouterr().out
    assert run((False, "")) is True
    assert calls == ["sync", "✓ Already up to date!"]
