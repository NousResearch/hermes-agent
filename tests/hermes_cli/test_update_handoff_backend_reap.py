"""Tests for the GUI-updater hand-off backend reap (_handoff_reapable_backend_pids).

Field incident (2026-08-20, Teknium's Windows box): a Desktop update hand-off
(`hermes update --yes --gateway --force`) left a *swarm* of per-profile `serve`
backends (mr-tester, probe-inherit, turqoise, clippy, maroon, …) holding
`cryptography\\_rust.pyd`. Some still had a live parent (the tearing-down
Electron process, or the venv launcher→worker two-hop chain mid-exit), so the
strict orphan-only reap (_orphaned_desktop_backend_pids) disqualified the whole
set and the update dead-ended — a 12-minute hang, then a force-close that
stranded bot sessions.

_handoff_reapable_backend_pids is the additional rung that ONLY runs in the
hand-off context (caller gates on args.gateway + the update-incomplete marker +
no live hermes.exe shim). There, any surviving Hermes `serve`/`dashboard`
backend from this venv is a leak — live parent or not — and safe to reap.
A non-backend holder still disqualifies the whole set.

Runs on any host via a fake psutil module (same approach as
test_update_orphan_backend_reap.py).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

from hermes_cli import main as cli_main
from hermes_cli import update_cmd


class _FakeNoSuchProcess(Exception):
    pass


def _fake_psutil(procs: dict[int, MagicMock]):
    def _process(pid: int):
        if pid not in procs:
            raise _FakeNoSuchProcess(pid)
        return procs[pid]

    return types.SimpleNamespace(Process=_process, NoSuchProcess=_FakeNoSuchProcess)


def _proc(pid: int, cmdline: list[str]):
    proc = MagicMock()
    proc.pid = pid
    proc.cmdline.return_value = cmdline
    return proc


def _serve_argv(profile: str = "mr-tester") -> list[str]:
    return [
        "C:\\hermes\\venv\\Scripts\\python.exe",
        "-m",
        "hermes_cli.main",
        "--profile",
        profile,
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ]


def _holder(pid: int, cmdline: str):
    return (pid, "python.exe", cmdline)


def test_live_parent_backend_reaped_in_handoff():
    # The exact case the orphan-only path REFUSES: a serve backend that still
    # has a live parent. In the hand-off context it must still be reaped.
    backend = _proc(200, _serve_argv("mr-tester"))
    fake = _fake_psutil({200: backend})
    with patch.dict(sys.modules, {"psutil": fake}):
        holders = [_holder(200, "python.exe -m hermes_cli.main --profile mr-tester serve")]
        assert cli_main._handoff_reapable_backend_pids(holders) == [200]


def test_swarm_of_profile_backends_all_reaped():
    profiles = ["mr-tester", "probe-inherit", "turqoise", "clippy", "maroon"]
    procs = {200 + i: _proc(200 + i, _serve_argv(p)) for i, p in enumerate(profiles)}
    fake = _fake_psutil(procs)
    with patch.dict(sys.modules, {"psutil": fake}):
        holders = [
            _holder(200 + i, f"python.exe -m hermes_cli.main --profile {p} serve")
            for i, p in enumerate(profiles)
        ]
        assert sorted(cli_main._handoff_reapable_backend_pids(holders)) == sorted(procs)


def test_dashboard_backend_reaped():
    backend = _proc(200, ["python.exe", "-m", "hermes_cli.main", "dashboard"])
    fake = _fake_psutil({200: backend})
    with patch.dict(sys.modules, {"psutil": fake}):
        holders = [_holder(200, "python.exe -m hermes_cli.main dashboard")]
        assert cli_main._handoff_reapable_backend_pids(holders) == [200]


def test_non_backend_holder_disqualifies_whole_set():
    # An operator REPL / stray script during a hand-off is unexpected — refuse
    # the whole set rather than reap something we can't justify.
    backend = _proc(200, _serve_argv("mr-tester"))
    repl = _proc(300, ["python.exe", "-m", "hermes_cli.main", "chat"])
    fake = _fake_psutil({200: backend, 300: repl})
    with patch.dict(sys.modules, {"psutil": fake}):
        holders = [
            _holder(200, "python.exe -m hermes_cli.main --profile mr-tester serve"),
            _holder(300, "python.exe -m hermes_cli.main chat"),
        ]
        assert cli_main._handoff_reapable_backend_pids(holders) is None


def test_exited_holder_skipped_not_fatal():
    # A holder that vanished between scan and classification is skipped, and the
    # remaining real backend still qualifies.
    backend = _proc(200, _serve_argv("mr-tester"))
    fake = _fake_psutil({200: backend})  # 300 absent → NoSuchProcess
    with patch.dict(sys.modules, {"psutil": fake}):
        holders = [
            _holder(300, "python.exe -m hermes_cli.main --profile gone serve"),
            _holder(200, "python.exe -m hermes_cli.main --profile mr-tester serve"),
        ]
        assert cli_main._handoff_reapable_backend_pids(holders) == [200]


def test_no_holders_returns_none():
    fake = _fake_psutil({})
    with patch.dict(sys.modules, {"psutil": fake}):
        assert cli_main._handoff_reapable_backend_pids([]) is None


def test_psutil_unavailable_returns_none():
    # Can't re-read argv to classify → refuse (leave the decision to the
    # caller's existing rungs / the dead-end).
    import builtins

    real_import = builtins.__import__

    def _no_psutil(name, *a, **k):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *a, **k)

    with patch.dict(sys.modules, {}, clear=False):
        sys.modules.pop("psutil", None)
        with patch("builtins.__import__", _no_psutil):
            holders = [_holder(200, "python.exe -m hermes_cli.main --profile x serve")]
            assert cli_main._handoff_reapable_backend_pids(holders) is None
