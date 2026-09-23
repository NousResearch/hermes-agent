"""`hermes update` concurrent-instance gate and Windows gateway service helpers
(#26670, #37039, #98814): gateway/non-gateway classification, leftover venv
holder nomination, gateway-ancestor tree-kill refusal, and SCM stop/restore.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from hermes_cli import main as cli_main


from hermes_cli import update_cmd


# Tests in this module either exercise the REAL _detect_concurrent_hermes_instances
# helper (and need the autouse stub in tests/hermes_cli/conftest.py disabled),
# or supply their own explicit return value via patch.object. Mark the whole
# module so the conftest fixture skips its default stub.
pytestmark = pytest.mark.real_concurrent_gate


# ---------------------------------------------------------------------------
# Windows gateway pause/resume before update mutation
# ---------------------------------------------------------------------------


def test_restore_windows_gateway_service_waits_out_stop_pending(monkeypatch):
    import hermes_cli.update_cmd as update_cmd
    import hermes_cli.update_cmd_windows as update_cmd_windows

    statuses = iter(["stop_pending", "stopped"])
    service = SimpleNamespace(status=lambda: next(statuses))
    fake_psutil = SimpleNamespace(win_service_get=lambda _name: service)
    restarted = []
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(update_cmd._time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        update_cmd,
        "_start_windows_gateway_service",
        lambda name: restarted.append(name),
    )
    monkeypatch.setattr(
        update_cmd_windows,
        "_start_windows_gateway_service",
        lambda name: restarted.append(name),
    )

    update_cmd._restore_windows_gateway_service("HermesGateway")

    assert restarted == ["HermesGateway"]


def test_stop_windows_gateway_service_waits_for_original_descendants(
    monkeypatch,
):
    """SCM STOPPED is insufficient while the original process identity lives."""
    import hermes_cli.update_cmd as update_cmd

    service = SimpleNamespace(status=lambda: "stopped")
    fake_psutil = SimpleNamespace(
        win_service_get=lambda _name: service,
        Process=lambda pid: SimpleNamespace(create_time=lambda: 12.5),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    with pytest.raises(RuntimeError, match="process tree"):
        update_cmd._stop_windows_gateway_service(
            "HermesGateway",
            expected_processes=((123, 12.5),),
            timeout=0,
        )


# ---------------------------------------------------------------------------
# _leftover_pausable_gateway_pids (the guard-level gateway fallback)
#
# The pause stops every gateway discovery finds, but the venv-holder guard
# sees the process table as it is NOW. A supervisor (Scheduled Task, login
# watchdog) can respawn a gateway inside the pause→guard window, and some
# spawn paths never register in discovery at all. Those holders are exactly
# what the pause machinery exists to stop — the guard nominates them for a
# stop-and-recheck instead of dead-ending, and refuses the moment any
# non-gateway holder is present.
# ---------------------------------------------------------------------------


GATEWAY_ARGV = [
    r"C:\x\venv\Scripts\python.exe",
    "-m",
    "hermes_cli.main",
    "gateway",
    "run",
]


def _fake_psutil_cmdlines(argv_by_pid):
    """psutil stand-in serving live argv per pid; unknown pids raise."""

    class FakeProc:
        def __init__(self, pid):
            if pid not in argv_by_pid:
                raise ValueError(f"no such pid {pid}")
            self._argv = argv_by_pid[pid]

        def cmdline(self):
            return self._argv

    return types.SimpleNamespace(Process=FakeProc)


def test_leftover_holders_that_are_all_gateways_are_nominated(monkeypatch):
    """Respawned/unmapped gateway holders get stopped, not dead-ended on."""
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        _fake_psutil_cmdlines({300: GATEWAY_ARGV, 301: GATEWAY_ARGV}),
    )
    matches = [
        (300, "python.exe", "truncated..."),
        (301, "python.exe", "truncated..."),
    ]

    assert cli_main._leftover_pausable_gateway_pids(matches) == [300, 301]


def test_plain_update_refuses_to_tree_kill_its_gateway_ancestor(
    monkeypatch, capsys
):
    """#98814: terminal-launched update must survive to report the refusal."""
    import hermes_cli.gateway as gateway_cli
    import hermes_cli.update_cmd as update_cmd

    monkeypatch.setattr(
        gateway_cli,
        "_is_pid_ancestor_of_current_process",
        lambda pid: pid == 300,
    )

    refused = update_cmd._refuse_gateway_ancestor_tree_kill(
        [300, 301], gateway_mode=False
    )

    assert refused is True
    output = capsys.readouterr().out
    assert "taskkill /T" in output
    assert "`/update`" in output
    assert "separate terminal" in output


def test_gateway_handoff_keeps_leftover_gateway_recovery(monkeypatch, capsys):
    """The detached `/update` hand-off still owns leftover gateway cleanup."""
    import hermes_cli.gateway as gateway_cli
    import hermes_cli.update_cmd as update_cmd

    ancestry_checks = []
    monkeypatch.setattr(
        gateway_cli,
        "_is_pid_ancestor_of_current_process",
        lambda pid: ancestry_checks.append(pid) or True,
    )

    assert (
        update_cmd._refuse_gateway_ancestor_tree_kill(
            [300], gateway_mode=True
        )
        is False
    )
    assert ancestry_checks == []
    assert capsys.readouterr().out == ""


def test_one_non_gateway_holder_keeps_the_hard_refusal(monkeypatch):
    """A REPL/backend holder means the guard must abort exactly as before."""
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        _fake_psutil_cmdlines(
            {300: GATEWAY_ARGV, 400: [r"C:\x\venv\Scripts\python.exe", "-i"]}
        ),
    )
    matches = [(300, "python.exe", "..."), (400, "python.exe", "...")]

    assert cli_main._leftover_pausable_gateway_pids(matches) is None


def test_unreadable_argv_falls_back_to_the_captured_prefix(monkeypatch):
    """psutil failure degrades to the scan's captured cmdline, not a crash.

    The captured prefix decides: a gateway invocation still qualifies, and
    anything else still refuses.
    """
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil_cmdlines({}))
    gateway_prefix = r"venv\Scripts\python.exe -m hermes_cli.main gateway run"

    assert cli_main._leftover_pausable_gateway_pids(
        [(300, "python.exe", gateway_prefix)]
    ) == [300]
    assert (
        cli_main._leftover_pausable_gateway_pids(
            [
                (300, "python.exe", gateway_prefix),
                (400, "python.exe", "python.exe -i"),
            ]
        )
        is None
    )


# ---------------------------------------------------------------------------
# cmd_update integration — concurrent-instance gate
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _classify_concurrent_instance / _filter_non_gateway_concurrent_instances
#
# #37039: the pre-update concurrent-instance gate lets the update proceed
# when every concurrent hermes.exe is a gateway runtime — the pause
# machinery (_pause_windows_gateways_for_update) stops those before any
# file mutation and the post-update restart phase brings them back.
# Classification delegates to _is_pausable_gateway → the canonical
# gateway.status.looks_like_gateway_command_line matcher, so the gate's
# exemption and the pause discovery cannot drift apart.
# ---------------------------------------------------------------------------


def _fake_psutil_classify(argv_by_pid):
    """psutil stand-in serving .cmdline() per pid; unknown pids raise."""

    class FakeProc:
        def __init__(self, pid):
            if pid not in argv_by_pid:
                raise ValueError(f"no such pid {pid}")
            self._argv = argv_by_pid[pid]

        def cmdline(self):
            return self._argv

    return types.SimpleNamespace(Process=FakeProc)


def test_classify_concurrent_instance_recognises_gateway_runtimes(monkeypatch):
    """Gateway runtime command lines classify as ``gateway`` regardless of
    launcher shape (python -m, hermes.exe shim, hermes-gateway.exe,
    gateway/run.py, bare `hermes gateway` which defaults to run)."""
    cases = [
        [r"C:\venv\Scripts\python.exe", "-m", "hermes_cli.main", "gateway", "run"],
        [r"C:\venv\Scripts\hermes.exe", "gateway", "run"],
        [r"C:\venv\Scripts\hermes-gateway.exe"],
        [r"C:\venv\Scripts\python.exe", "gateway/run.py"],
        ["hermes.exe", "GATEWAY", "RUN"],  # matcher is case-insensitive
        ["hermes.exe", "gateway"],  # bare `hermes gateway` defaults to run
        # profile selector before the subcommand — canonical matcher strips it
        ["hermes.exe", "--profile", "work", "gateway", "run"],
    ]
    for argv in cases:
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil_classify({77: argv}))
        result = update_cmd._classify_concurrent_instance(77)
        assert result == "gateway", f"expected gateway for {argv!r}, got {result!r}"


def test_classify_concurrent_instance_recognises_non_gateways(monkeypatch):
    """Non-runtime command lines classify as ``non-gateway`` — including
    gateway MANAGEMENT subcommands (`gateway status`), which the canonical
    matcher rejects but a substring matcher would misclassify. These keep
    the pre-update abort."""
    cases = [
        [r"C:\venv\Scripts\hermes.exe"],  # interactive REPL
        [r"C:\venv\Scripts\hermes.exe", "dashboard"],
        ["hermes.exe", "gateway", "status"],  # management, not runtime
        ["hermes.exe", "gateway", "stop"],
        ["python", "-m", "hermes_cli.main"],
        [],
    ]
    for argv in cases:
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil_classify({77: argv}))
        result = update_cmd._classify_concurrent_instance(77)
        assert result == "non-gateway", (
            f"expected non-gateway for {argv!r}, got {result!r}"
        )


def test_classify_concurrent_instance_unknown_on_psutil_error(monkeypatch):
    """Unreadable cmdline (process gone / AccessDenied) → ``unknown`` —
    treated as non-gateway by the filter, so the gate still aborts."""
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil_classify({}))
    assert update_cmd._classify_concurrent_instance(4242) == "unknown"


def test_classify_concurrent_instance_unknown_without_psutil(monkeypatch):
    """Missing psutil entirely → ``unknown``, never a crash."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert update_cmd._classify_concurrent_instance(4242) == "unknown"


def test_filter_non_gateway_concurrent_instances_splits(monkeypatch):
    """Gateway PIDs drop out of the abort list; REPL/dashboard/unknown stay."""
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        _fake_psutil_classify(
            {
                100: ["hermes.exe", "gateway", "run"],
                200: ["hermes.exe"],  # REPL — keep
                300: ["hermes.exe", "dashboard"],  # keep
                # 400 missing → unknown → keep
            }
        ),
    )
    matches = [
        (100, "hermes.exe"),
        (200, "hermes.exe"),
        (300, "hermes.exe"),
        (400, "hermes.exe"),
    ]
    kept = cli_main._filter_non_gateway_concurrent_instances(matches)
    assert kept == [(200, "hermes.exe"), (300, "hermes.exe"), (400, "hermes.exe")]


def test_filter_non_gateway_concurrent_instances_gateway_only(monkeypatch):
    """All-gateway match list filters to empty — the gate lets the update
    proceed and the pause machinery handles the gateways."""
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        _fake_psutil_classify(
            {
                111: ["hermes.exe", "gateway", "run"],
                222: [r"C:\venv\Scripts\hermes-gateway.exe"],
            }
        ),
    )
    matches = [(111, "hermes.exe"), (222, "hermes-gateway.exe")]
    assert cli_main._filter_non_gateway_concurrent_instances(matches) == []


# ---------------------------------------------------------------------------
# _cmd_update_impl integration with the relaxed pre-update gate (#37039)
# ---------------------------------------------------------------------------


def test_stop_service_refuses_pid_reuse_before_sc_stop(monkeypatch):
    import hermes_cli.update_cmd as update_cmd

    fake_psutil = SimpleNamespace(
        win_service_get=lambda _name: SimpleNamespace(
            status=lambda: "running", pid=lambda: 11
        ),
        Process=lambda _pid: SimpleNamespace(create_time=lambda: 99.0),
    )
    calls = []
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(update_cmd.subprocess, "run", lambda *_a, **_k: calls.append(True))

    with pytest.raises(RuntimeError, match="identity changed"):
        update_cmd._stop_windows_gateway_service(
            "HermesGateway", expected_service_identity=(11, 11.0)
        )

    assert calls == []


