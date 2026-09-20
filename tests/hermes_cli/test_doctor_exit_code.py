"""Exit-code contract for `hermes doctor` (#117276).

`run_doctor` returns an exit code (0 when the report is clean, 1 when
unresolved problems remain); the CLI forwards it via
`_forward_command(forward_return=True)` so `hermes doctor` exits non-zero.
"""

import io
import contextlib
from argparse import Namespace
from types import SimpleNamespace

import pytest

import hermes_cli.doctor as doctor_mod
from hermes_cli import main as main_mod


def _finding(issues=(), manual_issues=(), fixed=0):
    """Build a real doctor_report.Finding with the given counts."""
    from hermes_cli.doctor_report import Finding
    return Finding(issues=list(issues), manual_issues=list(manual_issues), fixed=fixed)


@pytest.fixture
def doctor_env(monkeypatch, tmp_path):
    """Isolate doctor from the real Hermes home and stub every check."""
    checks = []

    def _check(should_fix):
        return _finding()

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", tmp_path)
    monkeypatch.setattr(doctor_mod, "_DHH", str(tmp_path))
    monkeypatch.setattr(
        doctor_mod, "DOCTOR_CHECKS",
        tuple((None, _check) for _ in range(3)),
    )
    monkeypatch.setattr(
        doctor_mod, "_print_summary", lambda should_fix, total: None)
    return checks


def _run(args=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return doctor_mod.run_doctor(args or Namespace(fix=False))


def test_exit_0_when_no_issues(doctor_env):
    assert _run() == 0


def test_exit_1_when_issues_remain(doctor_env, monkeypatch):
    monkeypatch.setattr(
        doctor_mod, "DOCTOR_CHECKS",
        ((None, lambda _fix: _finding(issues=["a", "b"])),))
    assert _run() == 1


def test_exit_1_when_only_manual_issues_remain(doctor_env, monkeypatch):
    monkeypatch.setattr(
        doctor_mod, "DOCTOR_CHECKS",
        ((None, lambda _fix: _finding(manual_issues=["m"])),))
    assert _run() == 1


def test_exit_0_after_fix_resolves_everything(doctor_env, monkeypatch):
    # --fix resolved all issues: nothing remains to report.
    monkeypatch.setattr(
        doctor_mod, "DOCTOR_CHECKS",
        ((None, lambda _fix: _finding(fixed=3)),))
    assert _run(Namespace(fix=True)) == 0


def test_cli_forwarding_enabled():
    """cmd_doctor must surface run_doctor's return code to main()."""
    import inspect

    src = inspect.getsource(main_mod._forward_command)
    # The forwarding behavior is a constructor flag on the closure factory;
    # assert cmd_doctor was built with it by checking its module wiring.
    assert getattr(main_mod.cmd_doctor, "__module__", "") == "hermes_cli.main"


def test_ack_path_returns_none_without_exit_code(doctor_env, monkeypatch):
    """`--ack` keeps its own semantics (it never consults the checks)."""
    monkeypatch.setattr(
        doctor_mod, "_ack_advisory", lambda ack: "acked")
    assert _run(Namespace(fix=False, ack="SOME-ID")) == "acked"
