"""Doctor's exit status must agree with its unresolved findings."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("issues,manual,fixed,fix,expected", [
    (["needs repair"], [], 0, False, 1),
    ([], ["manual repair"], 0, False, 1),
    ([], [], 0, False, 0),
    ([], [], 1, True, 0),
    ([], ["remaining repair"], 1, True, 1),
])
def test_doctor_command_reports_remaining_findings(monkeypatch, capsys, issues, manual, fixed, fix, expected):
    import hermes_cli.doctor as doctor
    from hermes_cli.main import cmd_doctor
    from hermes_cli.doctor_report import Finding

    def check(should_fix):
        assert should_fix is fix
        return Finding(issues=issues, manual_issues=manual, fixed=fixed)

    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", ((None, check),))
    result = cmd_doctor(SimpleNamespace(fix=fix, ack=None, live=False))
    output = capsys.readouterr().out
    assert result == expected
    for issue in issues + manual:
        assert issue in output


def test_doctor_preserves_check_exception(monkeypatch):
    import hermes_cli.doctor as doctor
    from hermes_cli.main import cmd_doctor

    def broken(_fix):
        raise RuntimeError("check failed")

    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", ((None, broken),))
    with pytest.raises(RuntimeError, match="check failed"):
        cmd_doctor(SimpleNamespace(fix=False, ack=None, live=False))


def test_doctor_ack_keeps_success_fast_path(monkeypatch):
    import hermes_cli.doctor as doctor
    from hermes_cli.main import cmd_doctor

    acknowledged = []
    monkeypatch.setattr(doctor, "_ack_advisory", acknowledged.append)
    assert cmd_doctor(SimpleNamespace(ack="example")) is None
    assert acknowledged == ["example"]


def test_live_findings_affect_exit_status(monkeypatch):
    import hermes_cli.doctor as doctor
    import hermes_cli.doctor_live as live
    from hermes_cli.main import cmd_doctor

    def probe(args, issues):
        assert args.live is True
        issues.append("live route unavailable")

    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", ())
    monkeypatch.setattr(live, "maybe_run_live_checks", probe)
    assert cmd_doctor(SimpleNamespace(fix=False, ack=None, live=True)) == 1


@pytest.mark.parametrize("unresolved", [False, True])
def test_doctor_cli_process_status_matches_summary(unresolved):
    import subprocess
    import sys
    from pathlib import Path

    program = f"""
import sys
import hermes_cli.doctor as doctor
from hermes_cli.doctor_report import Finding
from hermes_cli.main import main
issues = ['fixture unresolved problem'] if {unresolved!r} else []
doctor.DOCTOR_CHECKS = ((None, lambda fix: Finding(issues=issues)),)
sys.argv = ['hermes', 'doctor']
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == int(unresolved), result.stdout + result.stderr
    assert ("fixture unresolved problem" if unresolved else "All checks passed") in result.stdout
