"""Regression for #117276: ``hermes doctor``'s exit status must agree with its unresolved findings."""
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
    from hermes_cli import doctor
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
        check=False,
    )
    assert result.returncode == int(unresolved), result.stdout + result.stderr
    assert ("fixture unresolved problem" if unresolved else "All checks passed") in result.stdout


@pytest.mark.parametrize("on_error", ["Some check failed: {e}", ""])
@pytest.mark.parametrize("fix", [False, True])
def test_doctor_crashed_check_is_unresolved(monkeypatch, capsys, on_error, fix):
    from hermes_cli import doctor
    from hermes_cli.doctor_report import doctor_check
    from hermes_cli.main import cmd_doctor

    @doctor_check(on_error)
    def _check_example_section(should_fix, f):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", ((None, _check_example_section),))
    result = cmd_doctor(SimpleNamespace(fix=fix, ack=None, live=False))
    output = capsys.readouterr().out
    assert result == 1
    assert "All checks passed" not in output
    assert "Doctor could not finish the example section check (RuntimeError)" in output
    assert "hermes doctor --fix" not in output
    if on_error:
        assert "⚠ Some check failed: fixture failure" in output
    else:
        assert "⚠ example section check could not finish" in output


def test_doctor_crashed_check_keeps_partial_findings(monkeypatch, capsys):
    from hermes_cli import doctor
    from hermes_cli.doctor_report import doctor_check

    @doctor_check("")
    def check_partial(should_fix, f):
        f.issues.append("existing repair")
        f.manual_issues.append("existing manual repair")
        f.fixed += 1
        raise RuntimeError("fixture failure")

    finding = check_partial(True)
    assert finding.issues == ["existing repair"]
    assert finding.fixed == 1
    assert finding.manual_issues == [
        "existing manual repair",
        "Doctor could not finish the partial check (RuntimeError); check the settings it reads, re-run `hermes doctor`, and report it if it repeats",
    ]
    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", ((None, lambda fix: finding),))
    assert doctor.run_doctor(SimpleNamespace(fix=True, ack=None, live=False)) == 1
    output = capsys.readouterr().out
    assert "Fixed 1 issue(s)" in output
    assert "existing repair" in output
    assert "existing manual repair" in output
    assert "could not finish the partial check" in output
    assert "All checks passed" not in output


def test_doctor_crashed_check_summary_uses_only_exception_type(capsys):
    from hermes_cli.doctor_report import doctor_check

    @doctor_check("Some check failed: {e}")
    def check_private(should_fix, f):
        raise RuntimeError("sk-live-SECRET-fixture")

    finding = check_private(False)
    capsys.readouterr()
    assert finding.manual_issues
    assert "RuntimeError" in finding.manual_issues[0]
    assert "sk-live-SECRET" not in str(finding.manual_issues)


def test_doctor_check_without_error_handler_still_propagates():
    from hermes_cli.doctor_report import doctor_check

    @doctor_check()
    def check_unhandled(should_fix, f):
        raise RuntimeError("fixture failure")

    with pytest.raises(RuntimeError, match="fixture failure"):
        check_unhandled(False)


def test_doctor_cli_crashed_check_exits_one():
    import subprocess
    import sys
    from pathlib import Path

    program = """
import sys
import hermes_cli.doctor as doctor
from hermes_cli.doctor_report import doctor_check
from hermes_cli.main import main
@doctor_check("")
def _check_subprocess(should_fix, f):
    raise RuntimeError("fixture failure")
doctor.DOCTOR_CHECKS = ((None, _check_subprocess),)
sys.argv = ['hermes', 'doctor']
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "subprocess check could not finish" in result.stdout
    assert "All checks passed" not in result.stdout
