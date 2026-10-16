"""Regression for #132335: a doctor check that crashes must surface in the exit
status and must not let doctor print "All checks passed!".

Complements the #117276 invariants: exit must reflect findings *and* crashes.
"""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("on_error", ["injected check failed", ""])
def test_doctor_check_crash_is_recorded_as_finding(monkeypatch, capsys, on_error):
    import hermes_cli.doctor as doctor
    from hermes_cli.main import cmd_doctor
    from hermes_cli.doctor_report import Finding, doctor_check

    @doctor_check(on_error)
    def boom(should_fix):
        raise RuntimeError("injected check crashed mid-run")

    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", ((None, boom),))
    result = cmd_doctor(SimpleNamespace(fix=False, ack=None, live=False))
    output = capsys.readouterr().out
    assert result == 1, output
    assert "All checks passed" not in output
    assert boom.__name__ in repr(boom)  # functools.wraps kept the check identity
    if on_error:
        assert "injected check failed" in output


def test_doctor_crash_finding_counts_merge(monkeypatch):
    from hermes_cli.doctor_report import Finding

    a, b = Finding(crashed=1), Finding(crashed=2)
    a.merge(b)
    assert a.crashed == 3
