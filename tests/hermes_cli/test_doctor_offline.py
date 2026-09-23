"""The offline doctor runs local checks but never provider or external probes (#120219)."""

from types import SimpleNamespace


def test_offline_doctor_skips_network_checks_and_rejects_conflicts(monkeypatch):
    from hermes_cli import doctor
    from hermes_cli.doctor_report import Finding

    calls = []

    def local(fix):
        calls.append("local")
        return Finding()

    def external(fix):
        calls.append("external")
        return Finding()

    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", (("local", local), ("external", external)))
    monkeypatch.setattr(doctor, "OFFLINE_CHECKS", frozenset({local}))
    assert doctor.run_doctor(SimpleNamespace(offline=True, live=False, fix=False, ack=None)) == 0
    assert calls == ["local"]
    assert doctor.run_doctor(SimpleNamespace(offline=True, live=True, fix=False, ack=None)) == 2
    assert calls == ["local"]
