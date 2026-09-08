"""Unit tests for :mod:`gateway.cpu_status` (t_4008d306 host-CPU admission)."""

from __future__ import annotations

from gateway import cpu_status as cs


def test_classify_load_pressure_tiers():
    assert cs.classify_load_pressure(1.0, 8) == "ok"
    assert cs.classify_load_pressure(8.0, 8) == "elevated"
    assert cs.classify_load_pressure(16.0, 8) == "critical"


def test_classify_load_pressure_unknown_on_bad_input():
    assert cs.classify_load_pressure(None, 8) == "unknown"
    assert cs.classify_load_pressure(1.0, None) == "unknown"
    assert cs.classify_load_pressure(1.0, 0) == "unknown"
    assert cs.classify_load_pressure(True, 8) == "unknown"  # bool rejected


def test_classify_psi_pressure_tiers():
    assert cs.classify_psi_pressure(2.0) == "ok"
    assert cs.classify_psi_pressure(25.0) == "elevated"
    assert cs.classify_psi_pressure(76.64) == "critical"
    assert cs.classify_psi_pressure(None) == "unknown"


def test_classify_cpu_pressure_worst_of_both():
    # Load says ok, PSI says critical -> critical.
    assert cs.classify_cpu_pressure(1.0, 8, 90.0) == "critical"
    # Load says critical, PSI unavailable -> critical (single known signal wins).
    assert cs.classify_cpu_pressure(51.47, 8, None) == "critical"
    # Both unavailable -> unknown, never silently "ok".
    assert cs.classify_cpu_pressure(None, None, None) == "unknown"


def test_sample_cpu_never_raises(monkeypatch):
    monkeypatch.setattr(cs.os, "getloadavg", lambda: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(cs, "_read_psi_some_avg60", lambda: None)
    assert cs.sample_cpu() == {"cpu_count": cs.os.cpu_count()} or cs.sample_cpu() == {}


def test_read_psi_some_avg60_parses_real_format(tmp_path):
    psi_file = tmp_path / "cpu"
    psi_file.write_text(
        "some avg10=12.34 avg60=76.64 avg300=50.00 total=1234567\n"
        "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
    )
    assert cs._read_psi_some_avg60(str(psi_file)) == 76.64


def test_read_psi_some_avg60_missing_file_returns_none(tmp_path):
    assert cs._read_psi_some_avg60(str(tmp_path / "does-not-exist")) is None
