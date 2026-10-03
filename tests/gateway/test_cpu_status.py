"""Unit coverage for the CPU-pressure classifier (gateway/cpu_status.py).

Host-independent: every reading is injected, so this file never depends on the
machine's live load and never loads the host it runs on.
"""

from __future__ import annotations

from gateway import cpu_status


def test_unreadable_signals_are_unknown_not_ok():
    assert cpu_status.classify_cpu_pressure() == "unknown"
    assert cpu_status.classify_cpu_pressure(None, None) == "unknown"
    assert cpu_status.classify_cpu_pressure("x", "y") == "unknown"
    assert cpu_status.classify_cpu_pressure(-1.0) == "unknown"
    assert cpu_status.classify_cpu_pressure(True) == "unknown"
    assert cpu_status.classify_cpu_pressure(float("nan")) == "unknown"
    assert cpu_status.classify_cpu_pressure(float("inf")) == "unknown"


def test_tiers_are_inclusive_at_their_thresholds():
    assert cpu_status.classify_cpu_pressure(cpu_status.LOAD_ELEVATED_PER_CORE) == "elevated"
    assert cpu_status.classify_cpu_pressure(cpu_status.LOAD_CRITICAL_PER_CORE) == "critical"
    assert cpu_status.classify_cpu_pressure(None, cpu_status.PSI_ELEVATED_PCT) == "elevated"
    assert cpu_status.classify_cpu_pressure(None, cpu_status.PSI_CRITICAL_PCT) == "critical"


def test_thresholds_can_be_overridden_conservatively():
    """A host that only wants the hard stop can raise `elevated` past `critical`."""
    assert cpu_status.classify_cpu_pressure(1.5, elevated=2.0, critical=4.0) == "ok"
    assert cpu_status.classify_cpu_pressure(2.5, elevated=2.0, critical=4.0) == "elevated"


def test_sample_cpu_pressure_never_raises_and_keys_are_optional(monkeypatch):
    monkeypatch.setattr(cpu_status, "_load1_per_core", lambda: None)
    monkeypatch.setattr(cpu_status, "_psi_some_avg60_pct", lambda: None)
    assert cpu_status.sample_cpu_pressure() == {}

    monkeypatch.setattr(cpu_status, "_load1_per_core", lambda: 0.4)
    monkeypatch.setattr(cpu_status, "_psi_some_avg60_pct", lambda: 80.0)
    sample = cpu_status.sample_cpu_pressure()
    assert sample == {"load1_per_core": 0.4, "psi_some_avg60_pct": 80.0}
    # Worse signal wins even when only the two differ in availability.
    assert cpu_status.classify_cpu_pressure(
        sample.get("load1_per_core"), sample.get("psi_some_avg60_pct")
    ) == "critical"


def test_loadavg_absent_platform_is_none_not_zero(monkeypatch):
    """Windows has no os.getloadavg; that must read as unknown, not idle."""
    monkeypatch.setattr(cpu_status.os, "getloadavg", None, raising=False)
    assert cpu_status._load1_per_core() is None


def test_psi_parser_ignores_unrelated_lines(tmp_path):
    path = tmp_path / "cpu"
    path.write_text(
        "some avg10=99.00 avg60=12.50 avg300=1.00 total=7\n"
        "full avg10=1.00 avg60=2.00 avg300=3.00 total=4\n",
        encoding="utf-8",
    )
    assert cpu_status._psi_some_avg60_pct(path) == 12.5


def test_psi_missing_file_is_none(tmp_path):
    assert cpu_status._psi_some_avg60_pct(tmp_path / "nope") is None
