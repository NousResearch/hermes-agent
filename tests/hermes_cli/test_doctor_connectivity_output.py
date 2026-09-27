"""Terminal progress rewrites must not leak into redirected doctor output."""

from __future__ import annotations

import io
import sys

from hermes_cli import doctor
from hermes_cli.doctor_connectivity import ProbeResult


def _run_connectivity(monkeypatch, stdout):
    result = ProbeResult("OpenRouter API", [("✓", "OpenRouter API", "")], [])
    monkeypatch.setattr(doctor, "build_probes", lambda: [("OpenRouter API", lambda: result)])
    monkeypatch.setattr(doctor, "run_probes", lambda probes: [result])
    monkeypatch.setattr(sys, "stdout", stdout)
    return doctor._check_api_connectivity(False)


def test_redirected_connectivity_output_has_no_carriage_returns(monkeypatch):
    output = io.StringIO()  # isatty() is False: pipe/file/captured output.

    finding = _run_connectivity(monkeypatch, output)

    rendered = output.getvalue()
    assert "\r" not in rendered
    assert "Running 1 connectivity checks in parallel" in rendered
    assert "OpenRouter API" in rendered
    assert rendered.endswith("\n")
    assert finding.issues == []


def test_tty_connectivity_output_keeps_in_place_progress_rewrite(monkeypatch):
    class _Tty(io.StringIO):
        def isatty(self):
            return True

    output = _Tty()

    _run_connectivity(monkeypatch, output)

    assert "\r" in output.getvalue()
