"""P13 isolation proof for scripts/sweep-skill-grants.py dry-run flag.

Verifies:
- --dry-run suppresses the sweep: sweep_expired_grants is NOT called, so
  no revoke events are appended to the profile activity ledger.
- --dry-run prints a dry-run JSON marker and exits 0.
- Without --dry-run, the wrapper still calls sweep_expired_grants (n>0
  emits revoked count; n==0 silent).
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "sweep-skill-grants.py"
_MOD = "sweep_skill_grants_p13_under_test"


def _run_wrapper(monkeypatch, capsys):
    import tools.skill_grants as sg
    sys.modules.pop(_MOD, None)
    spec = importlib.util.spec_from_file_location(_MOD, str(WRAPPER))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit as e:
        assert e.code == 0, f"wrapper exited non-zero: {e.code}"
    return capsys.readouterr()


def test_dry_run_does_not_call_sweep(monkeypatch, capsys):
    """--dry-run must not invoke sweep_expired_grants."""
    import tools.skill_grants as sg
    called = {"n": 0}

    def fake_sweep(ttl_hours=None):
        called["n"] += 1
        return 99

    monkeypatch.setattr(sg, "sweep_expired_grants", fake_sweep)
    monkeypatch.setattr(sys, "argv", ["sweep-skill-grants.py", "--dry-run"])
    out = _run_wrapper(monkeypatch, capsys)
    assert called["n"] == 0, "dry-run invoked sweep_expired_grants"
    parsed = json.loads(out.out)
    assert parsed == {"dry_run": True, "ttl_hours": 24, "action": "sweep_expired_grants"}


def test_live_mode_invokes_sweep_and_emits_json(monkeypatch, capsys):
    """Without --dry-run, the wrapper calls sweep and emits JSON for n>0."""
    import tools.skill_grants as sg

    def fake_sweep(ttl_hours=None):
        return 3

    monkeypatch.setattr(sg, "sweep_expired_grants", fake_sweep)
    monkeypatch.setattr(sys, "argv", ["sweep-skill-grants.py"])
    out = _run_wrapper(monkeypatch, capsys)
    parsed = json.loads(out.out)
    assert parsed == {"expired_grants_revoked": 3, "ttl_hours": 24}


def test_live_mode_silent_when_zero(monkeypatch, capsys):
    """Without --dry-run and n==0, no stdout."""
    import tools.skill_grants as sg

    def fake_sweep(ttl_hours=None):
        return 0

    monkeypatch.setattr(sg, "sweep_expired_grants", fake_sweep)
    monkeypatch.setattr(sys, "argv", ["sweep-skill-grants.py"])
    out = _run_wrapper(monkeypatch, capsys)
    assert out.out == "", f"expected no output for zero revocations, got {out.out!r}"
