"""Offline Langfuse gate contracts. No network."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts" / "langfuse_ci_gate.py"
BASELINES = Path(__file__).resolve().parent / "langfuse_baselines"

_spec = importlib.util.spec_from_file_location("langfuse_ci_gate", GATE)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load langfuse_ci_gate.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_decide_table():
    assert _mod.decide(1.0, 1.0)[0] == 0
    assert _mod.decide(0.5, 1.0)[0] == 1
    assert _mod.decide(None, 1.0)[0] == 1


def test_script_self_check():
    proc = subprocess.run([sys.executable, str(GATE), "--self-check"], check=False)
    assert proc.returncode == 0


def test_live_read_requires_env(monkeypatch):
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)
    assert _mod.live_read() == 2


def test_baselines_are_floors():
    paths = sorted(BASELINES.glob("baseline*.json"))
    assert paths
    for path in paths:
        data = json.loads(path.read_text())
        assert data["dataset"].startswith("hermes-")
        assert data["min_score"] == 1.0
        _mod.load_baseline(path)


if __name__ == "__main__":
    test_decide_table()
    test_baselines_are_floors()
    test_script_self_check()
    print("PASS product gate tests")
