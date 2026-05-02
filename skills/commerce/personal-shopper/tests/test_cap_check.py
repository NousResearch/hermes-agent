"""Pytest for cap_check ledger + decision logic.

Uses tmp_path fixtures to keep the production ledger untouched.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "cap_check.py"


def _run(*args: str) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout.strip()
    return proc.returncode, (json.loads(out) if out else {})


def test_under_cap_ok(tmp_path: Path) -> None:
    rc, payload = _run("--price-eur", "14.20", "--ledger", str(tmp_path / "l.json"))
    assert rc == 0
    assert payload["ok"] is True


def test_over_per_purchase_cap_blocked(tmp_path: Path) -> None:
    rc, payload = _run("--price-eur", "80", "--ledger", str(tmp_path / "l.json"))
    assert rc == 3
    assert payload["ok"] is False
    assert "per-purchase" in (payload["reason"] or "")


def test_per_day_cap_accumulates(tmp_path: Path) -> None:
    ledger = tmp_path / "l.json"
    # Commit 60
    _run("--price-eur", "60", "--ledger", str(ledger), "--commit", "--cap-per-purchase", "70")
    # Try another 50; total would be 110 > default per-day cap of 100
    rc, payload = _run("--price-eur", "50", "--ledger", str(ledger), "--cap-per-purchase", "70")
    assert rc == 3
    assert payload["ok"] is False
    assert "per-day" in (payload["reason"] or "")


def test_commit_persists(tmp_path: Path) -> None:
    ledger = tmp_path / "l.json"
    _run("--price-eur", "10", "--ledger", str(ledger), "--commit")
    data = json.loads(ledger.read_text())
    assert len(data["entries"]) == 1
    assert data["entries"][0]["price_eur"] == 10
