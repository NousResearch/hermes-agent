#!/usr/bin/env python3
"""Tests for learning ledger run_id linkage and reconcile (stdlib only)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PY = sys.executable
CLI = SCRIPT_DIR / "neon_genie.py"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PY, str(CLI), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_learn_with_envelope() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ng-learn-"))
    out = tmp / "run"
    ledger = tmp / "ledger.jsonl"
    try:
        r = run_cli("do", "run", "--recipe", "zero-option", "--out", str(out))
        assert r.returncode == 0, r.stderr + r.stdout
        env_path = out / "run-envelope.json"
        assert env_path.is_file()
        run_id = json.loads(env_path.read_text())["run_id"]

        r2 = run_cli(
            "do",
            "learn",
            "--class",
            "proof_failed",
            "--summary",
            "no cash in window",
            "--ledger",
            str(ledger),
            "--envelope",
            str(env_path),
            "--routing-quality",
            "correct",
            "--gate-quality",
            "correct_fail_closed",
        )
        assert r2.returncode == 0, r2.stderr + r2.stdout
        line = ledger.read_text(encoding="utf-8").strip().splitlines()[-1]
        entry = json.loads(line)
        assert entry["run_id"] == run_id
        assert entry["canon_status"] == "PROPOSED"
        assert entry["auto_apply_forbidden"] is True
        assert entry["routing_quality"] == "correct"

        r3 = subprocess.run(
            [
                PY,
                str(SCRIPT_DIR / "reconcile_learning.py"),
                "--ledger",
                str(ledger),
                "--runs-root",
                str(tmp),
                "--json",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert r3.returncode == 0, r3.stderr
        report = json.loads(r3.stdout)
        assert report["counts"]["linked"] == 1
        assert report["counts"]["orphan"] == 0
        assert report["counts"]["unlinked"] == 0
        print("PASS: learn + reconcile linked")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reconcile_orphan() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ng-learn-orphan-"))
    ledger = tmp / "ledger.jsonl"
    try:
        entry = {
            "entry_id": "ll-test",
            "observed_at": "2026-01-01T00:00:00Z",
            "source_run": "ng_run_missing",
            "run_id": "ng_run_missing",
            "observation_class": "other",
            "summary": "orphan test",
            "canon_status": "PROPOSED",
            "authority": "advisory_only",
            "auto_apply_forbidden": True,
        }
        ledger.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        r = subprocess.run(
            [
                PY,
                str(SCRIPT_DIR / "reconcile_learning.py"),
                "--ledger",
                str(ledger),
                "--runs-root",
                str(tmp),
                "--json",
                "--strict",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0
        report = json.loads(r.stdout)
        assert report["counts"]["orphan"] == 1
        print("PASS: reconcile orphan fails strict")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    failed = 0
    for fn in (test_learn_with_envelope, test_reconcile_orphan):
        try:
            fn()
        except AssertionError as exc:
            print(f"FAIL: {fn.__name__}: {exc}", file=sys.stderr)
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: {fn.__name__}: {exc}", file=sys.stderr)
            failed += 1
    if failed:
        return 1
    print("PASS: learning reconcile tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
