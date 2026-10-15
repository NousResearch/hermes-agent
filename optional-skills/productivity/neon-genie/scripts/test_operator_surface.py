#!/usr/bin/env python3
"""Tests for do run + do capabilities (stdlib only)."""

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


def test_capabilities_json() -> None:
    r = run_cli("do", "capabilities", "--json")
    assert r.returncode == 0, r.stderr
    cap = json.loads(r.stdout)
    assert cap["skill"] == "neon-genie"
    assert cap["authority"] == "advisory_only"
    assert cap["grants_execution"] is False
    assert "product-audit" in cap["recipes"]
    assert "envelope" in cap["packet_types"]
    assert cap["runtime_features"]["repo_mutation"] is False
    print("PASS: capabilities --json")


def test_run_recipe() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ng-run-"))
    out = tmp / "demo"
    try:
        r = run_cli("do", "run", "--recipe", "zero-option", "--out", str(out))
        assert r.returncode == 0, r.stderr + r.stdout
        assert (out / "run-envelope.json").is_file()
        assert (out / "run-receipt.json").is_file()
        assert (out / "HERMES_NEXT.md").is_file()
        env = json.loads((out / "run-envelope.json").read_text(encoding="utf-8"))
        assert env["authority"] == "advisory_only"
        print("PASS: do run --recipe")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_brief_auto_recipe() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ng-run-brief-"))
    out = tmp / "brief-run"
    try:
        brief = ROOT / "examples" / "product-audit.brief.yaml"
        r = run_cli("do", "run", "--brief", str(brief), "--out", str(out))
        assert r.returncode == 0, r.stderr + r.stdout
        assert (out / "run-envelope.json").is_file()
        print("PASS: do run --brief (auto recipe)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_capital_sprint_brief_auto_recipe() -> None:
    """capital_sprint preferred_profiles must not lose to commercial in hints."""
    tmp = Path(tempfile.mkdtemp(prefix="ng-run-capital-"))
    out = tmp / "capital-run"
    try:
        brief = ROOT / "examples" / "capital-sprint.brief.yaml"
        r = run_cli("do", "run", "--brief", str(brief), "--out", str(out))
        assert r.returncode == 0, r.stderr + r.stdout
        assert (out / "run-envelope.json").is_file()
        summary = out / "recipe-summary.json"
        assert summary.is_file(), "auto recipe should emit recipe-summary.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        assert data.get("recipe") == "capital-sprint", data
        print("PASS: do run --brief capital-sprint (auto recipe)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_text_scaffold() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ng-run-text-"))
    out = tmp / "text-run"
    try:
        r = run_cli(
            "do",
            "run",
            "--text",
            "curious exploration only",
            "--no-auto-recipe",
            "--out",
            str(out),
        )
        assert r.returncode == 0, r.stderr + r.stdout
        assert (out / "profile-route.json").is_file()
        assert (out / "run-envelope.json").is_file()
        assert (out / "HERMES_NEXT.md").is_file()
        print("PASS: do run --text scaffold")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    failed = 0
    for fn in (
        test_capabilities_json,
        test_run_recipe,
        test_run_brief_auto_recipe,
        test_run_capital_sprint_brief_auto_recipe,
        test_run_text_scaffold,
    ):
        try:
            fn()
        except AssertionError as exc:
            print(f"FAIL: {fn.__name__}: {exc}", file=sys.stderr)
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: {fn.__name__}: {exc}", file=sys.stderr)
            failed += 1
    if failed:
        print(f"FAIL: {failed} operator surface test(s)", file=sys.stderr)
        return 1
    print("PASS: operator surface tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
