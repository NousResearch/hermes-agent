#!/usr/bin/env python3
"""Tests for mandatory run-envelope emission (stdlib only)."""

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


def run_recipe(name: str, out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PY, str(SCRIPT_DIR / "neon_genie.py"), "do", "recipe", "--name", name, "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_recipe_emits_envelope() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ng-env-"))
    out = tmp / "product-audit"
    try:
        r = run_recipe("product-audit", out)
        assert r.returncode == 0, r.stderr + r.stdout
        env_path = out / "run-envelope.json"
        assert env_path.is_file(), "run-envelope.json missing"
        env = json.loads(env_path.read_text(encoding="utf-8"))
        assert env["schema_id"] == "neon-genie/run-envelope"
        assert env["schema_version"] == "1.0.0"
        # privacy_runtime (#17): nested privacy-context object
        assert isinstance(env.get("privacy"), dict) or env.get("privacy_mode") in {
            "local_only",
            "LOCAL_ONLY",
        }
        if isinstance(env.get("privacy"), dict):
            tel = env["privacy"].get("telemetry") or env["privacy"].get("telemetry_status")
            assert tel in (None, "disabled")
        assert env["authority"] == "advisory_only"
        assert env["grants_execution"] is False
        assert env["run_id"].startswith("ng_run_")
        assert "primary_artifact" in env
        assert env["primary_artifact"]["path"]
        assert (out / "run-receipt.json").is_file()
        assert env["mode_status"]["SEAL"] == "complete"
        assert env["wayfinder"]["ingest"] == "run-envelope.json"
        # validate
        v = subprocess.run(
            [
                PY,
                str(SCRIPT_DIR / "validate_packet.py"),
                "--packet",
                str(env_path),
                "--type",
                "envelope",
                "--strict-authority",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert v.returncode == 0, v.stderr + v.stdout
        print("PASS: recipe emits valid run-envelope.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_envelope_requires_receipt() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ng-env-norec-"))
    try:
        r = subprocess.run(
            [
                PY,
                str(SCRIPT_DIR / "build_envelope.py"),
                "--out-dir",
                str(tmp),
                "--write",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0
        assert "NG-SCHEMA-002" in (r.stderr + r.stdout) or "run-receipt" in (r.stderr + r.stdout)
        print("PASS: envelope without receipt fails")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_zero_option_envelope_primary() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ng-env-zo-"))
    out = tmp / "zo"
    try:
        r = run_recipe("zero-option", out)
        assert r.returncode == 0, r.stderr + r.stdout
        env = json.loads((out / "run-envelope.json").read_text(encoding="utf-8"))
        assert env["promotion"]["state"]
        assert len(env["artifacts"]) >= 1
        print("PASS: zero-option envelope")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    failed = 0
    for fn in (
        test_recipe_emits_envelope,
        test_envelope_requires_receipt,
        test_zero_option_envelope_primary,
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
        print(f"FAIL: {failed} envelope test(s)", file=sys.stderr)
        return 1
    print("PASS: run envelope tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
