#!/usr/bin/env python3
"""Integration tests for privacy packaging surface (stdlib only).

Aligned with privacy_runtime (#17) + dual-enforcement NG-PRIV gates.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PY = sys.executable


def test_receipt_includes_privacy_fields() -> None:
    with tempfile.TemporaryDirectory(prefix="ng-priv-rec-") as td:
        out = Path(td) / "receipt.json"
        r = subprocess.run(
            [
                PY,
                str(SCRIPT_DIR / "build_receipt.py"),
                "--profiles",
                "core,product_architecture",
                "--status",
                "PROPOSED",
                "--out",
                str(out),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr + r.stdout
        rec = json.loads(out.read_text(encoding="utf-8"))
        assert rec["privacy_mode"] in {"local_only", "LOCAL_ONLY"}
        assert rec["telemetry_status"] == "disabled"
        assert isinstance(rec.get("external_actions"), list)
        assert "privacy" in rec["profiles_loaded"]
        assert "core" in rec["profiles_loaded"]
        assert isinstance(rec.get("privacy"), dict)
        print("PASS: receipt privacy fields")


def test_validate_rejects_local_only_with_sent_action() -> None:
    with tempfile.TemporaryDirectory(prefix="ng-priv-val-") as td:
        path = Path(td) / "bad-receipt.json"
        bad = {
            "status": "PROPOSED",
            "profiles_loaded": ["core", "privacy"],
            "claims_by_label": {
                "OBSERVED": [],
                "INFERRED": [],
                "SPECULATIVE": [],
                "NOT_COMPUTABLE": [],
            },
            "not_computable_fields": [],
            "promotion_state": "RAW_SIGNAL",
            "human_review_required": True,
            "privacy_mode": "local_only",
            "external_actions": [
                {
                    "action_id": "ea_1",
                    "outcome": "ALLOW",
                    "decision": "ALLOW",
                    "sent": True,
                    "destination": "example.com",
                    "tool_or_provider": "web_search",
                    "purpose": "test",
                    "recorded_at": "2026-08-06T00:00:00Z",
                }
            ],
            "artifact_paths": [],
            "telemetry_status": "disabled",
            "retention_statement": "x",
            "privacy_warnings": [],
            "deletion_instructions": "x",
        }
        path.write_text(json.dumps(bad), encoding="utf-8")
        r = subprocess.run(
            [
                PY,
                str(SCRIPT_DIR / "validate_packet.py"),
                "--packet",
                str(path),
                "--type",
                "receipt",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0
        assert "NG-PRIV-003" in (r.stderr + r.stdout)
        print("PASS: local_only + sent rejected")


def test_validate_rejects_telemetry_enabled() -> None:
    with tempfile.TemporaryDirectory(prefix="ng-priv-tel-") as td:
        path = Path(td) / "tel.json"
        rec = {
            "status": "PROPOSED",
            "profiles_loaded": ["core", "privacy"],
            "claims_by_label": {
                "OBSERVED": [],
                "INFERRED": [],
                "SPECULATIVE": [],
                "NOT_COMPUTABLE": [],
            },
            "not_computable_fields": [],
            "promotion_state": "RAW_SIGNAL",
            "human_review_required": True,
            "telemetry_status": "enabled",
            "privacy_mode": "local_only",
            "external_actions": [],
            "artifact_paths": [],
            "retention_statement": "x",
            "privacy_warnings": [],
            "deletion_instructions": "x",
        }
        path.write_text(json.dumps(rec), encoding="utf-8")
        r = subprocess.run(
            [
                PY,
                str(SCRIPT_DIR / "validate_packet.py"),
                "--packet",
                str(path),
                "--type",
                "receipt",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0
        assert "NG-PRIV-002" in (r.stderr + r.stdout)
        print("PASS: telemetry enabled rejected")


def test_route_includes_privacy() -> None:
    r = subprocess.run(
        [
            PY,
            str(SCRIPT_DIR / "neon_genie.py"),
            "do",
            "route",
            "--text",
            "product audit for missing buyer",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    data = json.loads(r.stdout)
    selected = data.get("selected") or data.get("profiles") or []
    assert "core" in selected
    assert "privacy" in selected
    assert selected.index("privacy") == selected.index("core") + 1
    print("PASS: route includes privacy")


def test_do_privacy_json() -> None:
    r = subprocess.run(
        [PY, str(SCRIPT_DIR / "neon_genie.py"), "do", "privacy", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    data = json.loads(r.stdout)
    # privacy_diagnostics (#17) shape
    assert str(data.get("repository_telemetry") or data.get("telemetry_status") or "").lower() in {
        "disabled",
        "disabled",
    } or data.get("repository_telemetry") == "DISABLED"
    assert data.get("global_privacy_disable") == "FORBIDDEN" or data.get("wayfinder_required") is False
    print("PASS: do privacy --json")


def test_sealed_missing_privacy_mode_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="ng-priv-y-") as td:
        path = Path(td) / "sealed.json"
        rec = {
            "status": "SEALED",
            "profiles_loaded": ["core", "privacy"],
            "claims_by_label": {
                "OBSERVED": [],
                "INFERRED": [],
                "SPECULATIVE": [],
                "NOT_COMPUTABLE": [],
            },
            "not_computable_fields": [],
            "promotion_state": "RAW_SIGNAL",
            "human_review_required": True,
            "external_actions": [],
            "artifact_paths": [],
            "telemetry_status": "disabled",
            "retention_statement": "x",
            "privacy_warnings": [],
            "deletion_instructions": "x",
            # missing privacy_mode
        }
        path.write_text(json.dumps(rec), encoding="utf-8")
        r = subprocess.run(
            [
                PY,
                str(SCRIPT_DIR / "validate_packet.py"),
                "--packet",
                str(path),
                "--type",
                "receipt",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0
        assert "NG-PRIV-005" in (r.stderr + r.stdout)
        print("PASS: SEALED missing privacy_mode → NG-PRIV-005")


def test_recipe_envelope_validates() -> None:
    with tempfile.TemporaryDirectory(prefix="ng-priv-env-") as td:
        out = Path(td) / "run"
        r = subprocess.run(
            [
                PY,
                str(SCRIPT_DIR / "neon_genie.py"),
                "do",
                "recipe",
                "--name",
                "product-audit",
                "--out",
                str(out),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr + r.stdout
        env_path = out / "run-envelope.json"
        assert env_path.is_file()
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
        env = json.loads(env_path.read_text(encoding="utf-8"))
        assert env.get("privacy_mode") in {"local_only", "LOCAL_ONLY", None} or "privacy" in env
        print("PASS: recipe envelope validates with privacy")


def main() -> int:
    test_receipt_includes_privacy_fields()
    test_validate_rejects_local_only_with_sent_action()
    test_validate_rejects_telemetry_enabled()
    test_route_includes_privacy()
    test_do_privacy_json()
    test_sealed_missing_privacy_mode_fails()
    test_recipe_envelope_validates()
    print("PASS: all privacy surface tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
