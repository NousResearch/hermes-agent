#!/usr/bin/env python3
"""Build WEB-04 UI capture evidence packs from web-capture bundle indexes.

This is a tiny runtime seam that bridges B4 web-capture runtime artifacts
(`web_capture.bundle_index.v2`) into the B8 UI evidence bundle contract
(`clawd.b8_ui_evidence_bundle.v1`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def to_iso(value: Optional[dt.datetime]) -> str:
    if value is None:
        return now_iso()
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_token(raw: Any, fallback: str) -> str:
    text = str(raw or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text).strip("._-")
    return text or fallback


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_ref(path: Path) -> Dict[str, Any]:
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
        ref_uri = f"artifact://{rel.as_posix()}"
    except Exception:
        ref_uri = f"file://{path.resolve()}"
    payload: Dict[str, Any] = {"ref_uri": ref_uri}
    digest = sha256_file(path)
    if digest:
        payload["hash_sha256"] = digest
    return payload


def map_surface_theme(color_scheme: Any) -> str:
    value = str(color_scheme or "").strip().lower()
    if value in {"light", "dark", "system", "other"}:
        return value
    return "other"


def detect_environment(index: Dict[str, Any]) -> str:
    source_url = str(index.get("source_url") or "").lower()
    if any(x in source_url for x in ["127.0.0.1", "localhost", ".local"]):
        return "local"
    if any(x in source_url for x in ["staging", "stage."]):
        return "staging"
    if source_url:
        return "production"
    return "other"


def validate_index_shape(index: Any) -> Optional[str]:
    if not isinstance(index, dict):
        return "index_not_object"
    if str(index.get("schema_version") or "") != "web_capture.bundle_index.v2":
        return "index_schema_version_unsupported"
    if not str(index.get("run_id") or "").strip():
        return "index_run_id_missing"

    artifacts = index.get("artifacts")
    if not isinstance(artifacts, dict):
        return "index_artifacts_missing"
    for key in ["screenshot_png", "dom_snapshot_html", "execution_trace_json"]:
        if not str(artifacts.get(key) or "").strip():
            return f"index_artifacts_missing_{key}"
    return None


def build_bundle(index: Dict[str, Any], index_path: Path, bundle_id: Optional[str]) -> Dict[str, Any]:
    run_id = safe_token(index.get("run_id"), "run")
    macro_id = safe_token(index.get("macro_id"), "macro")
    status = str(index.get("status") or "unknown").strip().lower()
    route = str(index.get("route") or "other").strip().lower()

    viewport = index.get("environment", {}).get("viewport") if isinstance(index.get("environment"), dict) else {}
    width_px = int(viewport.get("width") or 1280) if isinstance(viewport, dict) else 1280
    height_px = int(viewport.get("height") or 800) if isinstance(viewport, dict) else 800
    width_px = max(320, width_px)
    height_px = max(240, height_px)

    captured_at = parse_iso(index.get("captured_at"))
    completed_at = parse_iso(index.get("completed_at"))
    if captured_at is None:
        captured_at = completed_at or dt.datetime.now(dt.timezone.utc)
    if completed_at is None or completed_at < captured_at:
        completed_at = captured_at
    generated_at = completed_at

    artifacts = index.get("artifacts") if isinstance(index.get("artifacts"), dict) else {}
    run_dir = index_path.parent

    screenshot_path = run_dir / str(artifacts.get("screenshot_png"))
    dom_path = run_dir / str(artifacts.get("dom_snapshot_html"))
    trace_path = run_dir / str(artifacts.get("execution_trace_json"))

    screenshot_ref = artifact_ref(screenshot_path)
    dom_ref = artifact_ref(dom_path)
    trace_ref = artifact_ref(trace_path)

    severity = "observation" if status == "ok" else "critical"
    priority = "p2" if status == "ok" else "p0"
    finding_id = "fnd_web_capture_bundle_actionability"
    rec_id = "rec_wire_bundle_into_operator_triage"

    base_bundle = safe_token(bundle_id or f"webcap_{run_id}", "webcap")
    derived_bundle_id = base_bundle if base_bundle.startswith("b8ev_") else f"b8ev_{base_bundle}"

    route_label = route if route in {"fetch", "browser"} else "other"

    return {
        "schema_version": "clawd.b8_ui_evidence_bundle.v1",
        "bundle_id": derived_bundle_id,
        "surface": {
            "surface_id": f"surf_web_capture_{macro_id}",
            "surface_kind": "workflow_view",
            "route": f"/web_capture/{macro_id}",
            "view_name": "Web Capture Runtime Evidence",
            "viewport": {
                "width_px": width_px,
                "height_px": height_px,
                "device_pixel_ratio": 1,
                "theme": map_surface_theme(index.get("environment", {}).get("color_scheme") if isinstance(index.get("environment"), dict) else None),
                "locale": str(index.get("environment", {}).get("locale") or "en-US") if isinstance(index.get("environment"), dict) else "en-US",
            },
        },
        "capture": {
            "captured_at": to_iso(captured_at),
            "capture_session_id": f"cap_webcap_{run_id}",
            "environment": detect_environment(index),
            "operator_id": "web_capture_runtime",
        },
        "evidence": {
            "screenshot": {
                "screenshot_ref_id": f"shot_webcap_{run_id}",
                "artifact_ref": screenshot_ref,
                "mime_type": "image/png",
                "width_px": width_px,
                "height_px": height_px,
                "captured_at": to_iso(captured_at),
                "regions": [
                    {
                        "region_id": "reg_primary_capture_surface",
                        "label": "Primary capture surface",
                        "bbox": {
                            "coordinate_space": "pixel",
                            "x": 0,
                            "y": 0,
                            "width": width_px,
                            "height": height_px,
                        },
                    }
                ],
            },
            "dom_component_unavailable_reason": "web_capture bundle index does not provide normalized component refs; use dom_snapshot_html source record.",
            "state_snapshot": {
                "state_ref_id": f"state_webcap_{run_id}",
                "artifact_ref": trace_ref,
                "format": "json",
                "captured_at": to_iso(completed_at),
                "state_fact_refs": [
                    {
                        "fact_id": "fact_capture_status",
                        "path": "$.status",
                        "value_excerpt": status or "unknown",
                    },
                    {
                        "fact_id": "fact_capture_route",
                        "path": "$.route",
                        "value_excerpt": route_label,
                    },
                    {
                        "fact_id": "fact_capture_result_reason",
                        "path": "$.result_reason",
                        "value_excerpt": str(index.get("result_reason") or "n/a"),
                    },
                ],
            },
            "ui_state_metadata": {
                "run_id": str(index.get("run_id") or ""),
                "macro_id": str(index.get("macro_id") or ""),
                "source_url": str(index.get("source_url") or ""),
            },
        },
        "findings": [
            {
                "finding_id": finding_id,
                "title": "Web capture bundle should be directly linkable from operator triage surfaces",
                "category": "actionability",
                "severity": severity,
                "confidence": 0.86,
                "rationale": "Capture artifacts exist but require manual run-dir traversal. A single bundle link in operator surfaces lowers navigation friction and improves incident triage speed.",
                "evidence_links": {
                    "screenshot_region_refs": ["reg_primary_capture_surface"],
                    "state_fact_refs": ["fact_capture_status", "fact_capture_route"],
                },
                "recommendations": [
                    {
                        "recommendation_id": rec_id,
                        "action": "Project this UI evidence bundle ID into operator triage rows for web-capture incidents.",
                        "expected_outcome": "Operators can open screenshot/DOM/trace evidence without manual artifact path lookup.",
                        "effort": "low",
                        "validation_hint": "Check triage output includes bundle_id and opens a valid packet.",
                        "guardrail": "Do not auto-resolve incidents based only on evidence link availability.",
                    }
                ],
                "status": "open",
                "provenance": {
                    "heuristic_refs": ["rule_B8_ACTIONABILITY_004"],
                    "evaluator": "web04.bridge.v1",
                    "evaluated_at": to_iso(generated_at),
                },
            }
        ],
        "recommendation_set": [
            {
                "recommendation_id": rec_id,
                "finding_ids": [finding_id],
                "priority": priority,
                "action": "Wire web-capture UI evidence bundles into operator-facing warning/actionability surfaces.",
                "rationale": "Bridges deterministic capture artifacts into bounded, operator-readable packets.",
                "verification_steps": [
                    "Build a bundle from web_capture bundle_index.v2 and validate it with b8_ui_evidence_bundle_validate.py.",
                    "Verify operator surface can reference bundle_id without path spelunking.",
                ],
                "owner": "web.capture.operator_surface",
                "target_release": "web04_followthrough",
            }
        ],
        "review_summary": {
            "severity_counts": {
                "observation": 1 if severity == "observation" else 0,
                "minor": 0,
                "moderate": 0,
                "critical": 1 if severity == "critical" else 0,
            },
            "risk_statement": "Blocked/failed captures require immediate operator actionability path; successful captures still benefit from one-click evidence linkage." if severity == "critical" else "Capture succeeded; remaining risk is operator navigation friction without direct evidence linking.",
        },
        "provenance": {
            "generated_at": to_iso(generated_at),
            "generator": {
                "layer": "B8",
                "run_id": f"b8run_webcap_bridge_{run_id}",
                "contract_version": "clawd.b8_ui_evidence_bundle.v1",
            },
            "source_records": [
                {
                    "source_id": "src_webcap_screenshot",
                    "source_kind": "screenshot",
                    "ref_uri": screenshot_ref["ref_uri"],
                    "captured_at": to_iso(captured_at),
                    "collector": "ops.web_capture.run_macro",
                    **({"hash_sha256": screenshot_ref["hash_sha256"]} if screenshot_ref.get("hash_sha256") else {}),
                    "immutable": True,
                },
                {
                    "source_id": "src_webcap_dom_snapshot",
                    "source_kind": "dom_component_tree",
                    "ref_uri": dom_ref["ref_uri"],
                    "captured_at": to_iso(captured_at),
                    "collector": "ops.web_capture.run_macro",
                    **({"hash_sha256": dom_ref["hash_sha256"]} if dom_ref.get("hash_sha256") else {}),
                    "immutable": True,
                },
                {
                    "source_id": "src_webcap_execution_trace",
                    "source_kind": "state_snapshot",
                    "ref_uri": trace_ref["ref_uri"],
                    "captured_at": to_iso(completed_at),
                    "collector": "ops.web_capture.run_macro",
                    **({"hash_sha256": trace_ref["hash_sha256"]} if trace_ref.get("hash_sha256") else {}),
                    "immutable": True,
                },
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WEB-04 UI capture evidence packs from web-capture index")
    parser.add_argument("--index", required=True, help="Path to web_capture bundle index JSON (v2)")
    parser.add_argument("--out", required=True, help="Output path for generated B8 UI evidence bundle JSON")
    parser.add_argument("--bundle-id", help="Optional explicit bundle_id (must still satisfy B8 schema pattern)")
    parser.add_argument("--json", action="store_true", help="Pretty JSON summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    index_path = Path(args.index).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    try:
        index = load_json(index_path)
    except Exception as exc:
        result = {
            "ok": False,
            "error": "index_unreadable",
            "index_path": str(index_path),
            "detail": str(exc),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
        return 1

    issue = validate_index_shape(index)
    if issue:
        result = {
            "ok": False,
            "error": "index_validation_failed",
            "issue": issue,
            "index_path": str(index_path),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
        return 1

    bundle = build_bundle(index, index_path=index_path, bundle_id=args.bundle_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = {
        "ok": True,
        "index_path": str(index_path),
        "output_path": str(out_path),
        "bundle_id": bundle.get("bundle_id"),
        "schema_version": bundle.get("schema_version"),
        "source_run_id": index.get("run_id"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
