#!/usr/bin/env python3
"""Build canonical UI-00 design arsenal registry surface.

Reads the authoritative bucket report and produces a machine-readable,
operator-visible registry artifact under state/continuity/latest/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover
    Draft202012Validator = None

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SOURCE_REPORT = DEFAULT_REPO_ROOT / "reports/openclaw_frontend_design_arsenal_future_bucket_2026-04-03.md"
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "ops/openclaw/architecture/schemas/ui00_design_arsenal_registry.schema.json"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "state/continuity/latest"
DEFAULT_LATEST_NAME = "ui00_design_arsenal_registry_latest.json"

SCHEMA = "openclaw.ui00_design_arsenal_registry.v1"
BUCKET_PREFIX_RE = re.compile(r"^[A-E]\.")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"file not found: {path}")
    except Exception as exc:
        raise SystemExit(f"failed to read {path}: {exc}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate UI-00 design arsenal registry surface")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    ap.add_argument("--source-report", default=str(DEFAULT_SOURCE_REPORT))
    ap.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--latest-name", default=DEFAULT_LATEST_NAME)
    ap.add_argument("--validate", action="store_true", help="Run JSON Schema validation if available")
    ap.add_argument("--json", action="store_true", help="Emit summary JSON to stdout")
    return ap.parse_args(argv)


def extract_bucket_entries(text: str) -> List[Dict[str, Any]]:
    """Parse markdown sections A-E and extract repo entries."""
    entries: List[Dict[str, Any]] = []
    lines = text.splitlines()
    current_bucket: Optional[str] = None

    for line in lines:
        # Detect bucket headers like "### A1. shadcn-ui/ui"
        if m := re.match(r"^### ([A-E]\d*)\.\s+(.+)$", line.strip()):
            repo_slug = m.group(2).strip()
            if "/" not in repo_slug:
                continue  # skip non-GitHub slugs
            # Map bucket letter to full name
            bucket_letter = m.group(1)[0]  # e.g. "A" from "A1"
            bucket_map = {
                "A": "A. Adoptable frontend substrate",
                "B": "B. Operator cockpit / admin donors",
                "C": "C. Collaborative canvas / design-system / provenance donors",
                "D": "D. Visual asset arsenal",
                "E": "E. Pattern/reference library",
            }
            current_bucket = bucket_map.get(bucket_letter)
            if not current_bucket:
                continue

            # Heuristic confidence based on deep-dive mentions in entry vicinity
            confidence = "medium"
            deep_dive_status = "medium-pass-later"
            # Look ahead a few lines for status keywords
            idx = lines.index(line)
            window = "\n".join(lines[idx:idx+6])
            if "DEEP DIVE NOW" in window:
                confidence = "high"
                deep_dive_status = "deep-dive-now"
            elif "REFERENCE ONLY" in window or "Reject" in window:
                confidence = "low"
                deep_dive_status = "reference-only"

            # Substrate vs reference heuristic
            substrate_vs_reference = "reference"
            if current_bucket.startswith("A.") and confidence == "high":
                substrate_vs_reference = "substrate"

            # Future lane mapping heuristic
            future_lane_mapping: List[str] = []
            if "component" in repo_slug.lower() or "ui" in repo_slug.lower():
                future_lane_mapping.append("UI-01")
            if "admin" in repo_slug.lower() or "cockpit" in repo_slug.lower():
                future_lane_mapping.append("UI-02")
            if "design" in repo_slug.lower() or "canvas" in repo_slug.lower():
                future_lane_mapping.append("UI-03")
            if "icon" in repo_slug.lower() or "font" in repo_slug.lower():
                future_lane_mapping.append("UI-05")

            entries.append({
                "repo": repo_slug,
                "bucket": current_bucket,
                "likely_use": f"Candidate for {current_bucket.lower()}",
                "confidence": confidence,
                "substrate_vs_reference": substrate_vs_reference,
                "future_lane_mapping": future_lane_mapping,
                "deep_dive_status": deep_dive_status,
                "note": "",
            })
            continue

            # Heuristic confidence based on deep-dive mentions
            confidence = "medium"
            deep_dive_status = "medium-pass-later"
            note = ""
            if "DEEP DIVE NOW" in text:
                confidence = "high"
                deep_dive_status = "deep-dive-now"
            elif "REFERENCE ONLY" in text or "Reject" in text:
                confidence = "low"
                deep_dive_status = "reference-only"

            # Substrate vs reference heuristic
            substrate_vs_reference = "reference"
            if current_bucket.startswith("A.") and confidence == "high":
                substrate_vs_reference = "substrate"

            # Future lane mapping heuristic
            future_lane_mapping: List[str] = []
            if "component" in repo_slug.lower() or "ui" in repo_slug.lower():
                future_lane_mapping.append("UI-01")
            if "admin" in repo_slug.lower() or "cockpit" in repo_slug.lower():
                future_lane_mapping.append("UI-02")
            if "design" in repo_slug.lower() or "canvas" in repo_slug.lower():
                future_lane_mapping.append("UI-03")
            if "icon" in repo_slug.lower() or "font" in repo_slug.lower():
                future_lane_mapping.append("UI-05")

            entries.append({
                "repo": repo_slug,
                "bucket": current_bucket,
                "likely_use": f"Candidate for {current_bucket.lower()}",
                "confidence": confidence,
                "substrate_vs_reference": substrate_vs_reference,
                "future_lane_mapping": future_lane_mapping,
                "deep_dive_status": deep_dive_status,
                "note": note,
            })

    return entries


def build_registry(source_report: Path, generated_at: str) -> Dict[str, Any]:
    text = source_report.read_text(encoding="utf-8")
    entries = extract_bucket_entries(text)

    summary: Dict[str, Any] = {
        "total_entries": len(entries),
        "bucket_breakdown": {},
    }
    for entry in entries:
        bucket = entry["bucket"]
        summary["bucket_breakdown"][bucket] = summary["bucket_breakdown"].get(bucket, 0) + 1

    return {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "slice_id": "UI-00",
        "source_report": str(source_report.resolve()),
        "summary": summary,
        "arsenal": entries,
    }


def validate_payload(payload: Dict[str, Any], schema_path: Path) -> None:
    if Draft202012Validator is None:
        print("jsonschema not available; skipping validation", file=sys.stderr)
        return
    schema = load_json(schema_path)
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(payload))
    if errors:
        for err in errors:
            print(f"schema validation error: {err.message} at {'.'.join(str(p) for p in err.path)}", file=sys.stderr)
        raise SystemExit(1)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    source_report = Path(args.source_report).resolve()
    schema_path = Path(args.schema_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    latest_path = output_dir / args.latest_name

    if not source_report.exists():
        raise SystemExit(f"source report not found: {source_report}")

    generated_at = now_iso()
    registry = build_registry(source_report, generated_at)

    if args.validate:
        validate_payload(registry, schema_path)

    write_json(latest_path, registry)

    summary = {
        "ok": True,
        "generated_at": generated_at,
        "latest_written": str(latest_path),
        "entry_count": len(registry["arsenal"]),
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()