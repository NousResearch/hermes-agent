#!/usr/bin/env python3
"""Deterministic anti-drift guard for repo-review queue truth-surface parity."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_PRIMARY_PATH = "reports/github_repo_review_queue_openclaw_patterns_2026-03-20.md"
DEFAULT_NORMALIZED_PATH = "reports/github_repo_review_queue_openclaw_patterns_2026-03-20_NORMALIZED.md"
DEFAULT_RECONCILIATION_MARKER_PATH = "state/continuity/latest/repo_review_queue_truth_reconciliation.json"

SKIP_STATE_TOKENS = {"skip", "skipped", "deleted", "removed", "dropped", "abandoned"}

ROW_HEADING_RE = re.compile(r"^###\s+(.+?)\s*$")
STATUS_RE = re.compile(r"^- Status:\s+(.+?)\s*$", re.MULTILINE)
REPO_RE = re.compile(r"^- Repo:\s+(.+?)\s*$", re.MULTILINE)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_repo_path(repo_root: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    else:
        p = p.resolve()
    return p


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n")).strip()
    return normalized + "\n"


def normalize_status(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    return token or None


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def collect_row_blocks(text: str) -> Dict[str, str]:
    lines = normalize_text(text).splitlines()
    blocks: Dict[str, str] = {}
    indices: List[Tuple[str, int]] = []

    for idx, line in enumerate(lines):
        m = ROW_HEADING_RE.match(line)
        if m:
            indices.append((m.group(1).strip(), idx))

    for i, (row_id, start) in enumerate(indices):
        end = indices[i + 1][1] if i + 1 < len(indices) else len(lines)
        while end > start and not lines[end - 1].strip():
            end -= 1
        blocks[row_id] = "\n".join(lines[start:end]).strip() + "\n"

    return blocks


def extract_field(pattern: re.Pattern[str], block: str) -> Optional[str]:
    m = pattern.search(block)
    if not m:
        return None
    return m.group(1).strip()


def evaluate(repo_root: Path, primary_path: Path, normalized_path: Path) -> Dict[str, Any]:
    reasons: List[str] = []
    checks: List[Dict[str, Any]] = []

    missing_paths = [
        str(path)
        for path in (primary_path, normalized_path)
        if not path.exists() or not path.is_file()
    ]
    if missing_paths:
        return {
            "schema": "clawd.repo_review_queue_parity_guard.v1",
            "generated_at": now_iso(),
            "decision": "BLOCK",
            "block_reason": "queue_surface_missing",
            "primary_path": str(primary_path),
            "normalized_path": str(normalized_path),
            "checks": [
                {
                    "check": "queue_surface_paths_exist",
                    "ok": False,
                    "missing": missing_paths,
                }
            ],
        }

    primary_text = load_text(primary_path)
    normalized_text = load_text(normalized_path)

    primary_norm = normalize_text(primary_text)
    normalized_norm = normalize_text(normalized_text)
    exact_match = primary_norm == normalized_norm
    checks.append({"check": "queue_surface_exact_match", "ok": exact_match})
    if not exact_match:
        reasons.append("queue_surface_text_drift")

    primary_rows = collect_row_blocks(primary_text)
    normalized_rows = collect_row_blocks(normalized_text)

    primary_ids = sorted(primary_rows.keys())
    normalized_ids = sorted(normalized_rows.keys())
    missing_in_primary = [row_id for row_id in normalized_ids if row_id not in primary_rows]
    missing_in_normalized = [row_id for row_id in primary_ids if row_id not in normalized_rows]
    row_ids_match = not missing_in_primary and not missing_in_normalized
    checks.append(
        {
            "check": "queue_row_ids_match",
            "ok": row_ids_match,
            "primary_row_count": len(primary_ids),
            "normalized_row_count": len(normalized_ids),
            "missing_in_primary": missing_in_primary,
            "missing_in_normalized": missing_in_normalized,
        }
    )
    if not row_ids_match:
        reasons.append("queue_row_id_drift")

    repo_mismatches: List[Dict[str, Any]] = []
    status_mismatches: List[Dict[str, Any]] = []
    block_mismatches: List[str] = []
    row_markers: List[Dict[str, Any]] = []
    explicit_marker_rows: List[str] = []

    all_row_ids = sorted(set(primary_ids).union(normalized_ids))
    for row_id in all_row_ids:
        primary_block = primary_rows.get(row_id)
        normalized_block = normalized_rows.get(row_id)

        primary_status = extract_field(STATUS_RE, primary_block) if isinstance(primary_block, str) else None
        normalized_status = extract_field(STATUS_RE, normalized_block) if isinstance(normalized_block, str) else None
        primary_status_normalized = normalize_status(primary_status)
        normalized_status_normalized = normalize_status(normalized_status)

        if primary_block is None:
            marker_state = "missing_in_primary"
            explicit_marker_rows.append(row_id)
        elif normalized_block is None:
            marker_state = "missing_in_normalized"
            explicit_marker_rows.append(row_id)
        elif primary_status_normalized != normalized_status_normalized:
            marker_state = "status_drift"
        elif primary_status_normalized in SKIP_STATE_TOKENS:
            marker_state = "skip_state"
            explicit_marker_rows.append(row_id)
        else:
            marker_state = "in_sync"

        row_markers.append(
            {
                "row_id": row_id,
                "state": marker_state,
                "present_in_primary": primary_block is not None,
                "present_in_normalized": normalized_block is not None,
                "primary_status": primary_status,
                "normalized_status": normalized_status,
            }
        )

        if primary_block is None or normalized_block is None:
            continue

        primary_repo = extract_field(REPO_RE, primary_block)
        normalized_repo = extract_field(REPO_RE, normalized_block)
        if primary_repo != normalized_repo:
            repo_mismatches.append(
                {
                    "row_id": row_id,
                    "primary_repo": primary_repo,
                    "normalized_repo": normalized_repo,
                }
            )

        if primary_status != normalized_status:
            status_mismatches.append(
                {
                    "row_id": row_id,
                    "primary_status": primary_status,
                    "normalized_status": normalized_status,
                }
            )

        if normalize_text(primary_block) != normalize_text(normalized_block):
            block_mismatches.append(row_id)

    checks.append(
        {
            "check": "queue_repo_urls_match",
            "ok": len(repo_mismatches) == 0,
            "mismatches": repo_mismatches,
        }
    )
    if repo_mismatches:
        reasons.append("queue_repo_url_drift")

    checks.append(
        {
            "check": "queue_statuses_match",
            "ok": len(status_mismatches) == 0,
            "mismatches": status_mismatches,
        }
    )
    if status_mismatches:
        reasons.append("queue_status_drift")

    checks.append(
        {
            "check": "queue_row_blocks_match",
            "ok": len(block_mismatches) == 0,
            "mismatch_count": len(block_mismatches),
            "mismatched_rows": block_mismatches[:50],
        }
    )
    if block_mismatches:
        reasons.append("queue_row_block_drift")

    checks.append(
        {
            "check": "queue_reconciliation_markers_emitted",
            "ok": True,
            "row_count": len(row_markers),
            "explicit_marker_count": len(explicit_marker_rows),
            "explicit_marker_rows": explicit_marker_rows[:50],
        }
    )

    decision = "PASS" if not reasons else "BLOCK"
    return {
        "schema": "clawd.repo_review_queue_parity_guard.v1",
        "generated_at": now_iso(),
        "decision": decision,
        "block_reason": reasons[0] if reasons else None,
        "primary_path": str(primary_path),
        "normalized_path": str(normalized_path),
        "checks": checks,
        "row_markers": row_markers,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic repo-review queue parity guard")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--primary-path", default=DEFAULT_PRIMARY_PATH, help="Primary queue markdown path")
    ap.add_argument("--normalized-path", default=DEFAULT_NORMALIZED_PATH, help="Normalized queue markdown path")
    ap.add_argument(
        "--reconciliation-marker-path",
        default=DEFAULT_RECONCILIATION_MARKER_PATH,
        help="Machine marker output path for row reconciliation state",
    )
    ap.add_argument("--json", action="store_true", help="Pretty-print JSON")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = resolve_repo_path(DEFAULT_REPO_ROOT, str(args.repo_root))
    primary_path = resolve_repo_path(repo_root, str(args.primary_path))
    normalized_path = resolve_repo_path(repo_root, str(args.normalized_path))
    reconciliation_marker_path = resolve_repo_path(repo_root, str(args.reconciliation_marker_path))

    payload = evaluate(repo_root=repo_root, primary_path=primary_path, normalized_path=normalized_path)

    marker_payload = {
        "schema": "clawd.repo_review_queue_truth_reconciliation.v1",
        "generated_at": now_iso(),
        "decision": payload.get("decision"),
        "block_reason": payload.get("block_reason"),
        "primary_path": str(primary_path),
        "normalized_path": str(normalized_path),
        "row_markers": payload.get("row_markers") or [],
    }

    marker_write_error: Optional[str] = None
    try:
        atomic_write(reconciliation_marker_path, json.dumps(marker_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    except Exception as exc:  # pragma: no cover - fail-closed safety path
        marker_write_error = f"{type(exc).__name__}:{exc}"

    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    if marker_write_error is None:
        checks.append(
            {
                "check": "reconciliation_marker_write",
                "ok": True,
                "path": str(reconciliation_marker_path),
            }
        )
    else:
        checks.append(
            {
                "check": "reconciliation_marker_write",
                "ok": False,
                "path": str(reconciliation_marker_path),
                "error": marker_write_error,
            }
        )
        if payload.get("decision") != "BLOCK":
            payload["decision"] = "BLOCK"
            payload["block_reason"] = "queue_reconciliation_marker_write_failed"

    payload["checks"] = checks
    payload["reconciliation_marker_path"] = str(reconciliation_marker_path)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    return 0 if payload.get("decision") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
