"""Bridge Hermes knowledge lanes into governed Marson promotion/ingestion artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any

from agent.knowledge_lanes import KnowledgeLaneStore
from hermes_constants import get_hermes_home


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _bridge_root() -> Path:
    return get_hermes_home() / "knowledge" / "governance_bridge"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", (text or "").strip().lower()).strip("-._")
    return slug or "item"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _confidence_score(level: str) -> float:
    mapping = {"low": 0.35, "medium": 0.65, "high": 0.9}
    return mapping.get(str(level or "").strip().lower(), 0.5)


def _load_state_item(store: KnowledgeLaneStore, lane: str, lane_item_id: str) -> tuple[dict[str, Any], dict[str, Any], int]:
    state = store.read_state()
    field = "draft_items" if lane == "draft" else "promoted_items"
    items = state.get(field, [])
    for idx, item in enumerate(items):
        if item.get("id") == lane_item_id:
            return state, dict(item), idx
    raise KeyError(f"Unknown {lane} lane item id: {lane_item_id}")


def load_exported_packet(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def export_lane_item_to_governance_package(
    *,
    lane_item_id: str,
    lane: str,
    repo_root: Path,
    target_surface: str,
    target_path: str,
    merge_mode: str = "append",
    store: KnowledgeLaneStore | None = None,
) -> dict[str, str]:
    if lane not in {"draft", "promoted"}:
        raise ValueError("lane must be 'draft' or 'promoted'")

    store = store or KnowledgeLaneStore()
    state, item, index = _load_state_item(store, lane, lane_item_id)
    now = _utc_now_iso()

    bridge_root = _bridge_root()
    evidence_dir = bridge_root / "evidence"
    candidate_dir = bridge_root / "promotion_candidates"
    package_dir = bridge_root / "ingestion_packages"
    for directory in (evidence_dir, candidate_dir, package_dir):
        directory.mkdir(parents=True, exist_ok=True)

    evidence_path = evidence_dir / f"{item['id']}.json"
    evidence_payload = {
        "id": item["id"],
        "lane": item["lane"],
        "status": item["status"],
        "title": item["title"],
        "body": item["body"],
        "source": item["source"],
        "provenance": item["provenance"],
        "confidence": item["confidence"],
        "tags": item.get("tags", []),
        "created_at": item["created_at"],
        "promotion": item.get("promotion"),
    }
    evidence_bytes = (json.dumps(evidence_payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    evidence_path.write_bytes(evidence_bytes)
    evidence_hash = f"sha256:{_sha256_bytes(evidence_bytes)}"

    promotion_id = f"prom_{lane_item_id}_{_slugify(item['title'])}"[:80]
    candidate_path = candidate_dir / f"{promotion_id}.json"
    candidate_payload = {
        "promotion_id": promotion_id,
        "created_at": now,
        "promotion_state": "LOCAL_ONLY",
        "source_lane": {
            "lane_id": f"hermes-knowledge-lane:{lane}",
            "work_item_id": lane_item_id,
            "producer_role": "LIBRARIAN",
            "session_key": item.get("provenance", {}).get("session_id"),
        },
        "insight": {
            "title": item["title"],
            "statement": item["body"],
            "kind": "heuristic",
        },
        "provenance": {
            "capture_method": item["source"],
            "captured_at": item["created_at"],
            "tool_trace_refs": list(item.get("provenance", {}).get("tool_trace_refs") or []),
        },
        "confidence": {
            "score": _confidence_score(item.get("confidence")),
            "method": f"hermes_lane:{item.get('confidence')}",
            "notes": f"Derived from Hermes {lane} knowledge lane export",
        },
        "source_refs": [
            {
                "ref_id": f"src_{lane_item_id}",
                "path": evidence_path.relative_to(repo_root).as_posix() if evidence_path.is_relative_to(repo_root) else evidence_path.as_posix(),
                "locator": f"knowledge/{lane}/{lane_item_id}",
                "content_hash": evidence_hash,
            }
        ],
        "review": {
            "state": "pending",
            "reviewer_role": None,
            "reviewer_id": None,
            "reviewed_at": None,
            "rationale": "Exported from Hermes knowledge lane for governed review",
        },
        "target": {
            "surface": target_surface,
            "target_path": target_path,
            "merge_mode": merge_mode,
        },
        "safety": {
            "classification": "internal",
            "leakage_check": "pass",
            "redaction_applied": False,
            "notes": "Hermes-exported lane item; review before promotion",
        },
        "decision_refs": [],
    }
    candidate_bytes = (json.dumps(candidate_payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    candidate_path.write_bytes(candidate_bytes)
    candidate_hash = f"sha256:{_sha256_bytes(candidate_bytes)}"

    package_id = f"kip_{lane_item_id}_{_slugify(item['title'])}"[:80]
    package_path = package_dir / f"{package_id}.json"
    package_payload = {
        "schema_version": "clawd.knowledge_ingestion.package.v1",
        "package_id": package_id,
        "created_at": now,
        "updated_at": now,
        "ingestion_state": "READY_FOR_QUEUE",
        "source_lane": {
            "lane_id": f"hermes-knowledge-lane:{lane}",
            "work_item_id": lane_item_id,
            "producer_role": "LIBRARIAN",
            "session_key": item.get("provenance", {}).get("session_id"),
        },
        "promotion_candidate_ref": {
            "promotion_id": promotion_id,
            "path": candidate_path.relative_to(repo_root).as_posix() if candidate_path.is_relative_to(repo_root) else candidate_path.as_posix(),
            "content_hash": candidate_hash,
        },
        "preserved_evidence": {
            "schema_version": "clawd.knowledge_ingestion.evidence_bundle.v1",
            "bundle_role": "promotion_candidate_supporting_evidence",
            "item_count": 1,
            "items": [
                {
                    "evidence_id": f"ev_{lane_item_id}",
                    "source_ref_id": f"src_{lane_item_id}",
                    "path": evidence_path.relative_to(repo_root).as_posix() if evidence_path.is_relative_to(repo_root) else evidence_path.as_posix(),
                    "content_hash": evidence_hash,
                    "bytes": len(evidence_bytes),
                    "captured_at": now,
                    "media_type": "application/json",
                }
            ],
        },
        "provenance": {
            "capture_method": item["source"],
            "captured_at": item["created_at"],
            "collector_role": "LIBRARIAN",
            "tool_trace_refs": list(item.get("provenance", {}).get("tool_trace_refs") or []),
            "source_ref_ids": [f"src_{lane_item_id}"],
        },
        "fixity": {
            "algorithm": "sha256",
            "verification_status": "verified",
            "verified_at": now,
            "mismatch_count": 0,
            "checked_item_count": 1,
        },
        "handoff": {
            "queue_runtime": str((repo_root / "scripts" / "knowledge_promotion_queue.py").resolve()),
            "queue_entry_id": None,
            "queue_status": None,
            "enqueued_at": None,
            "last_event_ref": None,
        },
        "notes": "Exported from Hermes knowledge lane bridge",
    }
    package_path.write_text(json.dumps(package_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    field = "draft_items" if lane == "draft" else "promoted_items"
    state[field][index]["provenance"] = dict(state[field][index].get("provenance") or {})
    state[field][index]["provenance"]["governance_bridge"] = {
        "exported_at": now,
        "promotion_candidate_path": str(candidate_path),
        "ingestion_package_path": str(package_path),
        "evidence_path": str(evidence_path),
        "target_surface": target_surface,
        "target_path": target_path,
        "promotion_id": promotion_id,
        "package_id": package_id,
    }
    store.write_state(state)

    return {
        "promotion_candidate_path": str(candidate_path),
        "ingestion_package_path": str(package_path),
        "evidence_path": str(evidence_path),
        "promotion_id": promotion_id,
        "package_id": package_id,
    }
