#!/usr/bin/env python3
"""Shared memory fabric lifecycle runtime (Wave 7, bounded v1).

Deterministic local runtime for typed memory objects:
- promote: materialize canonical memory object from an approved promotion candidate
- conflict: record contradiction/conflict-set rows with explicit ownership
- demote: apply staleness/divergence demotion with append-only records
- status: inspect registry/object state

Wrapper-only enforcement: mutating commands (promote, conflict, demote) require OPENCLAW_INTERNAL_MUTATION=1
and OPENCLAW_INTERNAL_MUTATION_CALLSITE allowlisted (default: continuity.sh:shared-memory).
Direct token‑path calls are not supported; use continuity.sh shared‑memory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_PROMOTION_SCHEMA_PATH = Path("docs/ops/schemas/promotion_candidate.schema.json")
DEFAULT_OBJECT_PATH = Path("state/continuity/shared_memory/objects")
DEFAULT_CONFLICT_PATH = Path("state/continuity/shared_memory/conflicts")
DEFAULT_DEMOTION_PATH = Path("state/continuity/shared_memory/demotions")
DEFAULT_REGISTRY_PATH = Path("state/continuity/shared_memory/registry.json")
DEFAULT_COMPACTION_PATH = Path("state/continuity/shared_memory/compaction/latest.json")

OBJECT_SCHEMA = "clawd.shared_memory_object.v1"
REGISTRY_SCHEMA = "clawd.shared_memory_registry.v1"
CONFLICT_SCHEMA = "clawd.shared_memory_conflict_record.v1"
DEMOTION_SCHEMA = "clawd.shared_memory_demotion_record.v1"
WRAPPER_REQUIRED_SCHEMA = "clawd.shared_memory_fabric.wrapper_contract.v1"
DEFAULT_ALLOWED_MUTATION_CALLSITES = {
    "continuity.sh:shared-memory",
}

ROLE_SET = {"PLANNER", "EXECUTOR", "VALIDATOR", "RESEARCHER", "SRE", "LIBRARIAN"}
CANONICAL_STATES = {
    "PROMOTED_CANONICAL",
    "CONFLICTED",
    "DEMOTED_STALE",
    "DEMOTED_SUPERSEDED",
    "DEMOTED_INVALIDATED",
    "ARCHIVED",
}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "when",
    "where",
    "while",
    "with",
    "will",
    "would",
    "should",
    "can",
    "cannot",
    "has",
    "have",
    "had",
    "must",
    "you",
    "we",
}
COMPACTION_STRATEGY_LEGACY = "legacy"
COMPACTION_STRATEGY_SIGNATURE = "signature"
RETRIEVAL_STRATEGY_TOKEN_OVERLAP = "token_overlap"
RETRIEVAL_STRATEGY_TFIDF_HYBRID = "tfidf_hybrid"

OBJECT_ID_RE = re.compile(r"^smo_[a-z0-9._-]+$")
PROMOTION_ID_RE = re.compile(r"^prom_[a-z0-9._-]+$")
QUEUE_ENTRY_RE = re.compile(r"^kpq_[a-z0-9._-]+$")


try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SHARED_MEMORY_SCHEMA_PATH = Path("docs/ops/schemas/shared_memory_object.schema.json")
CONFLICT_SCHEMA_PATH = Path("docs/ops/schemas/shared_memory_conflict_record.schema.json")
DEMOTION_SCHEMA_PATH = Path("docs/ops/schemas/shared_memory_demotion_record.schema.json")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def resolve_repo_path(repo_root: Path, raw: str | Path) -> Path:
    if isinstance(raw, Path):
        candidate = raw.expanduser()
    else:
        candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _safe_rel(repo_root: Path, target: Path) -> str:
    return target.resolve().relative_to(repo_root).as_posix()


def _load_registry(repo_root: Path, registry_path: Path) -> Dict[str, Any]:
    path = resolve_repo_path(repo_root, registry_path)
    if not is_within(repo_root, path):
        raise RuntimeError("unsafe_registry_path")

    if not path.exists():
        return {
            "schema": REGISTRY_SCHEMA,
            "updated_at": now_iso(),
            "objects": [],
        }

    payload = load_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError("registry_not_object")
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise RuntimeError("registry_schema_mismatch")

    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise RuntimeError("registry_objects_not_list")

    out = dict(payload)
    out["objects"] = [row for row in objects if isinstance(row, dict)]
    return out


def _upsert_registry_entry(registry: Dict[str, Any], row: Dict[str, Any]) -> None:
    objects = registry.get("objects") if isinstance(registry.get("objects"), list) else []
    obj_id = str(row.get("object_id") or "")
    updated = False
    for existing in objects:
        if str(existing.get("object_id") or "") == obj_id:
            existing.update(row)
            updated = True
            break
    if not updated:
        objects.append(row)
    registry["objects"] = sorted(objects, key=lambda r: str(r.get("updated_at") or ""), reverse=True)
    registry["updated_at"] = now_iso()


def _validate_against_schema(payload: Any, schema_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "validator_unavailable", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists() or not schema_path.is_file():
        return False, "schema_missing", {"error": "schema_missing", "schema_path": str(schema_path)}

    schema = load_json(schema_path)
    if not isinstance(schema, dict):
        return False, "schema_invalid", {"error": "schema_not_object"}

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return True, "", {}

    err = errors[0]
    return (
        False,
        "schema_invalid",
        {
            "error": "schema_validation_failed",
            "data_path": "$" if not err.absolute_path else "$/" + "/".join(str(x) for x in err.absolute_path),
            "message": str(err.message),
        },
    )


def _load_object_by_id(repo_root: Path, object_id: str, objects_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    if not OBJECT_ID_RE.fullmatch(object_id):
        raise RuntimeError("object_id_invalid")
    path = resolve_repo_path(repo_root, objects_dir / f"{object_id}.json")
    if not is_within(repo_root, path):
        raise RuntimeError("unsafe_object_path")
    if not path.exists() or not path.is_file():
        raise RuntimeError("object_not_found")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError("object_not_object")
    return path, payload


def _workflow_decision_ok(workflow_payload: Mapping[str, Any]) -> bool:
    return str(workflow_payload.get("decision") or "") == "PASS"


def cmd_promote(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    objects_dir = resolve_repo_path(repo_root, args.objects_path)
    registry_path = resolve_repo_path(repo_root, args.registry_path)

    if not is_within(repo_root, objects_dir):
        return 2, {"schema": "clawd.shared_memory_fabric.result.v1", "action": "promote", "ok": False, "error": "unsafe_objects_path"}
    if not is_within(repo_root, registry_path):
        return 2, {"schema": "clawd.shared_memory_fabric.result.v1", "action": "promote", "ok": False, "error": "unsafe_registry_path"}

    try:
        candidate_path = resolve_repo_path(repo_root, args.candidate)
        if not is_within(repo_root, candidate_path):
            raise RuntimeError("candidate_path_outside_repo")
        if not candidate_path.exists() or not candidate_path.is_file():
            raise RuntimeError("candidate_path_unresolved")

        workflow_path = resolve_repo_path(repo_root, args.workflow_decision_path)
        if not is_within(repo_root, workflow_path):
            raise RuntimeError("workflow_path_outside_repo")
        if not workflow_path.exists() or not workflow_path.is_file():
            raise RuntimeError("workflow_path_unresolved")

        candidate = load_json(candidate_path)
        workflow = load_json(workflow_path)
        if not isinstance(candidate, dict):
            raise RuntimeError("candidate_not_object")
        if not isinstance(workflow, dict):
            raise RuntimeError("workflow_not_object")

        promotion_schema_path = resolve_repo_path(repo_root, args.promotion_schema_path)
        ok_schema, reason, details = _validate_against_schema(candidate, promotion_schema_path)
        if not ok_schema:
            raise RuntimeError(f"candidate_{reason}:{details}")

        if not _workflow_decision_ok(workflow):
            raise RuntimeError("workflow_not_pass")

        promotion_id = str(candidate.get("promotion_id") or "").strip()
        if not PROMOTION_ID_RE.fullmatch(promotion_id):
            raise RuntimeError("promotion_id_invalid")

        queue_entry_id = str(args.queue_entry_id or "").strip()
        if not QUEUE_ENTRY_RE.fullmatch(queue_entry_id):
            raise RuntimeError("queue_entry_id_invalid")

        review = candidate.get("review") if isinstance(candidate.get("review"), Mapping) else {}
        if str(review.get("state") or "") != "approved":
            raise RuntimeError("review_state_not_approved")

        insight = candidate.get("insight") if isinstance(candidate.get("insight"), Mapping) else {}
        source_lane = candidate.get("source_lane") if isinstance(candidate.get("source_lane"), Mapping) else {}
        target = candidate.get("target") if isinstance(candidate.get("target"), Mapping) else {}

        object_id = str(args.object_id or "").strip()
        if not object_id:
            slug = re.sub(r"[^a-z0-9._-]+", "_", promotion_id.lower()).strip("._-")
            object_id = f"smo_{slug}"
        if not OBJECT_ID_RE.fullmatch(object_id):
            raise RuntimeError("object_id_invalid")

        object_path = objects_dir / f"{object_id}.json"
        object_path.parent.mkdir(parents=True, exist_ok=True)

        now = now_iso()
        payload: Dict[str, Any] = {
            "schema_version": OBJECT_SCHEMA,
            "object_id": object_id,
            "created_at": now,
            "updated_at": now,
            "promotion": {
                "promotion_id": promotion_id,
                "queue_entry_id": queue_entry_id,
                "candidate_path": _safe_rel(repo_root, candidate_path),
                "candidate_sha256": f"sha256:{file_sha256(candidate_path)}",
                "workflow_decision_path": _safe_rel(repo_root, workflow_path),
                "workflow_decision_sha256": f"sha256:{file_sha256(workflow_path)}",
            },
            "memory": {
                "object_type": str(insight.get("kind") or "heuristic"),
                "title": str(insight.get("title") or f"Shared memory object {promotion_id}"),
                "statement": str(insight.get("statement") or ""),
                "canonical_state": "PROMOTED_CANONICAL",
                "target_path": str(target.get("target_path") or ""),
                "source_lane_id": str(source_lane.get("lane_id") or "unknown_lane"),
            },
            "freshness_policy": {
                "stale_after_days": int(args.stale_after_days),
                "demote_after_days": int(args.demote_after_days),
                "owner_role": str(args.owner_role),
                "owner_id": str(args.owner_id),
                "last_verified_at": now,
                "staleness_state": "current",
            },
            "conflicts": [],
            "demotions": [],
            "traceability": {
                "source_refs": list(candidate.get("source_refs") or []),
                "decision_refs": list(candidate.get("decision_refs") or []),
                "implementation_refs": list(args.implementation_ref or []),
            },
        }

        shared_memory_schema = resolve_repo_path(repo_root, args.shared_memory_schema_path)
        ok_obj, reason_obj, details_obj = _validate_against_schema(payload, shared_memory_schema)
        if not ok_obj:
            raise RuntimeError(f"shared_memory_object_{reason_obj}:{details_obj}")

        atomic_write(object_path, payload)

        registry = _load_registry(repo_root, registry_path)
        _upsert_registry_entry(
            registry,
            {
                "object_id": object_id,
                "promotion_id": promotion_id,
                "canonical_state": payload["memory"]["canonical_state"],
                "updated_at": now,
                "object_path": _safe_rel(repo_root, object_path),
            },
        )
        atomic_write(registry_path, registry)

        return 0, {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "promote",
            "ok": True,
            "object_id": object_id,
            "object_path": str(object_path),
            "registry_path": str(registry_path),
            "object": payload,
        }
    except Exception as exc:
        return 2, {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "promote",
            "ok": False,
            "error": str(exc),
        }


def cmd_conflict(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    objects_dir = resolve_repo_path(repo_root, args.objects_path)
    conflict_dir = resolve_repo_path(repo_root, args.conflicts_path)
    registry_path = resolve_repo_path(repo_root, args.registry_path)

    if not is_within(repo_root, objects_dir) or not is_within(repo_root, conflict_dir) or not is_within(repo_root, registry_path):
        return 2, {"schema": "clawd.shared_memory_fabric.result.v1", "action": "conflict", "ok": False, "error": "unsafe_state_path"}

    try:
        if args.owner_role not in ROLE_SET:
            raise RuntimeError("owner_role_invalid")

        object_path, obj = _load_object_by_id(repo_root, args.object_id, objects_dir)
        other_path, other = _load_object_by_id(repo_root, args.conflict_with_object_id, objects_dir)

        status = str(args.status or "pending")
        if status not in {"pending", "resolved_keep", "resolved_merge", "resolved_demote"}:
            raise RuntimeError("conflict_status_invalid")

        conflict_id = str(args.conflict_id or "").strip()
        if not conflict_id:
            conflict_id = f"smc_{args.object_id}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dt%H%M%S').lower()}"
        if not re.fullmatch(r"smc_[a-z0-9._-]+", conflict_id):
            raise RuntimeError("conflict_id_invalid")

        now = now_iso()
        conflict_payload: Dict[str, Any] = {
            "schema_version": CONFLICT_SCHEMA,
            "conflict_id": conflict_id,
            "created_at": now,
            "resolved_at": now if status != "pending" else None,
            "object_id": args.object_id,
            "conflict_with_object_id": args.conflict_with_object_id,
            "status": status,
            "reason": str(args.reason),
            "owner_id": str(args.owner_id),
            "owner_role": str(args.owner_role),
            "resolution_notes": args.resolution_notes if status != "pending" else None,
        }

        conflict_schema_path = resolve_repo_path(repo_root, args.conflict_schema_path)
        ok_schema, reason_schema, details_schema = _validate_against_schema(conflict_payload, conflict_schema_path)
        if not ok_schema:
            raise RuntimeError(f"conflict_record_{reason_schema}:{details_schema}")

        conflict_path = conflict_dir / f"{conflict_id}.json"
        atomic_write(conflict_path, conflict_payload)

        for row in (obj, other):
            conflict_ids = row.get("conflicts") if isinstance(row.get("conflicts"), list) else []
            if conflict_id not in conflict_ids:
                conflict_ids.append(conflict_id)
            row["conflicts"] = conflict_ids

        if status == "pending":
            obj.setdefault("memory", {})["canonical_state"] = "CONFLICTED"
        elif status == "resolved_demote":
            obj.setdefault("memory", {})["canonical_state"] = "DEMOTED_SUPERSEDED"
        else:
            obj.setdefault("memory", {})["canonical_state"] = "PROMOTED_CANONICAL"

        obj["updated_at"] = now
        other["updated_at"] = now

        shared_memory_schema = resolve_repo_path(repo_root, args.shared_memory_schema_path)
        ok_obj, reason_obj, details_obj = _validate_against_schema(obj, shared_memory_schema)
        if not ok_obj:
            raise RuntimeError(f"object_after_conflict_{reason_obj}:{details_obj}")
        ok_other, reason_other, details_other = _validate_against_schema(other, shared_memory_schema)
        if not ok_other:
            raise RuntimeError(f"other_object_after_conflict_{reason_other}:{details_other}")

        atomic_write(object_path, obj)
        atomic_write(other_path, other)

        registry = _load_registry(repo_root, registry_path)
        _upsert_registry_entry(
            registry,
            {
                "object_id": args.object_id,
                "promotion_id": ((obj.get("promotion") or {}).get("promotion_id")),
                "canonical_state": ((obj.get("memory") or {}).get("canonical_state")),
                "updated_at": now,
                "object_path": _safe_rel(repo_root, object_path),
            },
        )
        _upsert_registry_entry(
            registry,
            {
                "object_id": args.conflict_with_object_id,
                "promotion_id": ((other.get("promotion") or {}).get("promotion_id")),
                "canonical_state": ((other.get("memory") or {}).get("canonical_state")),
                "updated_at": now,
                "object_path": _safe_rel(repo_root, other_path),
            },
        )
        atomic_write(registry_path, registry)

        return 0, {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "conflict",
            "ok": True,
            "conflict": conflict_payload,
            "conflict_path": str(conflict_path),
            "object_state": (obj.get("memory") or {}).get("canonical_state"),
        }
    except Exception as exc:
        return 2, {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "conflict",
            "ok": False,
            "error": str(exc),
        }


def cmd_demote(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    objects_dir = resolve_repo_path(repo_root, args.objects_path)
    demotion_dir = resolve_repo_path(repo_root, args.demotions_path)
    registry_path = resolve_repo_path(repo_root, args.registry_path)

    if not is_within(repo_root, objects_dir) or not is_within(repo_root, demotion_dir) or not is_within(repo_root, registry_path):
        return 2, {"schema": "clawd.shared_memory_fabric.result.v1", "action": "demote", "ok": False, "error": "unsafe_state_path"}

    try:
        if args.owner_role not in ROLE_SET:
            raise RuntimeError("owner_role_invalid")

        object_path, obj = _load_object_by_id(repo_root, args.object_id, objects_dir)
        prev_state = str(((obj.get("memory") or {}).get("canonical_state")) or "")
        if prev_state not in CANONICAL_STATES:
            raise RuntimeError("object_state_invalid")

        kind = str(args.demotion_kind)
        state_map = {
            "stale": "DEMOTED_STALE",
            "superseded": "DEMOTED_SUPERSEDED",
            "invalidated": "DEMOTED_INVALIDATED",
            "manual": "DEMOTED_SUPERSEDED",
        }
        new_state = state_map.get(kind)
        if new_state is None:
            raise RuntimeError("demotion_kind_invalid")

        demotion_id = str(args.demotion_id or "").strip()
        if not demotion_id:
            demotion_id = f"smd_{args.object_id}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dt%H%M%S').lower()}"
        if not re.fullmatch(r"smd_[a-z0-9._-]+", demotion_id):
            raise RuntimeError("demotion_id_invalid")

        now = now_iso()
        demotion_payload = {
            "schema_version": DEMOTION_SCHEMA,
            "demotion_id": demotion_id,
            "created_at": now,
            "object_id": args.object_id,
            "demotion_kind": kind,
            "reason": str(args.reason),
            "owner_id": str(args.owner_id),
            "owner_role": str(args.owner_role),
            "previous_state": prev_state,
            "new_state": new_state,
        }

        demotion_schema_path = resolve_repo_path(repo_root, args.demotion_schema_path)
        ok_schema, reason_schema, details_schema = _validate_against_schema(demotion_payload, demotion_schema_path)
        if not ok_schema:
            raise RuntimeError(f"demotion_record_{reason_schema}:{details_schema}")

        demotion_path = demotion_dir / f"{demotion_id}.json"
        atomic_write(demotion_path, demotion_payload)

        obj.setdefault("memory", {})["canonical_state"] = new_state
        fp = obj.setdefault("freshness_policy", {})
        fp["staleness_state"] = "demoted"
        fp["last_verified_at"] = now
        demotions = obj.get("demotions") if isinstance(obj.get("demotions"), list) else []
        if demotion_id not in demotions:
            demotions.append(demotion_id)
        obj["demotions"] = demotions
        obj["updated_at"] = now

        shared_memory_schema = resolve_repo_path(repo_root, args.shared_memory_schema_path)
        ok_obj, reason_obj, details_obj = _validate_against_schema(obj, shared_memory_schema)
        if not ok_obj:
            raise RuntimeError(f"object_after_demotion_{reason_obj}:{details_obj}")

        atomic_write(object_path, obj)

        registry = _load_registry(repo_root, registry_path)
        _upsert_registry_entry(
            registry,
            {
                "object_id": args.object_id,
                "promotion_id": ((obj.get("promotion") or {}).get("promotion_id")),
                "canonical_state": ((obj.get("memory") or {}).get("canonical_state")),
                "updated_at": now,
                "object_path": _safe_rel(repo_root, object_path),
            },
        )
        atomic_write(registry_path, registry)

        return 0, {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "demote",
            "ok": True,
            "demotion": demotion_payload,
            "demotion_path": str(demotion_path),
            "object_path": str(object_path),
        }
    except Exception as exc:
        return 2, {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "demote",
            "ok": False,
            "error": str(exc),
        }


def cmd_status(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    objects_dir = resolve_repo_path(repo_root, args.objects_path)
    registry_path = resolve_repo_path(repo_root, args.registry_path)

    if not is_within(repo_root, objects_dir) or not is_within(repo_root, registry_path):
        return 2, {"schema": "clawd.shared_memory_fabric.result.v1", "action": "status", "ok": False, "error": "unsafe_state_path"}

    try:
        registry = _load_registry(repo_root, registry_path)
        if args.object_id:
            object_path, payload = _load_object_by_id(repo_root, args.object_id, objects_dir)
            return 0, {
                "schema": "clawd.shared_memory_fabric.result.v1",
                "action": "status",
                "ok": True,
                "object_id": args.object_id,
                "object_path": str(object_path),
                "object": payload,
                "registry_path": str(registry_path),
            }

        return 0, {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "status",
            "ok": True,
            "registry_path": str(registry_path),
            "registry": registry,
            "count": len(registry.get("objects") or []),
        }
    except Exception as exc:
        return 2, {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "status",
            "ok": False,
            "error": str(exc),
        }


def _tokenize(text: str, *, remove_stopwords: bool = False) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    if remove_stopwords:
        return [tok for tok in tokens if tok not in STOP_WORDS]
    return tokens


def _norm_text(text: str, *, remove_stopwords: bool = False) -> str:
    return " ".join(_tokenize(text, remove_stopwords=remove_stopwords))


def _build_compaction_snapshot(
    repo_root: Path,
    *,
    registry_path: Path,
    objects_dir: Path,
    compaction_strategy: str = COMPACTION_STRATEGY_LEGACY,
) -> Dict[str, Any]:
    compaction_strategy = str(compaction_strategy or COMPACTION_STRATEGY_LEGACY).strip() or COMPACTION_STRATEGY_LEGACY
    if compaction_strategy not in {COMPACTION_STRATEGY_LEGACY, COMPACTION_STRATEGY_SIGNATURE}:
        raise RuntimeError(f"invalid_compaction_strategy:{compaction_strategy}")

    registry = _load_registry(repo_root, registry_path)
    groups: Dict[str, Dict[str, Any]] = {}
    source_object_count = 0
    source_bytes = 0

    for row in registry.get("objects") or []:
        if not isinstance(row, dict):
            continue
        object_id = str(row.get("object_id") or "").strip()
        if not object_id:
            continue

        object_path, obj = _load_object_by_id(repo_root, object_id, objects_dir)
        source_object_count += 1
        source_bytes += max(0, int(object_path.stat().st_size))

        memory = obj.get("memory") if isinstance(obj.get("memory"), Mapping) else {}
        object_type = str(memory.get("object_type") or "unknown").strip() or "unknown"
        title = str(memory.get("title") or "").strip()
        statement = str(memory.get("statement") or "").strip()
        canonical_state = str(memory.get("canonical_state") or "UNKNOWN").strip() or "UNKNOWN"

        type_seed = _norm_text(object_type)
        statement_norm = _norm_text(statement)
        title_norm = _norm_text(title)

        if compaction_strategy == COMPACTION_STRATEGY_SIGNATURE:
            signature_tokens = sorted(
                set(
                    _tokenize(
                        " ".join([type_seed, statement_norm]),
                        remove_stopwords=True,
                    )
                )
            )
            signature_seed = " ".join(signature_tokens) if signature_tokens else ""
            key_body = signature_seed or statement_norm or title_norm
            if not key_body:
                key_body = f"object_id:{object_id.lower()}"
            key_seed = f"{type_seed}|{key_body}"
        else:
            key_seed = "|".join([
                type_seed,
                statement_norm if statement_norm else title_norm,
            ])
            if not key_seed.strip("|"):
                key_seed = f"object_id:{object_id.lower()}"

        assertion_id = f"sma_{hashlib.sha256(key_seed.encode('utf-8')).hexdigest()[:20]}"
        grp = groups.get(assertion_id)
        if grp is None:
            grp = {
                "assertion_id": assertion_id,
                "object_type": object_type,
                "title": title or None,
                "statement": statement or None,
                "token_index": sorted(set(_tokenize(" ".join([object_type, title, statement])))),
                "source_object_ids": [],
                "canonical_state_counts": {},
                "relationship_refs": {
                    "conflicts": [],
                    "demotions": [],
                },
            }
            groups[assertion_id] = grp

        source_ids = grp["source_object_ids"]
        if object_id not in source_ids:
            source_ids.append(object_id)

        state_counts = grp["canonical_state_counts"]
        state_counts[canonical_state] = int(state_counts.get(canonical_state) or 0) + 1

        conflicts = obj.get("conflicts") if isinstance(obj.get("conflicts"), list) else []
        for conflict_id in conflicts:
            txt = str(conflict_id or "").strip()
            if txt and txt not in grp["relationship_refs"]["conflicts"]:
                grp["relationship_refs"]["conflicts"].append(txt)

        demotions = obj.get("demotions") if isinstance(obj.get("demotions"), list) else []
        for demotion_id in demotions:
            txt = str(demotion_id or "").strip()
            if txt and txt not in grp["relationship_refs"]["demotions"]:
                grp["relationship_refs"]["demotions"].append(txt)

    entries = sorted(groups.values(), key=lambda row: str(row.get("assertion_id") or ""))
    compacted_bytes = len(json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    reduction_pct = 0.0
    if source_bytes > 0:
        reduction_pct = round(max(0.0, (1.0 - (compacted_bytes / source_bytes)) * 100.0), 4)

    return {
        "schema": "clawd.shared_memory_compaction.snapshot.v1",
        "generated_at": now_iso(),
        "registry_path": _safe_rel(repo_root, registry_path),
        "compaction_strategy": compaction_strategy,
        "source_object_count": source_object_count,
        "compacted_assertion_count": len(entries),
        "source_bytes": source_bytes,
        "compacted_bytes": compacted_bytes,
        "reduction_pct": reduction_pct,
        "entries": entries,
    }


def _build_token_idf(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        token_index = entry.get("token_index") if isinstance(entry.get("token_index"), list) else []
        token_set = {str(tok).strip().lower() for tok in token_index if str(tok).strip()}
        for token in token_set:
            df[token] = int(df.get(token, 0)) + 1

    doc_count = max(1, len(entries))
    return {token: round(math.log((doc_count + 1) / (df_count + 1)) + 1.0, 6) for token, df_count in df.items()}


def _score_entry_for_query(
    entry: Dict[str, Any],
    query_tokens: List[str],
    *,
    strategy: str,
    token_idf: Optional[Dict[str, float]] = None,
) -> float:
    token_index = entry.get("token_index") if isinstance(entry.get("token_index"), list) else []
    token_set = {str(tok).strip().lower() for tok in token_index if str(tok).strip()}
    if not token_set:
        token_set = set(_tokenize(f"{entry.get('title') or ''} {entry.get('statement') or ''}"))

    query_set = set(query_tokens)
    overlap = query_set & token_set
    if not overlap:
        return 0.0

    denominator = max(1.0, float(len(query_set)))
    if strategy == RETRIEVAL_STRATEGY_TFIDF_HYBRID:
        tfidf = token_idf or {}
        weighted_overlap = sum(float(tfidf.get(token, 1.0)) for token in overlap)
        score = weighted_overlap / denominator
    else:
        score = len(overlap) / denominator

    combined_text = _norm_text(f"{entry.get('title') or ''} {entry.get('statement') or ''}")
    query_norm = _norm_text(" ".join(query_tokens))
    if query_norm and query_norm in combined_text:
        score = score + 0.15
    return score


def _retrieve_from_entries(
    entries: List[Dict[str, Any]],
    query: str,
    *,
    top_k: int,
    strategy: str = RETRIEVAL_STRATEGY_TOKEN_OVERLAP,
    token_idf: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    query_tokens = sorted(set(_tokenize(query)))
    if not query_tokens:
        return {"matches": [], "matched_object_ids": [], "query_tokens": []}

    strategy = str(strategy or RETRIEVAL_STRATEGY_TOKEN_OVERLAP).strip() or RETRIEVAL_STRATEGY_TOKEN_OVERLAP
    if strategy not in {RETRIEVAL_STRATEGY_TOKEN_OVERLAP, RETRIEVAL_STRATEGY_TFIDF_HYBRID}:
        raise RuntimeError(f"invalid_retrieval_strategy:{strategy}")

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        score = _score_entry_for_query(entry, query_tokens, strategy=strategy, token_idf=token_idf)
        if score <= 0:
            continue
        scored.append((score, entry))

    scored.sort(
        key=lambda row: (row[0], len((row[1] or {}).get("source_object_ids") or []), str((row[1] or {}).get("assertion_id") or "")),
        reverse=True,
    )
    chosen = [row[1] for row in scored[: max(1, int(top_k or 5))]]

    matched_object_ids: List[str] = []
    seen = set()
    for entry in chosen:
        source_ids = entry.get("source_object_ids") if isinstance(entry.get("source_object_ids"), list) else []
        for raw_id in source_ids:
            txt = str(raw_id or "").strip()
            if txt and txt not in seen:
                seen.add(txt)
                matched_object_ids.append(txt)

    matches: List[Dict[str, Any]] = []
    for entry in chosen:
        entry_query_tokens = query_tokens
        score = _score_entry_for_query(entry, entry_query_tokens, strategy=strategy, token_idf=token_idf)
        matches.append(
            {
                "assertion_id": entry.get("assertion_id"),
                "score": round(score, 6),
                "title": entry.get("title"),
                "statement": entry.get("statement"),
                "source_object_ids": entry.get("source_object_ids") if isinstance(entry.get("source_object_ids"), list) else [],
            }
        )

    return {
        "matches": matches,
        "matched_object_ids": matched_object_ids,
        "query_tokens": query_tokens,
    }


def cmd_compact(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    objects_dir = resolve_repo_path(repo_root, args.objects_path)
    registry_path = resolve_repo_path(repo_root, args.registry_path)
    output_path = resolve_repo_path(repo_root, args.output_path)

    if not is_within(repo_root, objects_dir) or not is_within(repo_root, registry_path) or not is_within(repo_root, output_path):
        return 2, {"schema": "clawd.shared_memory_fabric.result.v1", "action": "compact", "ok": False, "error": "unsafe_state_path"}

    try:
        snapshot = _build_compaction_snapshot(
            repo_root,
            registry_path=registry_path,
            objects_dir=objects_dir,
            compaction_strategy=args.compaction_strategy,
        )
        atomic_write(output_path, snapshot)
        return 0, {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "compact",
            "ok": True,
            "output_path": str(output_path),
            "compaction_strategy": snapshot.get("compaction_strategy"),
            "source_object_count": snapshot.get("source_object_count"),
            "compacted_assertion_count": snapshot.get("compacted_assertion_count"),
            "source_bytes": snapshot.get("source_bytes"),
            "compacted_bytes": snapshot.get("compacted_bytes"),
            "reduction_pct": snapshot.get("reduction_pct"),
        }
    except Exception as exc:
        return 2, {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "compact",
            "ok": False,
            "error": str(exc),
        }


def cmd_retrieve(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    objects_dir = resolve_repo_path(repo_root, args.objects_path)
    registry_path = resolve_repo_path(repo_root, args.registry_path)

    if not is_within(repo_root, objects_dir) or not is_within(repo_root, registry_path):
        return 2, {"schema": "clawd.shared_memory_fabric.result.v1", "action": "retrieve", "ok": False, "error": "unsafe_state_path"}

    query = str(args.query or "").strip()
    if not query:
        return 2, {"schema": "clawd.shared_memory_fabric.result.v1", "action": "retrieve", "ok": False, "error": "query_empty"}

    try:
        retrieval_strategy = str(args.retrieval_strategy or RETRIEVAL_STRATEGY_TOKEN_OVERLAP).strip() or RETRIEVAL_STRATEGY_TOKEN_OVERLAP
        snapshot = _build_compaction_snapshot(
            repo_root,
            registry_path=registry_path,
            objects_dir=objects_dir,
            compaction_strategy=args.compaction_strategy,
        )
        entries = snapshot.get("entries") if isinstance(snapshot.get("entries"), list) else []

        token_idf = None
        if retrieval_strategy == RETRIEVAL_STRATEGY_TFIDF_HYBRID:
            token_idf = _build_token_idf(entries)

        retrieval = _retrieve_from_entries(
            entries,
            query,
            top_k=max(1, int(args.top_k or 5)),
            strategy=retrieval_strategy,
            token_idf=token_idf,
        )
        return 0, {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "retrieve",
            "ok": True,
            "query": query,
            "retrieval_strategy": retrieval_strategy,
            "top_k": max(1, int(args.top_k or 5)),
            "query_tokens": retrieval.get("query_tokens"),
            "matches": retrieval.get("matches"),
            "matched_object_ids": retrieval.get("matched_object_ids"),
            "compaction": {
                "compaction_strategy": snapshot.get("compaction_strategy"),
                "source_object_count": snapshot.get("source_object_count"),
                "compacted_assertion_count": snapshot.get("compacted_assertion_count"),
                "reduction_pct": snapshot.get("reduction_pct"),
            },
        }
    except Exception as exc:
        return 2, {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "retrieve",
            "ok": False,
            "error": str(exc),
        }


def cmd_benchmark(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    objects_dir = resolve_repo_path(repo_root, args.objects_path)
    registry_path = resolve_repo_path(repo_root, args.registry_path)

    if not is_within(repo_root, objects_dir) or not is_within(repo_root, registry_path):
        return 2, {"schema": "clawd.shared_memory_fabric.result.v1", "action": "benchmark", "ok": False, "error": "unsafe_state_path"}

    try:
        golden_path = resolve_repo_path(repo_root, args.golden_queries)
        if not is_within(repo_root, golden_path):
            raise RuntimeError("golden_queries_outside_repo")
        if not golden_path.exists() or not golden_path.is_file():
            raise RuntimeError("golden_queries_missing")

        golden_payload = load_json(golden_path)
        if isinstance(golden_payload, dict):
            query_rows = golden_payload.get("queries") if isinstance(golden_payload.get("queries"), list) else []
        elif isinstance(golden_payload, list):
            query_rows = golden_payload
        else:
            query_rows = []

        snapshot = _build_compaction_snapshot(
            repo_root,
            registry_path=registry_path,
            objects_dir=objects_dir,
            compaction_strategy=args.compaction_strategy,
        )
        entries = snapshot.get("entries") if isinstance(snapshot.get("entries"), list) else []

        retrieval_strategy = str(args.retrieval_strategy or RETRIEVAL_STRATEGY_TOKEN_OVERLAP).strip() or RETRIEVAL_STRATEGY_TOKEN_OVERLAP
        token_idf = None
        if retrieval_strategy == RETRIEVAL_STRATEGY_TFIDF_HYBRID:
            token_idf = _build_token_idf(entries)

        expected_total = 0
        expected_matched = 0
        per_query: List[Dict[str, Any]] = []
        top_k = max(1, int(args.top_k or 5))

        for row in query_rows:
            if not isinstance(row, dict):
                continue
            query = str(row.get("query") or "").strip()
            expected_ids = [
                str(raw).strip()
                for raw in (row.get("expected_object_ids") if isinstance(row.get("expected_object_ids"), list) else [])
                if str(raw).strip()
            ]
            if not query or not expected_ids:
                continue

            retrieval = _retrieve_from_entries(
                entries,
                query,
                top_k=top_k,
                strategy=retrieval_strategy,
                token_idf=token_idf,
            )
            matched = set(retrieval.get("matched_object_ids") or [])
            hit_ids = [obj_id for obj_id in expected_ids if obj_id in matched]

            expected_total += len(expected_ids)
            expected_matched += len(hit_ids)
            per_query.append(
                {
                    "query": query,
                    "expected_object_ids": expected_ids,
                    "matched_object_ids": retrieval.get("matched_object_ids") or [],
                    "hit_object_ids": hit_ids,
                    "recall": round((len(hit_ids) / len(expected_ids)) if expected_ids else 0.0, 6),
                }
            )

        recall_pct = round(((expected_matched / expected_total) * 100.0) if expected_total > 0 else 0.0, 4)
        reduction_pct = float(snapshot.get("reduction_pct") or 0.0)
        min_reduction_pct = float(args.min_reduction_pct)
        min_recall_pct = float(args.min_recall_pct)
        pass_result = bool(reduction_pct >= min_reduction_pct and recall_pct >= min_recall_pct)

        payload = {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "benchmark",
            "ok": pass_result,
            "pass": pass_result,
            "thresholds": {
                "min_reduction_pct": min_reduction_pct,
                "min_recall_pct": min_recall_pct,
            },
            "compaction": {
                "compaction_strategy": snapshot.get("compaction_strategy"),
                "source_object_count": snapshot.get("source_object_count"),
                "compacted_assertion_count": snapshot.get("compacted_assertion_count"),
                "source_bytes": snapshot.get("source_bytes"),
                "compacted_bytes": snapshot.get("compacted_bytes"),
                "reduction_pct": reduction_pct,
            },
            "retrieval": {
                "strategy": retrieval_strategy,
                "top_k": top_k,
                "query_count": len(per_query),
                "expected_total": expected_total,
                "matched_total": expected_matched,
                "recall_pct": recall_pct,
                "queries": per_query,
            },
        }

        if pass_result:
            return 0, payload
        return 2, payload
    except Exception as exc:
        return 2, {
            "schema": "clawd.shared_memory_fabric.result.v1",
            "action": "benchmark",
            "ok": False,
            "error": str(exc),
        }


def _resolve_allowed_callsites() -> List[str]:
    allowed = set(DEFAULT_ALLOWED_MUTATION_CALLSITES)
    raw = str(os.environ.get("OPENCLAW_SHARED_MEMORY_FABRIC_ALLOWED_CALLSITES") or "").strip()
    if raw:
        for token in raw.split(","):
            value = token.strip()
            if value:
                allowed.add(value)
    return sorted(allowed)


def _enforce_wrapper_only_contract(command: str) -> Optional[Dict[str, Any]]:
    if command not in {"promote", "conflict", "demote"}:
        return None

    internal_mutation = str(os.environ.get("OPENCLAW_INTERNAL_MUTATION") or "").strip()
    callsite = str(os.environ.get("OPENCLAW_INTERNAL_MUTATION_CALLSITE") or "").strip()
    allowed_callsites = _resolve_allowed_callsites()

    if internal_mutation != "1":
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_env_missing",
            "required_env": ["OPENCLAW_INTERNAL_MUTATION=1", "OPENCLAW_INTERNAL_MUTATION_CALLSITE=<allowlisted>"],
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh shared-memory <promote|conflict|demote> ... --json",
        }

    if not callsite:
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_callsite_missing",
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh shared-memory <promote|conflict|demote> ... --json",
        }

    if callsite not in allowed_callsites:
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_callsite_not_allowlisted",
            "callsite": callsite,
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh shared-memory <promote|conflict|demote> ... --json",
        }

    return None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Shared memory fabric lifecycle runtime (v1)")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--objects-path", default=str(DEFAULT_OBJECT_PATH), help="Shared memory object directory")
    ap.add_argument("--conflicts-path", default=str(DEFAULT_CONFLICT_PATH), help="Conflict record directory")
    ap.add_argument("--demotions-path", default=str(DEFAULT_DEMOTION_PATH), help="Demotion record directory")
    ap.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH), help="Shared memory registry path")
    ap.add_argument("--promotion-schema-path", default=str(DEFAULT_PROMOTION_SCHEMA_PATH), help="Promotion candidate schema path")
    ap.add_argument("--shared-memory-schema-path", default=str(SHARED_MEMORY_SCHEMA_PATH), help="Shared memory object schema path")
    ap.add_argument("--conflict-schema-path", default=str(CONFLICT_SCHEMA_PATH), help="Conflict record schema path")
    ap.add_argument("--demotion-schema-path", default=str(DEMOTION_SCHEMA_PATH), help="Demotion record schema path")
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON")

    sub = ap.add_subparsers(dest="command", required=True)

    p_promote = sub.add_parser("promote", help="materialize canonical shared-memory object")
    p_promote.add_argument("--candidate", required=True, help="Promotion candidate path")
    p_promote.add_argument("--queue-entry-id", required=True, help="Queue entry id")
    p_promote.add_argument("--workflow-decision-path", required=True, help="Workflow decision JSON path")
    p_promote.add_argument("--object-id", default="", help="Optional explicit shared-memory object id")
    p_promote.add_argument("--owner-id", default="shared_memory_fabric")
    p_promote.add_argument("--owner-role", default="LIBRARIAN", choices=sorted(ROLE_SET))
    p_promote.add_argument("--stale-after-days", type=int, default=14)
    p_promote.add_argument("--demote-after-days", type=int, default=30)
    p_promote.add_argument("--implementation-ref", action="append", default=[])
    p_promote.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_conflict = sub.add_parser("conflict", help="record contradiction/conflict-set row")
    p_conflict.add_argument("--object-id", required=True)
    p_conflict.add_argument("--conflict-with-object-id", required=True)
    p_conflict.add_argument("--conflict-id", default="")
    p_conflict.add_argument("--status", default="pending", choices=["pending", "resolved_keep", "resolved_merge", "resolved_demote"])
    p_conflict.add_argument("--reason", required=True)
    p_conflict.add_argument("--owner-id", required=True)
    p_conflict.add_argument("--owner-role", required=True, choices=sorted(ROLE_SET))
    p_conflict.add_argument("--resolution-notes", default="")
    p_conflict.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_demote = sub.add_parser("demote", help="record demotion + mutate object state")
    p_demote.add_argument("--object-id", required=True)
    p_demote.add_argument("--demotion-id", default="")
    p_demote.add_argument("--demotion-kind", required=True, choices=["stale", "superseded", "invalidated", "manual"])
    p_demote.add_argument("--reason", required=True)
    p_demote.add_argument("--owner-id", required=True)
    p_demote.add_argument("--owner-role", required=True, choices=sorted(ROLE_SET))
    p_demote.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_status = sub.add_parser("status", help="inspect shared-memory registry/object")
    p_status.add_argument("--object-id", default="")
    p_status.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_compact = sub.add_parser("compact", help="build compacted shared-memory assertion snapshot")
    p_compact.add_argument("--output-path", default=str(DEFAULT_COMPACTION_PATH), help="Compaction snapshot output path")
    p_compact.add_argument(
        "--compaction-strategy",
        default=COMPACTION_STRATEGY_LEGACY,
        choices=[COMPACTION_STRATEGY_LEGACY, COMPACTION_STRATEGY_SIGNATURE],
        help="Compaction strategy to use",
    )
    p_compact.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_retrieve = sub.add_parser("retrieve", help="retrieve compacted assertions")
    p_retrieve.add_argument("--query", required=True, help="Query string")
    p_retrieve.add_argument("--top-k", type=int, default=5, help="Top K compacted assertions")
    p_retrieve.add_argument(
        "--compaction-strategy",
        default=COMPACTION_STRATEGY_LEGACY,
        choices=[COMPACTION_STRATEGY_LEGACY, COMPACTION_STRATEGY_SIGNATURE],
        help="Compaction strategy used during retrieval",
    )
    p_retrieve.add_argument(
        "--retrieval-strategy",
        default=RETRIEVAL_STRATEGY_TOKEN_OVERLAP,
        choices=[RETRIEVAL_STRATEGY_TOKEN_OVERLAP, RETRIEVAL_STRATEGY_TFIDF_HYBRID],
        help="Retrieval scoring strategy",
    )
    p_retrieve.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_benchmark = sub.add_parser("benchmark", help="benchmark compaction reduction and retrieval recall")
    p_benchmark.add_argument("--golden-queries", required=True, help="Path to golden query JSON")
    p_benchmark.add_argument("--top-k", type=int, default=5, help="Top K compacted assertions per query")
    p_benchmark.add_argument(
        "--compaction-strategy",
        default=COMPACTION_STRATEGY_LEGACY,
        choices=[COMPACTION_STRATEGY_LEGACY, COMPACTION_STRATEGY_SIGNATURE],
        help="Compaction strategy used by benchmark",
    )
    p_benchmark.add_argument(
        "--retrieval-strategy",
        default=RETRIEVAL_STRATEGY_TOKEN_OVERLAP,
        choices=[RETRIEVAL_STRATEGY_TOKEN_OVERLAP, RETRIEVAL_STRATEGY_TFIDF_HYBRID],
        help="Retrieval strategy used by benchmark",
    )
    p_benchmark.add_argument("--min-reduction-pct", type=float, default=50.0)
    p_benchmark.add_argument("--min-recall-pct", type=float, default=95.0)
    p_benchmark.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    wrapper_guard_error = _enforce_wrapper_only_contract(str(args.command or ""))
    if wrapper_guard_error is not None:
        print(json.dumps(wrapper_guard_error, ensure_ascii=False, indent=2 if args.json else None))
        return 2

    if args.command == "promote":
        rc, payload = cmd_promote(args)
    elif args.command == "conflict":
        rc, payload = cmd_conflict(args)
    elif args.command == "demote":
        rc, payload = cmd_demote(args)
    elif args.command == "status":
        rc, payload = cmd_status(args)
    elif args.command == "compact":
        rc, payload = cmd_compact(args)
    elif args.command == "retrieve":
        rc, payload = cmd_retrieve(args)
    elif args.command == "benchmark":
        rc, payload = cmd_benchmark(args)
    else:
        rc, payload = 2, {"schema": "clawd.shared_memory_fabric.result.v1", "ok": False, "error": f"unknown_command:{args.command}"}

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(stable_json(payload if isinstance(payload, dict) else {"payload": payload}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
