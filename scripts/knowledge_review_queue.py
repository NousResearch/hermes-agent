#!/usr/bin/env python3
"""Bounded Knowledge Review / Approval / Promotion Queue helper (v1, legacy compatibility).

Canonical queue runtime path is `scripts/knowledge_promotion_queue.py`.

This helper intentionally stays local + deterministic:
- queue snapshot file (`queue.json`) is mutable state,
- decision log (`decisions.jsonl`) is append-only,
- transitions are fail-closed and role-gated,
- knowledge objects must validate against existing Wave-4 schemas.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover (environment wiring)
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_QUEUE_ITEM_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "knowledge_review_queue_item.schema.json"
DEFAULT_QUEUE_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "knowledge_review_queue" / "queue.json"
DEFAULT_DECISION_LOG = DEFAULT_REPO_ROOT / "state" / "continuity" / "knowledge_review_queue" / "decisions.jsonl"
WRAPPER_REQUIRED_SCHEMA = "clawd.knowledge_review_queue.wrapper_contract.v1"
DEFAULT_ALLOWED_MUTATION_CALLSITES = {
    "continuity.sh:knowledge-queue",
}

OBJECT_CLASS_SCHEMA_MAP = {
    "promotion_candidate": "docs/ops/schemas/promotion_candidate.schema.json",
    "doctrine_object": "docs/ops/schemas/doctrine_object.schema.json",
}

REVIEWER_ROLES = {"PLANNER", "EXECUTOR", "VALIDATOR", "RESEARCHER", "SRE", "LIBRARIAN"}
APPROVER_ROLES = {"VALIDATOR", "LIBRARIAN"}
SAFETY_ROLES = {"VALIDATOR", "LIBRARIAN", "SRE"}
PROMOTABLE_TRUST_TIERS = {"t2_verified", "t3_canonical"}

ALLOWED_TRANSITIONS = {
    "PENDING_REVIEW": {"UNDER_REVIEW", "REJECTED", "BLOCKED", "EXPIRED"},
    "UNDER_REVIEW": {"CHANGES_REQUESTED", "APPROVAL_PENDING", "REJECTED", "BLOCKED"},
    "CHANGES_REQUESTED": {"PENDING_REVIEW", "REJECTED", "BLOCKED"},
    "REJECTED": set(),
    "APPROVAL_PENDING": {"APPROVED", "REJECTED", "BLOCKED"},
    "APPROVED": {"PROMOTION_READY", "BLOCKED", "EXPIRED"},
    "PROMOTION_READY": {"PROMOTED", "BLOCKED"},
    "PROMOTED": set(),
    "BLOCKED": {"PENDING_REVIEW", "EXPIRED"},
    "EXPIRED": set(),
}

SHA256_RE = re.compile(r"^(sha256:)?[a-f0-9]{64}$")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_sha256(raw: str) -> str:
    text = (raw or "").strip().lower()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _safe_repo_path(repo_root: Path, raw_path: Path | str) -> Tuple[bool, Path, Optional[str]]:
    try:
        if isinstance(raw_path, Path):
            path = raw_path.expanduser()
            if not path.is_absolute():
                path = (repo_root / path).resolve()
            else:
                path = path.resolve()
        else:
            path = resolve_repo_path(repo_root, str(raw_path))
    except Exception as exc:
        return False, repo_root, f"path_resolve_failed:{exc}"

    if not is_within(repo_root, path):
        return False, path, "unsafe_path"

    return True, path, None


def validate_jsonschema(payload: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "gate_unavailable", {"error": "jsonschema_validator_unavailable"}

    if not schema_path.exists() or not schema_path.is_file():
        return False, "gate_unavailable", {"error": "schema_missing", "schema_path": str(schema_path)}

    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        return False, "gate_unavailable", {"error": "schema_unreadable", "detail": str(exc)}

    if not isinstance(schema_doc, dict):
        return False, "gate_unavailable", {"error": "schema_not_object"}

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return True, None, {"schema_path": str(schema_path)}

    err = errors[0]
    return (
        False,
        "schema_invalid",
        {
            "error": "schema_validation_failed",
            "data_path": "$" if not err.absolute_path else "$/" + "/".join(str(p) for p in err.absolute_path),
            "schema_path": "$" if not err.absolute_schema_path else "$/" + "/".join(str(p) for p in err.absolute_schema_path),
            "message": str(err.message),
        },
    )


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(path.parent)) as tmp:
        tmp.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def append_decision_record(
    *,
    decision_log_path: Optional[Path],
    repo_root: Path,
    decision_row: Dict[str, Any],
) -> Dict[str, Any]:
    if decision_log_path is None:
        return {"enabled": False, "appended": False, "reason": "disabled"}

    ok, resolved, err = _safe_repo_path(repo_root, decision_log_path)
    if not ok:
        return {
            "enabled": True,
            "appended": False,
            "reason": err or "unsafe_path",
            "path": str(resolved),
        }

    try:
        if resolved.exists() and not resolved.is_file():
            return {
                "enabled": True,
                "appended": False,
                "reason": "path_not_file",
                "path": str(resolved),
            }

        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("a", encoding="utf-8") as fh:
            fh.write(stable_json_dumps(decision_row) + "\n")

        return {"enabled": True, "appended": True, "path": str(resolved)}
    except Exception as exc:
        return {
            "enabled": True,
            "appended": False,
            "reason": "append_failed",
            "path": str(resolved),
            "error": str(exc),
        }


def load_or_init_queue(*, repo_root: Path, queue_path: Path, max_items: int) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    ok, resolved, err = _safe_repo_path(repo_root, queue_path)
    if not ok:
        return None, "unsafe_path", {"path": str(resolved), "error": err}

    if not resolved.exists():
        queue_doc = {
            "schema": "clawd.knowledge_review_queue.v1",
            "updated_at": now_iso(),
            "max_items": int(max_items),
            "items": [],
        }
        return queue_doc, None, {"queue_path": str(resolved), "created": True}

    if not resolved.is_file():
        return None, "queue_path_not_file", {"queue_path": str(resolved)}

    try:
        doc = load_json_file(resolved)
    except Exception as exc:
        return None, "queue_unreadable", {"queue_path": str(resolved), "error": str(exc)}

    if not isinstance(doc, dict):
        return None, "queue_invalid", {"queue_path": str(resolved), "error": "top_level_not_object"}

    if not isinstance(doc.get("items"), list):
        return None, "queue_invalid", {"queue_path": str(resolved), "error": "items_not_array"}

    existing_max = doc.get("max_items")
    if not isinstance(existing_max, int) or existing_max < 1:
        doc["max_items"] = int(max_items)

    if not isinstance(doc.get("schema"), str):
        doc["schema"] = "clawd.knowledge_review_queue.v1"

    return doc, None, {"queue_path": str(resolved), "created": False}


def persist_queue(*, repo_root: Path, queue_path: Path, queue_doc: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    ok, resolved, err = _safe_repo_path(repo_root, queue_path)
    if not ok:
        return False, {"error": err or "unsafe_path", "queue_path": str(resolved)}

    try:
        atomic_write_json(resolved, queue_doc)
    except Exception as exc:
        return False, {"error": "queue_write_failed", "queue_path": str(resolved), "detail": str(exc)}

    return True, {"queue_path": str(resolved)}


def _validate_knowledge_object(item: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    knowledge_object = item.get("knowledge_object")
    if not isinstance(knowledge_object, dict):
        return False, "knowledge_object_unresolved", {"error": "knowledge_object_missing"}

    object_class = knowledge_object.get("object_class")
    object_path = knowledge_object.get("object_path")
    schema_ref = knowledge_object.get("schema_ref")
    object_hash = knowledge_object.get("object_hash")

    if not isinstance(object_class, str) or object_class not in OBJECT_CLASS_SCHEMA_MAP:
        return False, "knowledge_object_unresolved", {"error": "unknown_object_class", "object_class": object_class}

    expected_schema_ref = OBJECT_CLASS_SCHEMA_MAP[object_class]
    if schema_ref != expected_schema_ref:
        return (
            False,
            "knowledge_object_unresolved",
            {
                "error": "schema_ref_mismatch",
                "object_class": object_class,
                "schema_ref": schema_ref,
                "expected_schema_ref": expected_schema_ref,
            },
        )

    if not isinstance(object_path, str) or not object_path.strip():
        return False, "knowledge_object_unresolved", {"error": "object_path_missing"}

    ok, resolved_object_path, err = _safe_repo_path(repo_root, object_path)
    if not ok:
        return False, "knowledge_object_unresolved", {"error": err or "unsafe_path", "object_path": object_path}

    if not resolved_object_path.exists() or not resolved_object_path.is_file():
        return False, "knowledge_object_unresolved", {"error": "object_path_missing", "object_path": object_path}

    if not isinstance(object_hash, str) or not SHA256_RE.fullmatch(object_hash.strip().lower()):
        return False, "knowledge_object_unresolved", {"error": "object_hash_missing_or_invalid"}

    declared_hash = normalize_sha256(object_hash)
    actual_hash = file_sha256(resolved_object_path)
    if declared_hash != actual_hash:
        return (
            False,
            "knowledge_object_hash_mismatch",
            {
                "object_path": object_path,
                "declared_hash": declared_hash,
                "actual_hash": actual_hash,
            },
        )

    ok_schema_path, resolved_schema_path, schema_err = _safe_repo_path(repo_root, schema_ref)
    if not ok_schema_path:
        return False, "knowledge_object_unresolved", {"error": schema_err or "unsafe_path", "schema_ref": schema_ref}

    try:
        object_doc = load_json_file(resolved_object_path)
    except Exception as exc:
        return False, "knowledge_object_schema_invalid", {"error": "object_json_unreadable", "detail": str(exc)}

    is_valid, reason, schema_details = validate_jsonschema(object_doc, resolved_schema_path)
    if not is_valid:
        return False, "knowledge_object_schema_invalid", {"reason": reason, "details": schema_details}

    if object_class == "promotion_candidate":
        queue_promotion_id = knowledge_object.get("promotion_id")
        object_promotion_id = object_doc.get("promotion_id") if isinstance(object_doc, dict) else None
        if not isinstance(object_promotion_id, str) or not object_promotion_id.strip():
            return False, "promotion_id_missing", {"error": "promotion_id_missing_in_object"}
        if queue_promotion_id != object_promotion_id:
            return (
                False,
                "knowledge_object_unresolved",
                {
                    "error": "promotion_id_mismatch",
                    "queue_promotion_id": queue_promotion_id,
                    "object_promotion_id": object_promotion_id,
                },
            )

    if object_class == "doctrine_object":
        queue_doctrine_id = knowledge_object.get("doctrine_id")
        object_doctrine_id = object_doc.get("doctrine_id") if isinstance(object_doc, dict) else None
        if not isinstance(object_doctrine_id, str) or not object_doctrine_id.strip():
            return False, "knowledge_object_unresolved", {"error": "doctrine_id_missing_in_object"}
        if queue_doctrine_id != object_doctrine_id:
            return (
                False,
                "knowledge_object_unresolved",
                {
                    "error": "doctrine_id_mismatch",
                    "queue_doctrine_id": queue_doctrine_id,
                    "object_doctrine_id": object_doctrine_id,
                },
            )

    return (
        True,
        None,
        {
            "object_class": object_class,
            "object_path": object_path,
            "object_hash": actual_hash,
            "schema_ref": schema_ref,
        },
    )


def evaluate_evidence_requirements(item: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    req = item.get("evidence_requirements")
    evidence_refs = item.get("evidence_refs")
    decision_refs = item.get("decision_refs")

    issues: List[Dict[str, Any]] = []

    if not isinstance(req, dict):
        return False, {"issues": [{"reason": "evidence_requirements_missing"}]}

    if not isinstance(evidence_refs, list):
        return False, {"issues": [{"reason": "evidence_refs_missing"}]}

    if not isinstance(decision_refs, list):
        decision_refs = []

    min_source_refs = req.get("min_source_refs")
    require_hashes = bool(req.get("require_provenance_hashes"))
    required_gate_decisions = req.get("required_gate_decisions")

    if not isinstance(min_source_refs, int) or min_source_refs < 1:
        issues.append({"reason": "min_source_refs_invalid", "value": min_source_refs})
        min_source_refs = 1

    source_count = 0
    for idx, ref in enumerate(evidence_refs):
        if not isinstance(ref, dict):
            issues.append({"reason": "evidence_ref_not_object", "index": idx})
            continue
        kind = ref.get("kind")
        if kind == "source":
            source_count += 1

        if require_hashes:
            content_hash = ref.get("content_hash")
            if not isinstance(content_hash, str) or not SHA256_RE.fullmatch(content_hash.strip().lower()):
                issues.append({"reason": "evidence_hash_missing_or_invalid", "index": idx})

    if source_count < min_source_refs:
        issues.append(
            {
                "reason": "insufficient_source_evidence",
                "source_count": source_count,
                "required": min_source_refs,
            }
        )

    if not isinstance(required_gate_decisions, list):
        issues.append({"reason": "required_gate_decisions_invalid"})
        required_gate_decisions = []

    decision_ref_set = {str(row) for row in decision_refs if isinstance(row, str)}
    missing_gate_refs = [row for row in required_gate_decisions if isinstance(row, str) and row not in decision_ref_set]
    if missing_gate_refs:
        issues.append({"reason": "required_gate_decisions_missing", "missing": missing_gate_refs})

    ok = not issues
    return ok, {
        "ok": ok,
        "source_count": source_count,
        "required_source_count": min_source_refs,
        "missing_gate_refs": missing_gate_refs,
        "issues": issues,
    }


def _role_allowed_for_transition(to_state: str, actor_role: str) -> bool:
    if to_state in {"UNDER_REVIEW", "CHANGES_REQUESTED", "REJECTED", "APPROVAL_PENDING"}:
        return actor_role in REVIEWER_ROLES
    if to_state in {"APPROVED", "PROMOTION_READY", "PROMOTED"}:
        return actor_role in APPROVER_ROLES
    if to_state in {"BLOCKED", "EXPIRED"}:
        return actor_role in SAFETY_ROLES
    if to_state == "PENDING_REVIEW":
        return actor_role in REVIEWER_ROLES | APPROVER_ROLES
    return False


def _state_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in items:
        state = row.get("queue_state")
        if not isinstance(state, str):
            continue
        counts[state] = counts.get(state, 0) + 1
    return counts


def _resolve_allowed_callsites() -> List[str]:
    allowed = set(DEFAULT_ALLOWED_MUTATION_CALLSITES)
    raw = str(os.environ.get("OPENCLAW_KNOWLEDGE_REVIEW_QUEUE_ALLOWED_CALLSITES") or "").strip()
    if raw:
        for token in raw.split(","):
            value = token.strip()
            if value:
                allowed.add(value)
    return sorted(allowed)


def _enforce_wrapper_only_contract(command: str) -> Optional[Dict[str, Any]]:
    if command not in {"enqueue", "transition"}:
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
            "hint": "bash ops/openclaw/continuity.sh knowledge-queue <enqueue|transition> ... --json",
        }

    if not callsite:
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_callsite_missing",
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh knowledge-queue <enqueue|transition> ... --json",
        }

    if callsite not in allowed_callsites:
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_callsite_not_allowlisted",
            "callsite": callsite,
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh knowledge-queue <enqueue|transition> ... --json",
        }

    return None


def handle_enqueue(args: argparse.Namespace, repo_root: Path) -> Tuple[Dict[str, Any], int]:
    item_path = Path(args.item).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser().resolve()
    queue_path = Path(args.queue_path).expanduser()

    try:
        item_doc = load_json_file(item_path)
    except Exception as exc:
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "enqueue",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "block_reason": "schema_invalid",
                "details": {"error": "item_json_unreadable", "detail": str(exc)},
            },
            2,
        )

    valid_schema, reason, details = validate_jsonschema(item_doc, schema_path)
    if not valid_schema:
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "enqueue",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "block_reason": reason or "schema_invalid",
                "details": details,
            },
            2,
        )

    if not isinstance(item_doc, dict):
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "enqueue",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "block_reason": "schema_invalid",
                "details": {"error": "item_not_object"},
            },
            2,
        )

    queue_item_id = item_doc.get("queue_item_id")
    queue_state = item_doc.get("queue_state")

    if queue_state != "PENDING_REVIEW":
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "enqueue",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "block_reason": "state_transition_invalid",
                "queue_item_id": queue_item_id,
                "details": {"error": "initial_state_must_be_pending_review", "queue_state": queue_state},
            },
            2,
        )

    object_ok, object_reason, object_details = _validate_knowledge_object(item_doc, repo_root)
    if not object_ok:
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "enqueue",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "block_reason": object_reason,
                "queue_item_id": queue_item_id,
                "details": object_details,
            },
            2,
        )

    queue_doc, queue_error, queue_details = load_or_init_queue(
        repo_root=repo_root,
        queue_path=queue_path,
        max_items=int(args.max_items),
    )
    if queue_doc is None:
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "enqueue",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "block_reason": queue_error,
                "queue_item_id": queue_item_id,
                "details": queue_details,
            },
            2,
        )

    items = queue_doc.get("items") if isinstance(queue_doc.get("items"), list) else []

    if any(isinstance(row, dict) and row.get("queue_item_id") == queue_item_id for row in items):
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "enqueue",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "block_reason": "duplicate_queue_item_id",
                "queue_item_id": queue_item_id,
                "details": {"queue_size": len(items)},
            },
            2,
        )

    max_items = int(queue_doc.get("max_items") or args.max_items)
    if len(items) >= max_items:
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "enqueue",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "block_reason": "queue_full",
                "queue_item_id": queue_item_id,
                "details": {"queue_size": len(items), "max_items": max_items},
            },
            2,
        )

    items.append(item_doc)
    queue_doc["items"] = items
    queue_doc["updated_at"] = now_iso()

    write_ok, write_details = persist_queue(repo_root=repo_root, queue_path=queue_path, queue_doc=queue_doc)
    if not write_ok:
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "enqueue",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "block_reason": "gate_unavailable",
                "queue_item_id": queue_item_id,
                "details": write_details,
            },
            2,
        )

    result = {
        "schema": "clawd.knowledge_review_queue.decision.v1",
        "action": "enqueue",
        "evaluated_at": now_iso(),
        "decision": "PASS",
        "queue_item_id": queue_item_id,
        "queue_state": "PENDING_REVIEW",
        "queue": {
            "queue_size": len(items),
            "max_items": max_items,
            "state_counts": _state_counts(items),
            **queue_details,
            **write_details,
        },
        "knowledge_object": object_details,
    }
    return result, 0


def handle_transition(args: argparse.Namespace, repo_root: Path) -> Tuple[Dict[str, Any], int]:
    queue_path = Path(args.queue_path).expanduser()
    queue_doc, queue_error, queue_details = load_or_init_queue(
        repo_root=repo_root,
        queue_path=queue_path,
        max_items=int(args.max_items),
    )
    if queue_doc is None:
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "transition",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "block_reason": queue_error,
                "details": queue_details,
            },
            2,
        )

    items = queue_doc.get("items") if isinstance(queue_doc.get("items"), list) else []
    target_idx: Optional[int] = None
    target_item: Optional[Dict[str, Any]] = None

    for idx, row in enumerate(items):
        if isinstance(row, dict) and row.get("queue_item_id") == args.queue_item_id:
            target_idx = idx
            target_item = row
            break

    if target_idx is None or target_item is None:
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "transition",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "queue_item_id": args.queue_item_id,
                "block_reason": "knowledge_object_unresolved",
                "details": {"error": "queue_item_not_found"},
            },
            2,
        )

    from_state = target_item.get("queue_state")
    to_state = args.to_state
    actor_role = args.actor_role

    if not isinstance(from_state, str) or from_state not in ALLOWED_TRANSITIONS:
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "transition",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "queue_item_id": args.queue_item_id,
                "from_state": from_state,
                "to_state": to_state,
                "block_reason": "state_transition_invalid",
                "details": {"error": "unknown_from_state"},
            },
            2,
        )

    if to_state not in ALLOWED_TRANSITIONS[from_state]:
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "transition",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "queue_item_id": args.queue_item_id,
                "from_state": from_state,
                "to_state": to_state,
                "block_reason": "state_transition_invalid",
                "details": {
                    "allowed": sorted(list(ALLOWED_TRANSITIONS[from_state])),
                },
            },
            2,
        )

    if not _role_allowed_for_transition(to_state, actor_role):
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "transition",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "queue_item_id": args.queue_item_id,
                "from_state": from_state,
                "to_state": to_state,
                "block_reason": "actor_role_not_allowed",
                "details": {"actor_role": actor_role},
            },
            2,
        )

    trust_tier = ((target_item.get("trust") or {}).get("tier") if isinstance(target_item.get("trust"), dict) else None)
    if to_state in {"APPROVAL_PENDING", "APPROVED", "PROMOTION_READY", "PROMOTED"}:
        if trust_tier not in PROMOTABLE_TRUST_TIERS:
            return (
                {
                    "schema": "clawd.knowledge_review_queue.decision.v1",
                    "action": "transition",
                    "evaluated_at": now_iso(),
                    "decision": "BLOCK",
                    "queue_item_id": args.queue_item_id,
                    "from_state": from_state,
                    "to_state": to_state,
                    "block_reason": "trust_tier_insufficient",
                    "details": {"trust_tier": trust_tier},
                },
                2,
            )

    evidence_ok, evidence_details = evaluate_evidence_requirements(target_item)
    if to_state in {"APPROVAL_PENDING", "APPROVED", "PROMOTION_READY", "PROMOTED"} and not evidence_ok:
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "transition",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "queue_item_id": args.queue_item_id,
                "from_state": from_state,
                "to_state": to_state,
                "block_reason": "evidence_requirements_unsatisfied",
                "evidence": evidence_details,
            },
            2,
        )

    if to_state == "PROMOTED":
        knowledge_object = target_item.get("knowledge_object") if isinstance(target_item.get("knowledge_object"), dict) else {}
        if knowledge_object.get("object_class") != "promotion_candidate":
            return (
                {
                    "schema": "clawd.knowledge_review_queue.decision.v1",
                    "action": "transition",
                    "evaluated_at": now_iso(),
                    "decision": "BLOCK",
                    "queue_item_id": args.queue_item_id,
                    "from_state": from_state,
                    "to_state": to_state,
                    "block_reason": "promotion_id_missing",
                    "details": {"error": "promoted_state_requires_promotion_candidate"},
                },
                2,
            )

        if not isinstance(knowledge_object.get("promotion_id"), str) or not str(knowledge_object.get("promotion_id") or "").strip():
            return (
                {
                    "schema": "clawd.knowledge_review_queue.decision.v1",
                    "action": "transition",
                    "evaluated_at": now_iso(),
                    "decision": "BLOCK",
                    "queue_item_id": args.queue_item_id,
                    "from_state": from_state,
                    "to_state": to_state,
                    "block_reason": "promotion_id_missing",
                },
                2,
            )

    review = target_item.get("review") if isinstance(target_item.get("review"), dict) else {}
    approval = target_item.get("approval") if isinstance(target_item.get("approval"), dict) else {}

    ts = now_iso()
    reason_note = args.reason or ""

    if to_state == "UNDER_REVIEW":
        review.update(
            {
                "state": "in_review",
                "reviewer_id": args.actor_id,
                "reviewer_role": actor_role,
                "reviewed_at": ts,
                "notes": reason_note,
            }
        )
    elif to_state == "CHANGES_REQUESTED":
        review.update(
            {
                "state": "changes_requested",
                "reviewer_id": args.actor_id,
                "reviewer_role": actor_role,
                "reviewed_at": ts,
                "notes": reason_note,
            }
        )
    elif to_state == "REJECTED":
        review.update(
            {
                "state": "rejected",
                "reviewer_id": args.actor_id,
                "reviewer_role": actor_role,
                "reviewed_at": ts,
                "notes": reason_note,
            }
        )
        approval.update({"decision": "rejected", "notes": reason_note})
    elif to_state == "APPROVAL_PENDING":
        review.update(
            {
                "state": "approved",
                "reviewer_id": args.actor_id,
                "reviewer_role": actor_role,
                "reviewed_at": ts,
                "notes": reason_note,
                "evidence_ok": True,
            }
        )
    elif to_state == "APPROVED":
        approval.update(
            {
                "decision": "approved",
                "approver_id": args.actor_id,
                "approver_role": actor_role,
                "approved_at": ts,
                "notes": reason_note,
            }
        )
    elif to_state == "BLOCKED":
        approval.update({"decision": "blocked", "notes": reason_note})

    target_item["review"] = review
    target_item["approval"] = approval
    target_item["queue_state"] = to_state
    target_item["last_transition_at"] = ts

    items[target_idx] = target_item
    queue_doc["items"] = items
    queue_doc["updated_at"] = ts

    write_ok, write_details = persist_queue(repo_root=repo_root, queue_path=queue_path, queue_doc=queue_doc)
    if not write_ok:
        return (
            {
                "schema": "clawd.knowledge_review_queue.decision.v1",
                "action": "transition",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "queue_item_id": args.queue_item_id,
                "from_state": from_state,
                "to_state": to_state,
                "block_reason": "gate_unavailable",
                "details": write_details,
            },
            2,
        )

    return (
        {
            "schema": "clawd.knowledge_review_queue.decision.v1",
            "action": "transition",
            "evaluated_at": now_iso(),
            "decision": "PASS",
            "queue_item_id": args.queue_item_id,
            "actor": {
                "actor_id": args.actor_id,
                "actor_role": actor_role,
            },
            "from_state": from_state,
            "to_state": to_state,
            "evidence": evidence_details,
            "queue": {
                "queue_size": len(items),
                "state_counts": _state_counts(items),
                **queue_details,
                **write_details,
            },
        },
        0,
    )


def handle_status(args: argparse.Namespace, repo_root: Path) -> Tuple[Dict[str, Any], int]:
    queue_path = Path(args.queue_path).expanduser()
    queue_doc, queue_error, queue_details = load_or_init_queue(
        repo_root=repo_root,
        queue_path=queue_path,
        max_items=int(args.max_items),
    )

    if queue_doc is None:
        return (
            {
                "schema": "clawd.knowledge_review_queue.status.v1",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "block_reason": queue_error,
                "details": queue_details,
            },
            2,
        )

    items = queue_doc.get("items") if isinstance(queue_doc.get("items"), list) else []
    state_counts = _state_counts(items)
    selected_item = None
    if args.queue_item_id:
        for row in items:
            if isinstance(row, dict) and row.get("queue_item_id") == args.queue_item_id:
                selected_item = row
                break

    payload = {
        "schema": "clawd.knowledge_review_queue.status.v1",
        "evaluated_at": now_iso(),
        "decision": "PASS",
        "queue": {
            "queue_size": len(items),
            "max_items": int(queue_doc.get("max_items") or args.max_items),
            "state_counts": state_counts,
            **queue_details,
        },
        "item": selected_item,
    }
    return payload, 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Knowledge review/approval/promotion queue helper (bounded v1)")
    sub = ap.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    common.add_argument(
        "--schema-path",
        default=str(DEFAULT_QUEUE_ITEM_SCHEMA_PATH),
        help="Queue-item schema path",
    )
    common.add_argument(
        "--queue-path",
        default=str(DEFAULT_QUEUE_PATH),
        help="Queue state JSON path",
    )
    common.add_argument(
        "--decision-log",
        default=str(DEFAULT_DECISION_LOG),
        help="Append-only decision log path",
    )
    common.add_argument("--no-decision-log", action="store_true", help="Disable append-only decision recording")
    common.add_argument("--max-items", type=int, default=256, help="Maximum queue size (default: 256)")
    common.add_argument("--json", action="store_true", help="Emit pretty JSON")

    ap_enqueue = sub.add_parser("enqueue", parents=[common], help="Enqueue a queue-item JSON")
    ap_enqueue.add_argument("--item", required=True, help="Queue-item JSON path")

    ap_transition = sub.add_parser("transition", parents=[common], help="Transition queue item state")
    ap_transition.add_argument("--queue-item-id", required=True, help="Queue item id")
    ap_transition.add_argument("--to-state", required=True, choices=sorted(ALLOWED_TRANSITIONS.keys()), help="Target queue state")
    ap_transition.add_argument("--actor-id", required=True, help="Actor identity")
    ap_transition.add_argument(
        "--actor-role",
        required=True,
        choices=sorted(list(REVIEWER_ROLES)),
        help="Actor role",
    )
    ap_transition.add_argument("--reason", default="", help="Transition rationale")

    ap_status = sub.add_parser("status", parents=[common], help="Queue status")
    ap_status.add_argument("--queue-item-id", default=None, help="Optional item id filter")

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    wrapper_guard_error = _enforce_wrapper_only_contract(str(args.command or ""))
    if wrapper_guard_error is not None:
        print(json.dumps(wrapper_guard_error, ensure_ascii=False, indent=2 if args.json else None))
        return 2

    repo_root = Path(args.repo_root).expanduser().resolve()

    if args.command == "enqueue":
        result, rc = handle_enqueue(args, repo_root)
    elif args.command == "transition":
        result, rc = handle_transition(args, repo_root)
    elif args.command == "status":
        result, rc = handle_status(args, repo_root)
    else:  # pragma: no cover
        result = {
            "schema": "clawd.knowledge_review_queue.decision.v1",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "block_reason": "unsupported_command",
            "details": {"command": args.command},
        }
        rc = 2

    decision_log_path: Optional[Path] = None
    if not args.no_decision_log and args.command in {"enqueue", "transition"}:
        decision_log_path = Path(args.decision_log).expanduser()

    if args.command in {"enqueue", "transition"}:
        record = append_decision_record(
            decision_log_path=decision_log_path,
            repo_root=repo_root,
            decision_row=result,
        )
        result["decision_record"] = record

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(result if isinstance(result, dict) else {"result": result}))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
