"""Deep StudyOS model interface backed by immutable learning evidence.

Legacy handlers remain available to desktop and Python callers.  The plugin
registers only the two schemas in this module for model use.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from plugins.study_os import tools as legacy
from plugins.study_os.schemas import (
    ATTEMPT_SCHEMA_VERSION,
    PATTERN_PROPOSAL_SCHEMA_VERSION,
    validate_pattern_proposal,
    validate_study_attempt,
)


MASTERY_DIMENSIONS = (
    "recall",
    "recognition",
    "execution",
    "explanation",
    "near_transfer",
    "far_transfer",
)

ANSWER_HEADING_RE = re.compile(
    r"^#{1,6}\s*(?:答案|解析|解答|参考答案|solution|answer)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _payload(args: dict[str, Any]) -> dict[str, Any]:
    data = args.get("data")
    result = dict(data) if isinstance(data, dict) else {}
    for key in ("vault_path", "project_id"):
        if args.get(key) is not None:
            result[key] = args[key]
    return result


def _project(vault: Path, project_id: Any = None) -> dict[str, Any]:
    return legacy._read_project_manifest(vault, project_id)


def _activity_dir(vault: Path, project_id: str) -> Path:
    path = legacy._project_dir(vault, project_id) / "activity"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _attempt_path(vault: Path, project_id: str, occurred_at: str) -> Path:
    month = occurred_at[:7]
    return _activity_dir(vault, project_id) / f"attempts-{month}.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid activity JSON at {path.name}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Activity record at {path.name}:{line_number} must be an object")
        records.append(value)
    return records


def _all_attempts(vault: Path, project_id: str) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for path in sorted(_activity_dir(vault, project_id).glob("attempts-*.jsonl")):
        attempts.extend(_read_jsonl(path))
    attempts.sort(key=lambda item: (str(item.get("occurred_at", "")), str(item.get("attempt_id", ""))))
    return attempts


def _filtered_attempts(vault: Path, project_id: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = _all_attempts(vault, project_id)
    concept = str(filters.get("concept") or "").casefold()
    pattern = str(filters.get("pattern") or "").casefold()
    result = str(filters.get("result") or "")
    item_id = str(filters.get("item_id") or "")
    session_id = str(filters.get("session_id") or "")
    attempt_ids = {str(value) for value in filters.get("attempt_ids", [])} if isinstance(filters.get("attempt_ids"), list) else set()
    start = str(filters.get("start_date") or "")
    end = str(filters.get("end_date") or "")

    def matches(attempt: dict[str, Any]) -> bool:
        occurred = str(attempt.get("occurred_at") or "")[:10]
        return not (
            (concept and concept not in {str(value).casefold() for value in attempt.get("concepts", [])})
            or (pattern and pattern not in {str(value).casefold() for value in attempt.get("patterns", [])})
            or (result and attempt.get("result") != result)
            or (item_id and attempt.get("item_id") != item_id)
            or (session_id and attempt.get("session_id") != session_id)
            or (attempt_ids and attempt.get("attempt_id") not in attempt_ids)
            or (start and occurred < start)
            or (end and occurred > end)
        )

    return [attempt for attempt in attempts if matches(attempt)]


def _record_attempt(args: dict[str, Any]) -> str:
    vault = legacy.resolve_vault_path(args.get("vault_path"))
    project = _project(vault, args.get("project_id"))
    result = str(args.get("result") or "").strip()
    default_score = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0, "abandoned": 0.0}.get(result)
    occurred_at = str(args.get("occurred_at") or datetime.now().astimezone().isoformat(timespec="seconds"))
    attempt = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_id": str(args.get("attempt_id") or f"att-{uuid4().hex[:16]}").strip(),
        "project_id": project["project_id"],
        "item_id": str(args.get("item_id") or "").strip(),
        "occurred_at": occurred_at,
        "response": str(args.get("response") or "").strip(),
        "result": result,
        "score": args.get("score", default_score),
        "duration_seconds": args.get("duration_seconds"),
        "hints_used": args.get("hints_used", 0),
        "self_confidence": args.get("self_confidence"),
        "evaluator_confidence": args.get("evaluator_confidence"),
        "transfer_level": args.get("transfer_level"),
        "concepts": args.get("concepts", []),
        "patterns": args.get("patterns", []),
        "diagnoses": args.get("diagnoses", []),
        "source": args.get("source"),
        "session_id": args.get("session_id"),
        "revision_of": args.get("revision_of"),
    }
    attempt = {key: value for key, value in attempt.items() if value is not None}
    ok, validated = validate_study_attempt(attempt)
    if not ok:
        return legacy._err("VALIDATION_FAILED", "; ".join(validated), {"errors": validated})
    if any(existing.get("attempt_id") == attempt["attempt_id"] for existing in _all_attempts(vault, project["project_id"])):
        return legacy._err("ATTEMPT_EXISTS", f"Attempt already exists: {attempt['attempt_id']}")
    path = _attempt_path(vault, project["project_id"], occurred_at)
    legacy._append_text(path, json.dumps(validated, ensure_ascii=False) + "\n")
    return legacy._ok({"attempt": validated, "path": path.relative_to(vault).as_posix()})


def _remove_attempt(vault: Path, relative_path: str, attempt_id: str) -> None:
    """Remove one just-appended attempt while rolling back a compound write."""
    path = legacy._safe_relative_path(vault, relative_path)
    if not path.exists():
        return
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("attempt_id") != attempt_id
    ]
    if kept:
        legacy._write_text(path, "\n".join(kept) + "\n")
    else:
        path.unlink()


def handle_study_review_detail(args: dict[str, Any], **_kwargs: Any) -> str:
    """Return one example split into prompt and initially-hidden answer."""
    try:
        vault = legacy.resolve_vault_path(args.get("vault_path"))
        note_ref = str(args.get("note") or args.get("path") or "").strip()
        path, matches = legacy._find_note(vault, note_ref)
        if matches:
            return legacy._err(
                "NOTE_AMBIGUOUS",
                f"More than one note matched {note_ref!r}",
                {"matches": [item.relative_to(vault).as_posix() for item in matches[:20]]},
            )
        if path is None:
            return legacy._err("NOTE_NOT_FOUND", f"Note not found: {note_ref}")
        note, warnings = legacy.parse_note(path, vault, include_body=True)
        if note.get("layer") != "example":
            return legacy._err("NOT_REVIEW_ITEM", "Only example notes can be opened in the review runner")
        body = str(note.pop("body", ""))
        match = ANSWER_HEADING_RE.search(body)
        prompt = body[: match.start()].strip() if match else body.strip()
        answer = body[match.start() :].strip() if match else None
        return legacy._ok(
            {
                "item": note,
                "prompt_markdown": prompt,
                "answer_markdown": answer,
                "has_answer": answer is not None,
            },
            warnings,
        )
    except Exception as exc:
        return legacy._err("REVIEW_DETAIL_FAILED", str(exc))


def handle_study_review_submission(args: dict[str, Any], **_kwargs: Any) -> str:
    """Atomically record evidence and advance one spaced-repetition item."""
    vault: Path | None = None
    note_path: Path | None = None
    original_note: str | None = None
    attempt_path: str | None = None
    attempt_id: str | None = None
    try:
        vault = legacy.resolve_vault_path(args.get("vault_path"))
        project = _project(vault, args.get("project_id"))
        note_ref = str(args.get("note") or "").strip()
        note_path, matches = legacy._find_note(vault, note_ref)
        if matches:
            return legacy._err("NOTE_AMBIGUOUS", f"More than one note matched {note_ref!r}")
        if note_path is None:
            return legacy._err("NOTE_NOT_FOUND", f"Note not found: {note_ref}")
        note, warnings = legacy.parse_note(note_path, vault, include_body=False)
        if note.get("layer") != "example":
            return legacy._err("NOT_REVIEW_ITEM", "Only example notes can be submitted for review")

        result = str(args.get("result") or "").strip()
        if result not in {"correct", "partial", "incorrect"}:
            return legacy._err("VALIDATION_FAILED", "result must be correct, partial, or incorrect")
        duration = args.get("duration_seconds")
        if not isinstance(duration, int) or isinstance(duration, bool) or duration < 0:
            return legacy._err("VALIDATION_FAILED", "duration_seconds must be a non-negative integer")
        confidence = args.get("self_confidence")
        if not isinstance(confidence, int) or isinstance(confidence, bool) or not 1 <= confidence <= 5:
            return legacy._err("VALIDATION_FAILED", "self_confidence must be an integer from 1 to 5")
        old_level = int(note.get("frontmatter", {}).get("review_level", 0))
        new_level = {
            "correct": min(5, old_level + 1),
            "partial": old_level,
            "incorrect": max(0, old_level - 1),
        }[result]
        occurred_at = str(args.get("occurred_at") or datetime.now().astimezone().isoformat(timespec="seconds"))
        attempt_args = {
            "vault_path": str(vault),
            "project_id": project["project_id"],
            "attempt_id": args.get("attempt_id"),
            "item_id": note["path"],
            "occurred_at": occurred_at,
            "response": args.get("response"),
            "result": result,
            "score": {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[result],
            "duration_seconds": duration,
            "hints_used": args.get("hints_used", 0),
            "self_confidence": confidence,
            "transfer_level": args.get("transfer_level", "execution"),
            "concepts": note.get("concepts", []),
            "patterns": note.get("patterns", []),
            "diagnoses": args.get("diagnoses", []),
            "source": note["path"],
            "session_id": args.get("session_id"),
        }
        original_note = note_path.read_text(encoding="utf-8")
        attempt_result = json.loads(_record_attempt(attempt_args))
        if not attempt_result.get("ok"):
            return json.dumps(attempt_result, ensure_ascii=False)
        attempt = attempt_result["data"]["attempt"]
        attempt_id = str(attempt["attempt_id"])
        attempt_path = str(attempt_result["data"]["path"])

        review_result = json.loads(
            legacy.handle_study_record_review(
                {
                    "vault_path": str(vault),
                    "note": note["path"],
                    "passed": result == "correct",
                    "new_review_level": new_level,
                    "log_error": False,
                    "detail": args.get("detail"),
                }
            )
        )
        if not review_result.get("ok"):
            legacy._write_text(note_path, original_note)
            _remove_attempt(vault, attempt_path, attempt_id)
            return legacy._err("REVIEW_SUBMISSION_FAILED", review_result.get("error", {}).get("message", "Review update failed"))
        return legacy._ok(
            {
                "attempt": attempt,
                "review": review_result["data"],
                "completed_today_increment": 1,
            },
            warnings + review_result.get("warnings", []),
        )
    except Exception as exc:
        if vault is not None and note_path is not None and original_note is not None:
            try:
                legacy._write_text(note_path, original_note)
                if attempt_path and attempt_id:
                    _remove_attempt(vault, attempt_path, attempt_id)
            except Exception:
                pass
        return legacy._err("REVIEW_SUBMISSION_FAILED", str(exc))


def _attempt_activity(action: str, args: dict[str, Any]) -> str:
    if action == "record":
        return _record_attempt(args)
    vault = legacy.resolve_vault_path(args.get("vault_path"))
    project = _project(vault, args.get("project_id"))
    if action == "list":
        limit = max(1, min(int(args.get("limit", 100)), 500))
        attempts = _filtered_attempts(vault, project["project_id"], args)
        return legacy._ok({"project_id": project["project_id"], "count": len(attempts), "attempts": attempts[-limit:]})
    if action == "read":
        attempt_id = str(args.get("attempt_id") or "").strip()
        for attempt in _all_attempts(vault, project["project_id"]):
            if attempt.get("attempt_id") == attempt_id:
                return legacy._ok({"attempt": attempt})
        return legacy._err("ATTEMPT_NOT_FOUND", f"Attempt not found: {attempt_id}")
    return legacy._err("INVALID_ACTION", f"Unsupported attempt action: {action}")


def _proposal_dir(vault: Path, project_id: str) -> Path:
    path = legacy._project_dir(vault, project_id) / "pattern-proposals"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _proposal_activity(action: str, args: dict[str, Any]) -> str:
    vault = legacy.resolve_vault_path(args.get("vault_path"))
    project = _project(vault, args.get("project_id"))
    root = _proposal_dir(vault, project["project_id"])
    if action == "save":
        proposal = dict(args.get("proposal") or {})
        proposal.setdefault("schema_version", PATTERN_PROPOSAL_SCHEMA_VERSION)
        proposal.setdefault("project_id", project["project_id"])
        proposal.setdefault("status", "candidate")
        proposal.setdefault("created_at", datetime.now().astimezone().isoformat(timespec="seconds"))
        ok, validated = validate_pattern_proposal(proposal)
        if not ok:
            return legacy._err("VALIDATION_FAILED", "; ".join(validated), {"errors": validated})
        known_ids = {attempt.get("attempt_id") for attempt in _all_attempts(vault, project["project_id"])}
        missing = [item for item in validated["evidence_attempt_ids"] if item not in known_ids]
        if missing:
            return legacy._err("EVIDENCE_NOT_FOUND", "Proposal references unknown attempts", {"attempt_ids": missing})
        proposal_id = legacy._validate_schedule_id(validated["proposal_id"])
        path = root / f"{proposal_id}.json"
        if path.exists():
            return legacy._err("PROPOSAL_EXISTS", f"Pattern proposal already exists: {proposal_id}")
        legacy._write_text(path, json.dumps(validated, ensure_ascii=False, indent=2) + "\n")
        return legacy._ok({"proposal": validated, "path": path.relative_to(vault).as_posix()})
    if action == "list":
        proposals = [legacy._read_json_file(path) for path in sorted(root.glob("*.json"))]
        return legacy._ok({"project_id": project["project_id"], "proposals": proposals})
    if action == "read":
        proposal_id = legacy._validate_schedule_id(args.get("proposal_id"))
        path = root / f"{proposal_id}.json"
        if not path.exists():
            return legacy._err("PROPOSAL_NOT_FOUND", f"Pattern proposal not found: {proposal_id}")
        return legacy._ok({"proposal": legacy._read_json_file(path)})
    return legacy._err("INVALID_ACTION", f"Unsupported pattern_proposal action: {action}")


_RESOURCE_HANDLERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "project": legacy.handle_study_project,
    "schedule": legacy.handle_study_schedule,
    "learning_record": legacy.handle_study_learning_record,
    "decision": legacy.handle_study_decision,
    "lesson": legacy.handle_study_lesson,
    "prompt_context": legacy.handle_study_prompt_context,
    "session": legacy.handle_study_log_session,
    "memory": legacy.handle_study_sync_memory,
}


def _dispatch_resource(resource: str, action: str, data: dict[str, Any]) -> str:
    if resource in _RESOURCE_HANDLERS:
        if resource not in {"session", "memory"}:
            data.setdefault("action", action)
        return _RESOURCE_HANDLERS[resource](data)
    if resource == "note":
        handler = {
            "list": legacy.handle_study_list_notes,
            "read": legacy.handle_study_read_note,
            "extract": legacy.handle_study_extract_concepts,
        }.get(action)
    elif resource == "review":
        handler = {
            "due": legacy.handle_study_due_reviews,
            "record": legacy.handle_study_record_review,
            "submit": handle_study_review_submission,
            "create_task": legacy.handle_study_create_review_task,
            "stats": legacy.handle_study_review_stats,
            "weekly_report": legacy.handle_study_generate_weekly_report,
            "export_anki": legacy.handle_study_export_anki_candidates,
        }.get(action)
    elif resource == "error":
        handler = legacy.handle_study_log_error if action == "record" else None
    elif resource == "concept":
        handler = {
            "graph": legacy.handle_study_concept_graph,
            "queue": legacy.handle_study_learning_queue,
            "update_state": legacy.handle_study_update_concept_state,
        }.get(action)
    elif resource == "curriculum":
        handler = {
            "create": legacy.handle_study_create_curriculum,
            "list": legacy.handle_study_list_curricula,
            "import_plan": legacy.handle_study_import_plan,
            "progress": legacy.handle_study_plan_progress,
        }.get(action)
    else:
        handler = None
    if handler is None:
        return legacy._err("INVALID_RESOURCE_ACTION", f"Unsupported StudyOS operation: {resource}.{action}")
    return handler(data)


def handle_study_activity(args: dict[str, Any], **_kwargs: Any) -> str:
    """Record/query evidence and access all durable StudyOS resources."""
    try:
        resource = str(args.get("resource") or "").strip()
        action = str(args.get("action") or "").strip()
        data = _payload(args)
        if resource == "attempt":
            return _attempt_activity(action, data)
        if resource == "pattern_proposal":
            return _proposal_activity(action, data)
        return _dispatch_resource(resource, action, data)
    except ValueError as exc:
        return legacy._err("VALIDATION_FAILED", str(exc))
    except FileNotFoundError as exc:
        return legacy._err("NOT_FOUND", str(exc))
    except Exception as exc:
        return legacy._err("STUDY_ACTIVITY_FAILED", str(exc))


def _attempt_score(attempt: dict[str, Any]) -> float:
    score = attempt.get("score")
    return float(score) if isinstance(score, (int, float)) else 0.0


def _diagnosis(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    diagnosis_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    concept_attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transfer: Counter[str] = Counter()
    dimension_attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    high_confidence_errors: list[str] = []
    low_confidence_successes: list[str] = []
    for attempt in attempts:
        attempt_id = str(attempt.get("attempt_id"))
        score = _attempt_score(attempt)
        confidence = attempt.get("self_confidence")
        if isinstance(confidence, int) and confidence >= 4 and score < 0.8:
            high_confidence_errors.append(attempt_id)
        if isinstance(confidence, int) and confidence <= 2 and score >= 0.8:
            low_confidence_successes.append(attempt_id)
        if attempt.get("transfer_level"):
            dimension = str(attempt["transfer_level"])
            transfer[dimension] += 1
            dimension_attempts[dimension].append(attempt)
        for concept in attempt.get("concepts", []):
            concept_attempts[str(concept)].append(attempt)
        for item in attempt.get("diagnoses", []):
            concept = str(item.get("concept") or "unscoped")
            diagnosis_groups[(str(item.get("kind") or "unclassified"), concept)].append(attempt_id)

    concepts = []
    for concept, items in concept_attempts.items():
        average = sum(_attempt_score(item) for item in items) / len(items)
        concepts.append({
            "concept": concept,
            "attempt_count": len(items),
            "average_score": round(average, 3),
            "evidence_attempt_ids": [str(item.get("attempt_id")) for item in items],
        })
    concepts.sort(key=lambda item: (item["average_score"], -item["attempt_count"], item["concept"]))
    clusters = [
        {"kind": kind, "concept": concept, "count": len(ids), "evidence_attempt_ids": ids}
        for (kind, concept), ids in diagnosis_groups.items()
    ]
    clusters.sort(key=lambda item: (-item["count"], item["concept"], item["kind"]))
    dimensions: dict[str, dict[str, Any]] = {}
    for dimension in MASTERY_DIMENSIONS:
        items = dimension_attempts.get(dimension, [])
        dimensions[dimension] = {
            "status": "observed" if items else "unobserved",
            "attempt_count": len(items),
            "average_score": round(sum(_attempt_score(item) for item in items) / len(items), 3) if items else None,
            "evidence_attempt_ids": [str(item.get("attempt_id")) for item in items],
        }
    midpoint = len(attempts) // 2
    score_delta = None
    if midpoint:
        earlier = attempts[:midpoint]
        later = attempts[midpoint:]
        early_average = sum(_attempt_score(item) for item in earlier) / len(earlier)
        late_average = sum(_attempt_score(item) for item in later) / len(later)
        score_delta = round(late_average - early_average, 3)
    return {
        "attempt_count": len(attempts),
        "average_score": round(sum(_attempt_score(item) for item in attempts) / len(attempts), 3) if attempts else 0.0,
        "concepts": concepts,
        "diagnosis_clusters": clusters,
        "calibration": {
            "high_confidence_error_attempt_ids": high_confidence_errors,
            "low_confidence_success_attempt_ids": low_confidence_successes,
        },
        "transfer_evidence": dict(transfer),
        "mastery_dimensions": dimensions,
        "score_delta_earlier_to_later": score_delta,
    }


def _recommendations(diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for cluster in diagnosis["diagnosis_clusters"][:3]:
        if cluster["count"] < 2:
            continue
        recommendations.append({
            "priority": "high",
            "intervention": "prerequisite_repair" if cluster["kind"] == "concept_confusion" else "misconception_probe",
            "concept": cluster["concept"],
            "reason": f"{cluster['kind']} repeated {cluster['count']} times",
            "evidence_attempt_ids": cluster["evidence_attempt_ids"],
        })
    if diagnosis["calibration"]["high_confidence_error_attempt_ids"]:
        recommendations.append({
            "priority": "high",
            "intervention": "calibration_check",
            "reason": "High-confidence errors need explicit prediction before feedback",
            "evidence_attempt_ids": diagnosis["calibration"]["high_confidence_error_attempt_ids"],
        })
    transfer = diagnosis["transfer_evidence"]
    if diagnosis["attempt_count"] and not (transfer.get("near_transfer") or transfer.get("far_transfer")):
        recommendations.append({
            "priority": "medium",
            "intervention": "near_transfer_probe",
            "reason": "No transfer evidence has been recorded yet",
            "evidence_attempt_ids": [],
        })
    if not recommendations and diagnosis["attempt_count"]:
        recommendations.append({
            "priority": "medium",
            "intervention": "retention_probe",
            "reason": "No repeated gap dominates; verify retention after spacing",
            "evidence_attempt_ids": [],
        })
    return recommendations


def _pattern_proposals(project_id: str, diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    for index, cluster in enumerate(diagnosis["diagnosis_clusters"]):
        if cluster["count"] < 2:
            continue
        concept_slug = legacy._slugify(cluster["concept"], "unscoped")
        kind_slug = legacy._slugify(cluster["kind"], "gap")
        proposals.append({
            "schema_version": PATTERN_PROPOSAL_SCHEMA_VERSION,
            "proposal_id": f"proposal-{date.today().isoformat()}-{concept_slug}-{kind_slug}-{index + 1}",
            "project_id": project_id,
            "title": f"补充 {cluster['concept']} 的 {cluster['kind']} 失败路径",
            "change_type": "supplement",
            "status": "candidate",
            "rationale": f"The same diagnosis appeared in {cluster['count']} attempts; keep it a candidate until a transfer probe validates the change.",
            "evidence_attempt_ids": cluster["evidence_attempt_ids"],
            "suggested_change": {
                "recognition_signal": cluster["concept"],
                "failure_path": cluster["kind"],
                "validation_needed": "near_transfer",
            },
            "created_at": created_at,
        })
    return proposals


def _probe_blueprint(diagnosis: dict[str, Any]) -> dict[str, Any] | None:
    if not diagnosis["attempt_count"]:
        return None
    recommendations = _recommendations(diagnosis)
    selected = recommendations[0] if recommendations else {
        "intervention": "retention_probe",
        "reason": "Verify retained understanding",
        "evidence_attempt_ids": [],
    }
    weakest = diagnosis["concepts"][0] if diagnosis["concepts"] else None
    purpose = selected["intervention"]
    variation = {
        "prerequisite_repair": "isolate the prerequisite before the original procedure",
        "misconception_probe": "change the condition that distinguishes the observed wrong rule from the correct rule",
        "calibration_check": "require a confidence prediction and justification before feedback",
        "near_transfer_probe": "change surface details while preserving the solution invariant",
        "retention_probe": "use delayed free retrieval without cues",
    }.get(purpose, "test the targeted gap with one controlled variation")
    evidence_ids = list(dict.fromkeys(
        list(selected.get("evidence_attempt_ids", []))
        + (list(weakest.get("evidence_attempt_ids", [])) if weakest else [])
    ))
    return {
        "purpose": purpose,
        "target_concept": weakest.get("concept") if weakest else None,
        "variation_instruction": variation,
        "difficulty_policy": "change one diagnostic variable at a time",
        "response_policy": "ask for the learner's answer and confidence before revealing feedback",
        "rubric_requirements": ["correct outcome", "valid reasoning", "conditions checked", "independent completion"],
        "evidence_attempt_ids": evidence_ids,
        "reason": selected.get("reason"),
    }


def handle_study_coach(args: dict[str, Any], **_kwargs: Any) -> str:
    """Derive diagnoses and next actions from immutable attempt evidence."""
    try:
        action = str(args.get("action") or "diagnose").strip()
        scope = str(args.get("scope") or "project").strip()
        data = _payload(args)
        vault = legacy.resolve_vault_path(data.get("vault_path"))
        project = _project(vault, data.get("project_id"))
        if scope == "week" and not data.get("start_date") and not data.get("end_date"):
            today = date.today()
            data["start_date"] = (today - timedelta(days=today.weekday())).isoformat()
            data["end_date"] = today.isoformat()
        elif scope == "session" and not data.get("session_id") and not data.get("attempt_ids"):
            return legacy._err("MISSING_SCOPE_FILTER", "session scope requires data.session_id or data.attempt_ids")
        elif scope == "concept" and not data.get("concept"):
            return legacy._err("MISSING_SCOPE_FILTER", "concept scope requires data.concept")
        attempts = _filtered_attempts(vault, project["project_id"], data)
        diagnosis = _diagnosis(attempts)
        evidence_ids = [str(item.get("attempt_id")) for item in attempts]
        if action == "diagnose":
            output: dict[str, Any] = {"diagnosis": diagnosis}
        elif action == "summarize":
            weakest = diagnosis["concepts"][:3]
            strongest = sorted(diagnosis["concepts"], key=lambda item: (-item["average_score"], -item["attempt_count"]))[:3]
            transfer = diagnosis["transfer_evidence"]
            output = {
                "summary": {
                    "scope": scope,
                    "attempt_count": diagnosis["attempt_count"],
                    "average_score": diagnosis["average_score"],
                    "strongest_concepts": strongest,
                    "weakest_concepts": weakest,
                    "unverified": [] if transfer.get("near_transfer") or transfer.get("far_transfer") else ["transfer"],
                    "unverified_dimensions": [
                        dimension
                        for dimension, result in diagnosis["mastery_dimensions"].items()
                        if result["status"] == "unobserved"
                    ],
                    "evidence_attempt_ids": evidence_ids,
                }
            }
        elif action == "recommend":
            output = {"recommendations": _recommendations(diagnosis), "diagnosis": diagnosis}
        elif action == "propose_pattern":
            output = {
                "proposals": _pattern_proposals(project["project_id"], diagnosis),
                "policy": "Proposals are not persisted or applied automatically; save explicitly with study_activity after review.",
            }
        elif action == "generate_probe":
            blueprint = _probe_blueprint(diagnosis)
            if blueprint is None:
                return legacy._err("INSUFFICIENT_EVIDENCE", "Record at least one attempt before generating a diagnostic probe")
            output = {
                "probe_blueprint": blueprint,
                "policy": "Generate one problem from this blueprint; record the learner response as a new attempt before judging transfer.",
            }
        else:
            return legacy._err("INVALID_ACTION", f"Unsupported coach action: {action}")
        return legacy._ok({"project_id": project["project_id"], **output, "evidence_attempt_ids": evidence_ids})
    except ValueError as exc:
        return legacy._err("VALIDATION_FAILED", str(exc))
    except FileNotFoundError as exc:
        return legacy._err("NOT_FOUND", str(exc))
    except Exception as exc:
        return legacy._err("STUDY_COACH_FAILED", str(exc))


STUDY_ACTIVITY_SCHEMA = {
    "description": "Single StudyOS persistence interface. Record/query immutable attempts and manage projects, notes, reviews, concepts, curricula, schedules, records, lessons, and evidence-backed pattern proposals. For review.due, data supports explicit notes, subjects, tags, concepts, difficulties, levels, review_state, match, sort, and limit selectors. For a graded interactive review, prefer review.submit: it atomically stores the immutable attempt and advances spaced repetition. Put operation parameters in data.",
    "parameters": {
        "type": "object",
        "properties": {
            "resource": {
                "type": "string",
                "enum": ["attempt", "pattern_proposal", "project", "schedule", "note", "review", "error", "concept", "curriculum", "learning_record", "decision", "lesson", "prompt_context", "session", "memory"],
            },
            "action": {"type": "string", "description": "Resource action, e.g. attempt.record/list/read, review.due/submit/stats, concept.graph/queue/update_state, or project.init/status."},
            "vault_path": {"type": "string"},
            "project_id": {"type": "string"},
            "data": {"type": "object", "description": "Parameters for the selected resource action."},
        },
        "required": ["resource", "action"],
    },
}


STUDY_COACH_SCHEMA = {
    "description": "Evidence-driven StudyOS coach. Diagnose attempts, summarize demonstrated change, recommend an intervention, generate a diagnostic-probe blueprint, or propose a versioned problem-pattern improvement. Never claims mastery without attempt evidence.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["diagnose", "summarize", "recommend", "generate_probe", "propose_pattern"]},
            "scope": {"type": "string", "enum": ["session", "concept", "week", "project"]},
            "vault_path": {"type": "string"},
            "project_id": {"type": "string"},
            "data": {"type": "object", "description": "Attempt filters: concept, pattern, item_id, result, start_date, or end_date."},
        },
        "required": ["action"],
    },
}
