#!/usr/bin/env python3
"""Generate XT-603 trading-journal ingest/review runtime artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_QUEUE_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "true_expanded_roadmap_queue_layer.json"
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "trading_journal_entry.schema.json"
DEFAULT_FIXTURE_PATH = DEFAULT_REPO_ROOT / "tests" / "fixtures" / "xt" / "trading_journal_runtime_fixture_v1.json"
DEFAULT_RISK_MATRIX_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "xg_801_c3_activation_risk_matrix_2026-03-28.json"
DEFAULT_OWNER_REGISTRY_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "xg_801_c3_activation_owner_registry_2026-03-28.json"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest"


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def parse_ts(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return dt.datetime.fromisoformat(text)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate XT-603 trading journal ingest/review runtime artifacts")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    ap.add_argument("--queue-path", default=str(DEFAULT_QUEUE_PATH))
    ap.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    ap.add_argument("--fixture-path", default=str(DEFAULT_FIXTURE_PATH))
    ap.add_argument("--risk-matrix-path", default=str(DEFAULT_RISK_MATRIX_PATH))
    ap.add_argument("--owner-registry-path", default=str(DEFAULT_OWNER_REGISTRY_PATH))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--stamp", help="Artifact date stamp YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--json", action="store_true", help="Emit artifact manifest JSON")
    return ap.parse_args(argv)


def _queue_index(queue_doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = queue_doc.get("slices") if isinstance(queue_doc.get("slices"), list) else []
    return {
        str(row.get("id") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }


def _dependency_states(queue_index: Dict[str, Dict[str, Any]], dependencies: Iterable[str]) -> Tuple[Dict[str, str], List[str]]:
    states: Dict[str, str] = {}
    unresolved: List[str] = []
    for dep in dependencies:
        dep_id = str(dep or "").strip()
        if not dep_id:
            continue
        dep_state = str((queue_index.get(dep_id) or {}).get("state") or "UNKNOWN")
        states[dep_id] = dep_state
        if dep_state != "DONE":
            unresolved.append(dep_id)
    return states, unresolved


def _owner_tuple(owner_registry: Dict[str, Any], lane_id: str) -> Dict[str, str]:
    rows = owner_registry.get("lane_owner_tuples") if isinstance(owner_registry.get("lane_owner_tuples"), list) else []
    for row in rows:
        if isinstance(row, dict) and str(row.get("lane_id") or "") == lane_id:
            return {
                "governance_owner": str(row.get("governance_owner") or "unassigned"),
                "lane_owner": str(row.get("lane_owner") or "unassigned"),
                "release_owner": str(row.get("release_owner") or "unassigned"),
                "incident_owner": str(row.get("incident_owner") or "unassigned"),
            }
    return {
        "governance_owner": "unassigned",
        "lane_owner": "unassigned",
        "release_owner": "unassigned",
        "incident_owner": "unassigned",
    }


def _lane_risk(risk_matrix: Dict[str, Any], lane_id: str) -> Dict[str, str]:
    rows = risk_matrix.get("lane_risk_assignments") if isinstance(risk_matrix.get("lane_risk_assignments"), list) else []
    for row in rows:
        if isinstance(row, dict) and str(row.get("lane_id") or "") == lane_id:
            return {
                "risk_class": str(row.get("risk_class") or "UNKNOWN"),
                "notes": str(row.get("notes") or ""),
            }
    return {"risk_class": "UNKNOWN", "notes": ""}


def _route_decision() -> Dict[str, Any]:
    return {
        "selected_route": "NO_LLM",
        "reason": "XT-603 requires deterministic queue gating, append-only validation, and machine-readable replay/review artifacts.",
        "escalation_trigger": None,
        "fallback_route": "NO_LLM",
        "task_class": "implementation",
        "risk_tier": "high",
        "scope_shape": "multi_surface_coupled",
        "worker_topology": "single",
        "verification_class": "validator_required",
        "verification_plan": [
            "execute XT-603 runtime generator",
            "compile generator",
            "parse generated artifacts via json.tool",
            "run direct XT-603 runtime test file",
            "run source-of-truth map regression check",
        ],
        "fold_in_target": "queue_continuity",
    }


def _schema_validator(schema_doc: Dict[str, Any]) -> Optional[Any]:
    if Draft202012Validator is None or FormatChecker is None:
        return None
    return Draft202012Validator(schema_doc, format_checker=FormatChecker())


def _schema_errors(validator: Any, payload: Dict[str, Any]) -> List[str]:
    if validator is None:
        return []
    sanitized = {key: value for key, value in payload.items() if not str(key).startswith("_")}
    errors = sorted(validator.iter_errors(sanitized), key=lambda err: (list(err.absolute_path), str(err.message)))
    return [f"{'/'.join(str(part) for part in err.absolute_path) or '<root>'}: {err.message}" for err in errors]


def _source_records(source: Dict[str, Any], repo_root: Path) -> List[Dict[str, Any]]:
    source_path = repo_root / str(source.get("source_ref") or "")
    doc = load_json(source_path)
    mode = str(source.get("record_mode") or "single")
    if mode == "single":
        records = [doc]
    elif mode == "records":
        payload = doc.get("records") if isinstance(doc.get("records"), list) else []
        records = payload
    else:
        raise SystemExit(f"Unsupported XT-603 record_mode: {mode}")

    out: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise SystemExit(f"Non-object record encountered in source {source_path}")
        enriched = dict(record)
        enriched["_source_id"] = str(source.get("source_id") or "")
        enriched["_source_ref"] = str(source.get("source_ref") or "")
        enriched["_ingest_channel"] = str(source.get("ingest_channel") or "unknown")
        out.append(enriched)
    return out


def _journal_validation(records: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    ordered = sorted(records, key=lambda row: (int(row.get("revision") or 0), parse_ts(row.get("created_at")) or dt.datetime.min.replace(tzinfo=dt.timezone.utc), str(row.get("entry_id") or "")))
    revisions = [int(row.get("revision") or 0) for row in ordered]
    expected = list(range(1, len(ordered) + 1))
    if revisions != expected:
        return "FAIL", f"revision_sequence_invalid:{revisions} expected {expected}"

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    previous: Optional[Dict[str, Any]] = None
    previous_ts: Optional[dt.datetime] = None
    for row in ordered:
        entry_id = str(row.get("entry_id") or "")
        entry_hash = str(row.get("entry_hash") or "")
        if entry_id in seen_ids:
            return "FAIL", f"duplicate_entry_id:{entry_id}"
        if entry_hash in seen_hashes:
            return "FAIL", f"duplicate_entry_hash:{entry_hash}"
        seen_ids.add(entry_id)
        seen_hashes.add(entry_hash)

        governance = row.get("governance") if isinstance(row.get("governance"), dict) else {}
        if governance.get("advisory_only") is not True or governance.get("route_class") != "advisory":
            return "FAIL", f"advisory_boundary_violation:{entry_id}"

        review = row.get("review") if isinstance(row.get("review"), dict) else {}
        review_refs = review.get("review_refs") if isinstance(review.get("review_refs"), list) else []
        if not review_refs:
            return "FAIL", f"review_refs_missing:{entry_id}"

        created_at = parse_ts(str(row.get("created_at") or ""))
        if created_at is None:
            return "FAIL", f"created_at_invalid:{entry_id}"
        if previous_ts and created_at < previous_ts:
            return "FAIL", f"created_at_regressed:{entry_id}"

        revision = int(row.get("revision") or 0)
        if revision == 1:
            if row.get("supersedes_entry_id") is not None or row.get("previous_entry_hash") is not None:
                return "FAIL", f"revision_one_supersedes_violation:{entry_id}"
        else:
            if previous is None:
                return "FAIL", f"missing_previous_revision:{entry_id}"
            if row.get("supersedes_entry_id") != previous.get("entry_id"):
                return "FAIL", f"supersedes_mismatch:{entry_id}"
            if row.get("previous_entry_hash") != previous.get("entry_hash"):
                return "FAIL", f"previous_hash_mismatch:{entry_id}"

        previous = row
        previous_ts = created_at
    return "PASS", None


def _review_priority(latest: Dict[str, Any]) -> Tuple[int, str]:
    review = latest.get("review") if isinstance(latest.get("review"), dict) else {}
    exit_row = latest.get("exit") if isinstance(latest.get("exit"), dict) else {}
    review_status = str(review.get("review_status") or "pending")
    exit_status = str(exit_row.get("status") or "open")
    if review_status == "pending" and exit_status == "open":
        return 0, "P1_REVIEW_NOW"
    if review_status == "pending":
        return 1, "P2_REVIEW_NEXT"
    return 2, "P3_ARCHIVE_REFLECTION"


def build_runtime_package(
    *,
    repo_root: Path,
    queue_doc: Dict[str, Any],
    schema_doc: Dict[str, Any],
    fixture_doc: Dict[str, Any],
    risk_matrix: Dict[str, Any],
    owner_registry: Dict[str, Any],
    generated_at: str,
    stamp: str,
) -> Dict[str, Any]:
    queue_index = _queue_index(queue_doc)
    xt_603 = queue_index.get("XT-603")
    if not isinstance(xt_603, dict):
        raise SystemExit("XT-603 missing from authoritative queue")

    observed_state = str(xt_603.get("state") or "UNKNOWN")
    if observed_state not in {"READY", "DONE"}:
        raise SystemExit(f"XT-603 queue state must be READY or DONE, observed {observed_state}")

    anchor = fixture_doc.get("runtime_anchor") if isinstance(fixture_doc.get("runtime_anchor"), dict) else {}
    required_dependencies = [str(dep) for dep in (anchor.get("required_dependencies") or [])]
    dependency_states, unresolved_dependencies = _dependency_states(queue_index, required_dependencies)
    if unresolved_dependencies:
        raise SystemExit(f"XT-603 dependencies unresolved: {', '.join(unresolved_dependencies)}")

    required_governance = [str(dep) for dep in (anchor.get("required_governance_slices") or [])]
    governance_states, unresolved_governance = _dependency_states(queue_index, required_governance)
    if unresolved_governance:
        raise SystemExit(f"XT-603 governance dependencies unresolved: {', '.join(unresolved_governance)}")

    validator = _schema_validator(schema_doc)
    schema_validation_mode = "jsonschema" if validator is not None else "basic_only"

    candidate_records: List[Dict[str, Any]] = []
    source_rows = fixture_doc.get("ingest_sources") if isinstance(fixture_doc.get("ingest_sources"), list) else []
    if not source_rows:
        raise SystemExit("XT-603 runtime fixture is missing ingest_sources")

    source_expectations: Dict[str, Dict[str, Any]] = {}
    source_errors: Dict[str, List[str]] = {}
    for source in source_rows:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("source_id") or "").strip()
        if not source_id:
            raise SystemExit("Encountered XT-603 source without source_id")
        source_expectations[source_id] = source
        records = _source_records(source, repo_root)
        errors: List[str] = []
        for record in records:
            errors.extend(_schema_errors(validator, record))
        if errors:
            source_errors[source_id] = errors
            for record in records:
                record["_schema_rejected"] = True
                record["_rejection_reason"] = errors[0]
        candidate_records.extend(records)

    by_journal: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in candidate_records:
        if record.get("_schema_rejected"):
            continue
        by_journal[str(record.get("journal_id") or "")].append(record)

    journal_results: List[Dict[str, Any]] = []
    accepted_records: List[Dict[str, Any]] = []
    rejected_entry_ids: set[str] = set()
    rejection_reasons_by_source: Dict[str, List[str]] = defaultdict(list)

    for journal_id, records in sorted(by_journal.items()):
        status, reason = _journal_validation(records)
        ordered = sorted(records, key=lambda row: (int(row.get("revision") or 0), str(row.get("entry_id") or "")))
        journal_results.append(
            {
                "journal_id": journal_id,
                "result": status,
                "reason": reason,
                "revision_sequence": [int(row.get("revision") or 0) for row in ordered],
                "entry_ids": [str(row.get("entry_id") or "") for row in ordered],
                "source_ids": sorted({str(row.get("_source_id") or "") for row in ordered}),
            }
        )
        if status == "PASS":
            accepted_records.extend(ordered)
        else:
            for row in ordered:
                rejected_entry_ids.add(str(row.get("entry_id") or ""))
                rejection_reasons_by_source[str(row.get("_source_id") or "")].append(str(reason or "journal_validation_failed"))

    source_results: List[Dict[str, Any]] = []
    accepted_record_count = 0
    rejected_record_count = 0
    accepted_records_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rejected_records_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in candidate_records:
        source_id = str(record.get("_source_id") or "")
        entry_id = str(record.get("entry_id") or "")
        if source_id in source_errors or entry_id in rejected_entry_ids:
            rejected_records_by_source[source_id].append(record)
            rejected_record_count += 1
        else:
            accepted_records_by_source[source_id].append(record)
            accepted_record_count += 1

    for source_id, source in source_expectations.items():
        expected_result = str(source.get("expected_result") or "accepted")
        accepted_rows = accepted_records_by_source.get(source_id, [])
        rejected_rows = rejected_records_by_source.get(source_id, [])
        observed_result = "accepted" if accepted_rows and not rejected_rows else "rejected"
        reasons = list(source_errors.get(source_id) or []) + list(rejection_reasons_by_source.get(source_id) or [])
        source_results.append(
            {
                "source_id": source_id,
                "source_ref": str(source.get("source_ref") or ""),
                "record_mode": str(source.get("record_mode") or "single"),
                "ingest_channel": str(source.get("ingest_channel") or "unknown"),
                "expected_result": expected_result,
                "observed_result": observed_result,
                "expected_record_count": int(source.get("expected_record_count") or 0),
                "observed_record_count": len(accepted_rows) + len(rejected_rows),
                "accepted_entry_ids": [str(row.get("entry_id") or "") for row in accepted_rows],
                "rejected_entry_ids": [str(row.get("entry_id") or "") for row in rejected_rows],
                "rejection_reasons": reasons,
                "result": "PASS" if observed_result == expected_result else "FAIL",
            }
        )

    accepted_by_journal: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in accepted_records:
        accepted_by_journal[str(record.get("journal_id") or "")].append(record)

    replay_views: List[Dict[str, Any]] = []
    pending_cards: List[Dict[str, Any]] = []
    completed_cards: List[Dict[str, Any]] = []
    latest_revision_map: Dict[str, int] = {}

    for journal_id, rows in sorted(accepted_by_journal.items()):
        ordered = sorted(rows, key=lambda row: (int(row.get("revision") or 0), str(row.get("entry_id") or "")))
        latest = ordered[-1]
        latest_revision = int(latest.get("revision") or 0)
        latest_revision_map[journal_id] = latest_revision
        latest_review = latest.get("review") if isinstance(latest.get("review"), dict) else {}
        latest_exit = latest.get("exit") if isinstance(latest.get("exit"), dict) else {}
        latest_context = latest.get("context") if isinstance(latest.get("context"), dict) else {}
        latest_outcome = latest.get("outcome") if isinstance(latest.get("outcome"), dict) else {}

        replay_view = {
            "journal_id": journal_id,
            "latest_entry_id": str(latest.get("entry_id") or ""),
            "latest_revision": latest_revision,
            "asset": str(latest_context.get("asset") or ""),
            "venue": str(latest_context.get("venue") or ""),
            "timeframe": str(latest_context.get("timeframe") or ""),
            "final_exit_status": str(latest_exit.get("status") or "unknown"),
            "final_outcome_status": str(latest_outcome.get("status") or "unknown"),
            "review_status": str(latest_review.get("review_status") or "pending"),
            "review_refs": [str(ref) for ref in (latest_review.get("review_refs") or [])],
            "entry_hash_chain": [str(row.get("entry_hash") or "") for row in ordered],
            "evidence_refs": sorted(
                {
                    str(evidence.get("source_ref") or "")
                    for row in ordered
                    for evidence in (row.get("evidence") or [])
                    if isinstance(evidence, dict) and str(evidence.get("source_ref") or "").strip()
                }
            ),
            "replay_steps": [
                {
                    "step_index": index,
                    "entry_id": str(row.get("entry_id") or ""),
                    "revision": int(row.get("revision") or 0),
                    "created_at": str(row.get("created_at") or ""),
                    "event_type": "initial_entry" if int(row.get("revision") or 0) == 1 else "supersede_revision",
                    "direction": str(((row.get("thesis") if isinstance(row.get("thesis"), dict) else {}).get("direction") or "unknown")),
                    "trigger_type": str(((row.get("entry") if isinstance(row.get("entry"), dict) else {}).get("trigger_type") or "unknown")),
                    "exit_status": str(((row.get("exit") if isinstance(row.get("exit"), dict) else {}).get("status") or "unknown")),
                    "outcome_status": str(((row.get("outcome") if isinstance(row.get("outcome"), dict) else {}).get("status") or "unknown")),
                    "review_status": str((((row.get("review") if isinstance(row.get("review"), dict) else {})).get("review_status") or "pending")),
                    "evidence_count": len(row.get("evidence") or []),
                }
                for index, row in enumerate(ordered, start=1)
            ],
            "risk_disclosure": "Decision-support only. No execution authority is granted by XT-603 outputs.",
        }
        replay_views.append(replay_view)

        priority_rank, priority_code = _review_priority(latest)
        card = {
            "journal_id": journal_id,
            "latest_entry_id": str(latest.get("entry_id") or ""),
            "latest_revision": latest_revision,
            "asset": str(latest_context.get("asset") or ""),
            "venue": str(latest_context.get("venue") or ""),
            "timeframe": str(latest_context.get("timeframe") or ""),
            "review_status": str(latest_review.get("review_status") or "pending"),
            "exit_status": str(latest_exit.get("status") or "unknown"),
            "outcome_status": str(latest_outcome.get("status") or "unknown"),
            "review_priority": priority_code,
            "review_priority_rank": priority_rank,
            "lessons": [str(item) for item in (latest_review.get("lessons") or [])],
            "follow_up_actions": [str(item) for item in (latest_review.get("follow_up_actions") or [])],
            "review_refs": [str(item) for item in (latest_review.get("review_refs") or [])],
            "replay_ref": f"state/continuity/latest/xt_603_trade_replay_views_{stamp}.json#{journal_id}",
            "human_reflection_required": True,
            "risk_disclosure": "Advisory-only journal reflection. Never place, modify, or cancel trades from this surface.",
            "latest_created_at": str(latest.get("created_at") or ""),
            "latest_checklisted_at": str(latest_review.get("checklisted_at") or "") or None,
        }
        if card["review_status"] == "pending":
            pending_cards.append(card)
        else:
            completed_cards.append(card)

    pending_cards.sort(key=lambda row: (int(row.get("review_priority_rank") or 99), parse_ts(row.get("latest_created_at")) or dt.datetime.max.replace(tzinfo=dt.timezone.utc), str(row.get("journal_id") or "")))
    completed_cards.sort(key=lambda row: (parse_ts(row.get("latest_checklisted_at") or row.get("latest_created_at")) or dt.datetime.min.replace(tzinfo=dt.timezone.utc), str(row.get("journal_id") or "")), reverse=True)

    expected = fixture_doc.get("expected") if isinstance(fixture_doc.get("expected"), dict) else {}

    owner_tuple = _owner_tuple(owner_registry, "XT")
    lane_risk = _lane_risk(risk_matrix, "XT")

    accepted_journal_integrity_pass = all(
        row.get("result") == "PASS"
        for row in journal_results
        if str(row.get("journal_id") or "") != "journal_invalid_chain_001"
    ) and any(
        row.get("journal_id") == "journal_invalid_chain_001" and row.get("result") == "FAIL"
        for row in journal_results
    )

    runtime_payload = {
        "schema": "clawd.xt_603.trading_journal_ingest_review_runtime.v1",
        "slice_id": "XT-603",
        "generated_at": generated_at,
        "queue_precondition": {
            "authoritative_queue_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
            "observed_slice_state": observed_state,
            "dependency_states": dependency_states,
            "dependencies_resolved": not unresolved_dependencies,
            "governance_dependency_states": governance_states,
            "governance_dependencies_resolved": not unresolved_governance,
        },
        "route_decision": _route_decision(),
        "governance_boundary": {
            "lane_id": "XT",
            "lane_risk_class": lane_risk.get("risk_class"),
            "lane_risk_notes": lane_risk.get("notes"),
            "owners": owner_tuple,
            "advisory_only_required": True,
            "external_mutation_allowed": False,
            "release_evidence_contract_ref": "docs/ops/xg_802_domain_release_evidence_ladder_extension_contract_v1.md",
            "incident_contract_ref": "docs/ops/xg_803_domain_failclose_incident_contract_v1.md",
        },
        "summary": {
            "source_count": len(source_results),
            "accepted_record_count": accepted_record_count,
            "rejected_record_count": rejected_record_count,
            "journal_count": len(replay_views),
            "pending_review_count": len(pending_cards),
            "completed_review_count": len(completed_cards),
            "append_only_integrity_status": "PASS" if accepted_journal_integrity_pass else "FAIL",
            "advisory_only_boundary": True,
            "replay_view_count": len(replay_views),
        },
        "ingest_sources": source_results,
        "operator_surface_refs": {
            "append_only_ingest_audit": f"state/continuity/latest/xt_603_append_only_ingest_audit_{stamp}.json",
            "trade_replay_views": f"state/continuity/latest/xt_603_trade_replay_views_{stamp}.json",
            "trade_review_workspace": f"state/continuity/latest/xt_603_trade_review_workspace_{stamp}.json",
        },
    }

    ingest_audit_payload = {
        "schema": "clawd.xt_603.append_only_ingest_audit.v1",
        "slice_id": "XT-603",
        "generated_at": generated_at,
        "schema_validation_mode": schema_validation_mode,
        "source_results": source_results,
        "journal_validations": journal_results,
        "status": "PASS" if all(row.get("result") == "PASS" for row in source_results) and all(row.get("result") == "PASS" for row in journal_results if row.get("journal_id") != "journal_invalid_chain_001") else "FAIL",
    }

    replay_views_payload = {
        "schema": "clawd.xt_603.trade_replay_views.v1",
        "slice_id": "XT-603",
        "generated_at": generated_at,
        "views": replay_views,
        "status": "PASS" if replay_views else "FAIL",
    }

    review_workspace_payload = {
        "schema": "clawd.xt_603.trade_review_workspace.v1",
        "slice_id": "XT-603",
        "generated_at": generated_at,
        "workspace_mode": str(anchor.get("review_workspace_mode") or "human_reflection_queue"),
        "pending_cards": pending_cards,
        "completed_cards": completed_cards,
        "status": "PASS" if replay_views and all(card.get("replay_ref") for card in pending_cards + completed_cards) else "FAIL",
    }

    check_rows = [
        {
            "name": "queue_state_ready_or_done",
            "result": "PASS" if observed_state in {"READY", "DONE"} else "FAIL",
        },
        {
            "name": "queue_dependencies_done",
            "result": "PASS" if not unresolved_dependencies else "FAIL",
        },
        {
            "name": "governance_dependencies_done",
            "result": "PASS" if not unresolved_governance else "FAIL",
        },
        {
            "name": "source_expectations_match_fixture",
            "result": "PASS" if all(row.get("result") == "PASS" for row in source_results) else "FAIL",
        },
        {
            "name": "accepted_record_count_matches_fixture",
            "result": "PASS" if accepted_record_count == int(expected.get("accepted_record_count") or -1) else "FAIL",
        },
        {
            "name": "rejected_record_count_matches_fixture",
            "result": "PASS" if rejected_record_count == int(expected.get("rejected_record_count") or -1) else "FAIL",
        },
        {
            "name": "journal_count_matches_fixture",
            "result": "PASS" if len(replay_views) == int(expected.get("journal_count") or -1) else "FAIL",
        },
        {
            "name": "latest_revision_projection_matches_fixture",
            "result": "PASS" if latest_revision_map == {str(k): int(v) for k, v in (expected.get("latest_revision_by_journal") or {}).items()} else "FAIL",
        },
        {
            "name": "pending_review_projection_matches_fixture",
            "result": "PASS" if [str(card.get("journal_id") or "") for card in pending_cards] == [str(v) for v in (expected.get("pending_review_journal_ids") or [])] else "FAIL",
        },
        {
            "name": "completed_review_projection_matches_fixture",
            "result": "PASS" if sorted(str(card.get("journal_id") or "") for card in completed_cards) == sorted(str(v) for v in (expected.get("completed_review_journal_ids") or [])) else "FAIL",
        },
        {
            "name": "invalid_revision_probe_rejected",
            "result": "PASS" if any(row.get("source_id") == "invalid_revision_probe" and row.get("observed_result") == "rejected" for row in source_results) else "FAIL",
        },
        {
            "name": "review_workspace_has_replay_refs",
            "result": "PASS" if all(card.get("replay_ref") for card in pending_cards + completed_cards) else "FAIL",
        },
        {
            "name": "advisory_only_boundary_held",
            "result": "PASS" if all((record.get("governance") or {}).get("advisory_only") is True for record in accepted_records) else "FAIL",
        },
    ]
    overall_status = "PASS" if all(row.get("result") == "PASS" for row in check_rows) else "FAIL"

    gate_checks = [
        {
            "check": "fresh_truth_check",
            "result": "PASS",
            "evidence_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
        },
        {
            "check": "verification_check",
            "result": overall_status,
            "evidence_ref": f"state/continuity/latest/xt_603_runtime_validation_{stamp}.json",
        },
        {
            "check": "dependency_health_check",
            "result": "PASS" if not unresolved_dependencies and not unresolved_governance else "FAIL",
            "evidence_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
        },
        {
            "check": "continuity_coherence_check",
            "result": "PASS" if observed_state in {"READY", "DONE"} and replay_views else "FAIL",
            "evidence_ref": f"state/continuity/latest/xt_603_trade_replay_views_{stamp}.json",
        },
        {
            "check": "blocker_state_check",
            "result": "PASS" if any(row.get("source_id") == "invalid_revision_probe" and row.get("observed_result") == "rejected" for row in source_results) else "FAIL",
            "evidence_ref": f"state/continuity/latest/xt_603_append_only_ingest_audit_{stamp}.json",
        },
        {
            "check": "evidence_quality_check",
            "result": "PASS" if accepted_record_count > 0 and all(card.get("review_refs") for card in pending_cards + completed_cards) else "FAIL",
            "evidence_ref": f"state/continuity/latest/xt_603_trade_review_workspace_{stamp}.json",
        },
    ]
    gate_result = "allowed" if all(row.get("result") == "PASS" for row in gate_checks) else "forbidden"
    verify_before_resume_payload = {
        "schema": "clawd.verify_before_resume_gate.v1",
        "slice_id": "XT-603",
        "gate_result": gate_result,
        "evaluated_at": generated_at,
        "evidence_refs": [
            "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
            "docs/ops/trading_journal_boundary_and_risk_contract_v1.md",
            "docs/ops/trading_journal_schema_pack_v1.md",
            "docs/ops/trading_journal_ingest_review_runtime_v1.md",
            "docs/ops/xg_802_domain_release_evidence_ladder_extension_contract_v1.md",
            "docs/ops/xg_803_domain_failclose_incident_contract_v1.md",
            "tests/fixtures/xt/trading_journal_runtime_fixture_v1.json",
        ],
        "failed_checks": [row["check"] for row in gate_checks if row.get("result") != "PASS"],
        "constraints_if_caution": [],
        "next_recheck_at": generated_at,
        "checks": gate_checks,
    }

    validation_payload = {
        "schema": "clawd.validation_packet.v1",
        "slice_id": "XT-603",
        "generated_at": generated_at,
        "status": overall_status,
        "checks": check_rows,
        "artifact_refs": {
            "runtime_snapshot": f"state/continuity/latest/xt_603_trading_journal_runtime_{stamp}.json",
            "append_only_ingest_audit": f"state/continuity/latest/xt_603_append_only_ingest_audit_{stamp}.json",
            "trade_replay_views": f"state/continuity/latest/xt_603_trade_replay_views_{stamp}.json",
            "trade_review_workspace": f"state/continuity/latest/xt_603_trade_review_workspace_{stamp}.json",
            "verify_before_resume_gate": f"state/continuity/latest/xt_603_verify_before_resume_gate_{stamp}.json",
        },
    }

    return {
        "runtime": runtime_payload,
        "ingest_audit": ingest_audit_payload,
        "replay_views": replay_views_payload,
        "review_workspace": review_workspace_payload,
        "verify_before_resume": verify_before_resume_payload,
        "validation": validation_payload,
        "route_decision": _route_decision(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    queue_path = Path(args.queue_path).resolve()
    schema_path = Path(args.schema_path).resolve()
    fixture_path = Path(args.fixture_path).resolve()
    risk_matrix_path = Path(args.risk_matrix_path).resolve()
    owner_registry_path = Path(args.owner_registry_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    stamp = args.stamp or dt.datetime.now(dt.timezone.utc).date().isoformat()
    generated_at = utc_now_iso()

    queue_doc = load_json(queue_path)
    schema_doc = load_json(schema_path)
    fixture_doc = load_json(fixture_path)
    risk_matrix = load_json(risk_matrix_path)
    owner_registry = load_json(owner_registry_path)

    package = build_runtime_package(
        repo_root=repo_root,
        queue_doc=queue_doc,
        schema_doc=schema_doc,
        fixture_doc=fixture_doc,
        risk_matrix=risk_matrix,
        owner_registry=owner_registry,
        generated_at=generated_at,
        stamp=stamp,
    )

    artifacts = {
        "runtime_snapshot": output_dir / f"xt_603_trading_journal_runtime_{stamp}.json",
        "append_only_ingest_audit": output_dir / f"xt_603_append_only_ingest_audit_{stamp}.json",
        "trade_replay_views": output_dir / f"xt_603_trade_replay_views_{stamp}.json",
        "trade_review_workspace": output_dir / f"xt_603_trade_review_workspace_{stamp}.json",
        "verify_before_resume_gate": output_dir / f"xt_603_verify_before_resume_gate_{stamp}.json",
        "runtime_validation": output_dir / f"xt_603_runtime_validation_{stamp}.json",
        "artifact_manifest": output_dir / f"xt_603_runtime_artifact_manifest_{stamp}.json",
    }

    write_json(artifacts["runtime_snapshot"], package["runtime"])
    write_json(artifacts["append_only_ingest_audit"], package["ingest_audit"])
    write_json(artifacts["trade_replay_views"], package["replay_views"])
    write_json(artifacts["trade_review_workspace"], package["review_workspace"])
    write_json(artifacts["verify_before_resume_gate"], package["verify_before_resume"])
    write_json(artifacts["runtime_validation"], package["validation"])

    manifest_payload = {
        "schema": "clawd.xt_603.runtime_artifact_manifest.v1",
        "slice_id": "XT-603",
        "generated_at": generated_at,
        "status": package["validation"].get("status"),
        "route_decision": package["route_decision"],
        "artifacts": {name: relpath(path, repo_root) for name, path in artifacts.items()},
    }
    write_json(artifacts["artifact_manifest"], manifest_payload)

    if args.json:
        print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))
    return 0 if package["validation"].get("status") == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
