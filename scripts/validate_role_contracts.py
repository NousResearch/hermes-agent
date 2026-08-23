#!/usr/bin/env python3
"""Deterministic conformance checks for the semantic role-contract registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_ROLE_IDS = (
    "role.ares_supervisor",
    "role.explorer",
    "role.public_evidence_editor",
    "role.data_evidence",
    "role.ml_evaluation",
    "role.statistician",
    "role.cognitive_scientist",
    "role.psychometrician",
    "role.inbox_manager",
    "role.durable_profile_history",
)
EXPLORER_FIELDS = (
    "alternatives",
    "evidence",
    "falsifier",
    "kill_criterion",
    "reconciliation_disposition",
)
DATA_EVIDENCE_PROFILE = "longmemeval-bench"
ALLOWED_EVIDENCE_STATES = {
    "unknown",
    "observed",
    "verified",
    "derived",
    "reconciled",
    "blocked",
    "redacted",
    "superseded",
}
ALLOWED_READINESS = {"draft", "blocked", "public_ready"}
NON_VERIFYING_EVIDENCE_STATES = {"blocked", "unknown", "superseded"}
REQUIRED_DATA_EVIDENCE_FORBIDDEN_PROMOTIONS = {
    "publication_readiness",
    "runtime_enforcement",
    "unverified_claim",
    "final_role_authority",
    "source_truth",
}


def _role_map(registry: dict[str, Any], errors: list[str]) -> dict[str, dict[str, Any]]:
    roles = registry.get("roles")
    if not isinstance(roles, list):
        errors.append("roles must be a list")
        return {}
    role_map: dict[str, dict[str, Any]] = {}
    for role in roles:
        if not isinstance(role, dict) or not isinstance(role.get("role_id"), str):
            errors.append("role contract missing role_id")
            continue
        role_id = role["role_id"]
        if role_id in role_map:
            errors.append(f"duplicate role_id: {role_id}")
        else:
            role_map[role_id] = role
    return role_map


def validate_registry(registry: dict[str, Any]) -> list[str]:
    """Return stable, human-readable conformance errors; empty means valid."""
    errors: list[str] = []
    role_map = _role_map(registry, errors)

    for role_id in registry.get("required_role_ids", REQUIRED_ROLE_IDS):
        if role_id not in role_map:
            errors.append(f"missing required role contract: {role_id}")

    if tuple(registry.get("required_role_ids", ())) != REQUIRED_ROLE_IDS:
        errors.append("required_role_ids do not match the canonical role set")

    evidence_states = set(registry.get("evidence_states", ()))
    if evidence_states != ALLOWED_EVIDENCE_STATES:
        errors.append("evidence_states do not match the canonical evidence-state set")

    artifact_ids: set[str] = set()
    for role_id in REQUIRED_ROLE_IDS:
        role = role_map.get(role_id)
        if role is None:
            continue
        for key in ("name", "kind", "authority", "required_artifacts"):
            if key not in role:
                errors.append(f"role {role_id} missing required contract field {key}")
        artifacts = role.get("required_artifacts", [])
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"role {role_id} has no required artifacts")
            continue
        for artifact in artifacts:
            artifact_id = artifact.get("artifact_id") if isinstance(artifact, dict) else None
            if not isinstance(artifact_id, str):
                errors.append(f"role {role_id} has artifact without artifact_id")
                continue
            if artifact_id in artifact_ids:
                errors.append(f"duplicate artifact_id: {artifact_id}")
            artifact_ids.add(artifact_id)
            fields = artifact.get("required_fields")
            if not isinstance(fields, list) or not fields:
                errors.append(f"artifact {artifact_id} has no required fields")

    explorer = role_map.get("role.explorer")
    if explorer:
        policy = explorer.get("dissent_policy", {})
        if policy.get("preservation") != "preserved_artifact" or policy.get("summarization_may_not_replace_artifact") is not True:
            errors.append("Explorer dissent must preserve the dissent artifact")
        dissent = next((a for a in explorer.get("required_artifacts", []) if a.get("artifact_id") == "explorer_dissent"), None)
        if dissent:
            for field in EXPLORER_FIELDS:
                if field not in dissent.get("required_fields", []):
                    errors.append(f"artifact explorer_dissent missing field {field}")

    editor = role_map.get("role.public_evidence_editor")
    if editor:
        artifacts = {a.get("artifact_id"): a for a in editor.get("required_artifacts", [])}
        ledger = artifacts.get("claim_ledger", {})
        for field in ("claim_id", "claim_text", "evidence_ids", "evidence_state", "blocking_gaps", "provenance"):
            if field not in ledger.get("required_fields", []):
                errors.append(f"artifact claim_ledger missing field {field}")
        if editor.get("publication_policy", {}).get("readiness_requires_no_blocking_evidence_gaps") is not True:
            errors.append("Public Evidence Editor must require no blocking evidence gaps for publication")

    data_lane = role_map.get("role.data_evidence")
    if data_lane:
        if data_lane.get("mapped_profile") != DATA_EVIDENCE_PROFILE:
            errors.append("role.data_evidence must map to longmemeval-bench")
        authority = data_lane.get("authority", {})
        forbidden = set(data_lane.get("forbidden_promotions", [])) | set(authority.get("forbidden_promotions", []))
        for promotion in REQUIRED_DATA_EVIDENCE_FORBIDDEN_PROMOTIONS:
            if promotion not in forbidden:
                errors.append(f"role.data_evidence must forbid promotion: {promotion}")
        for promotion in authority.get("can_promote", []):
            if promotion in forbidden:
                errors.append(f"forbidden promotion in role.data_evidence: {promotion}")

    for decision in registry.get("publication_decision_examples", []):
        if decision.get("readiness") == "public_ready" and decision.get("blocking_evidence_gaps"):
            errors.append(f"public_ready decision has blocking evidence gaps: {decision.get('decision_id', '<unknown>')}")

    fiv = registry.get("fiv_contract", {})
    nodes = fiv.get("nodes", [])
    node_ids = [node.get("id") for node in nodes if isinstance(node, dict)]
    if len(node_ids) != len(set(node_ids)):
        errors.append("duplicate FIV node id")
    node_map = {node.get("id"): node for node in nodes if isinstance(node, dict)}
    for node in nodes:
        if node.get("evidence_state") not in evidence_states:
            errors.append(f"FIV node {node.get('id', '<unknown>')} has invalid evidence state")
    for link in registry.get("fiv_examples", []):
        for field in ("finding_id", "implementation_id", "verification_id"):
            target = link.get(field)
            if target not in node_map:
                errors.append(f"FIV link {field} -> {target} is unresolved")
        if link.get("finding_id") in node_map and node_map[link["finding_id"]].get("kind") != "finding":
            errors.append(f"FIV link finding_id -> {link.get('finding_id')} has wrong node kind")
        if link.get("implementation_id") in node_map and node_map[link["implementation_id"]].get("kind") != "implementation":
            errors.append(f"FIV link implementation_id -> {link.get('implementation_id')} has wrong node kind")
        if link.get("verification_id") in node_map and node_map[link["verification_id"]].get("kind") != "verification":
            errors.append(f"FIV link verification_id -> {link.get('verification_id')} has wrong node kind")

    return errors


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _records(instances: dict[str, Any], key: str, errors: list[str]) -> list[dict[str, Any]]:
    value = instances.get(key, [])
    if not isinstance(value, list):
        errors.append(f"artifact instances field {key} must be a list")
        return []
    records = [record for record in value if isinstance(record, dict)]
    if len(records) != len(value):
        errors.append(f"artifact instances field {key} contains a non-object")
    return records


def validate_artifact_instances(registry: dict[str, Any], instances: dict[str, Any]) -> list[str]:
    """Validate concrete role artifacts without changing registry conformance."""
    errors: list[str] = []
    if not isinstance(instances, dict):
        return ["artifact instances must be an object"]
    evidence_states = set(registry.get("evidence_states", ALLOWED_EVIDENCE_STATES))
    editor = next((role for role in registry.get("roles", [])
                   if isinstance(role, dict) and role.get("role_id") == "role.public_evidence_editor"), {})
    publication_policy = editor.get("publication_policy", {}) if isinstance(editor, dict) else {}
    readiness_vocabulary = set(publication_policy.get("allowed_readiness", ALLOWED_READINESS))

    evidence = _records(instances, "evidence", errors)
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for item in evidence:
        evidence_id = item.get("evidence_id")
        if not _nonempty_string(evidence_id):
            errors.append("evidence artifact missing evidence_id")
            continue
        if evidence_id in evidence_by_id:
            errors.append(f"duplicate evidence_id: {evidence_id}")
        evidence_by_id[evidence_id] = item
        if item.get("evidence_state") not in evidence_states:
            errors.append(f"evidence {evidence_id} has invalid evidence state")

    ledgers = _records(instances, "claim_ledgers", errors)
    ledger_by_id: dict[str, dict[str, Any]] = {}
    for ledger in ledgers:
        claim_id = ledger.get("claim_id")
        if not _nonempty_string(claim_id):
            errors.append("claim ledger missing claim_id")
            continue
        ledger_by_id[claim_id] = ledger
        for field in ("claim_text", "evidence_ids", "evidence_state", "blocking_gaps", "provenance"):
            if field not in ledger:
                errors.append(f"claim ledger {claim_id} missing field {field}")
        evidence_ids = ledger.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids or not all(_nonempty_string(item) for item in evidence_ids):
            errors.append(f"claim ledger {claim_id} must contain concrete evidence_ids")
        else:
            for evidence_id in evidence_ids:
                if evidence_id not in evidence_by_id:
                    errors.append(f"claim ledger {claim_id} references unresolved evidence: {evidence_id}")
        if ledger.get("evidence_state") not in evidence_states:
            errors.append(f"claim ledger {claim_id} has invalid evidence state")
        if not isinstance(ledger.get("blocking_gaps"), list):
            errors.append(f"claim ledger {claim_id} blocking_gaps must be a list")

    decisions = _records(instances, "publication_decisions", errors)
    for decision in decisions:
        decision_id = decision.get("decision_id", "<unknown>")
        readiness = decision.get("readiness")
        if readiness not in readiness_vocabulary:
            errors.append(f"publication decision {decision_id} has invalid readiness: {readiness}")
        claim_ids = decision.get("claim_ids", [])
        if not isinstance(claim_ids, list) or not claim_ids or not all(claim_id in ledger_by_id for claim_id in claim_ids):
            errors.append(f"publication decision {decision_id} has unresolved claim references")
        gaps = decision.get("blocking_evidence_gaps", [])
        if not isinstance(gaps, list):
            errors.append(f"publication decision {decision_id} blocking_evidence_gaps must be a list")
            gaps = ["<invalid>"]
        referenced_ledgers = [ledger_by_id[claim_id] for claim_id in claim_ids if claim_id in ledger_by_id]
        unresolved = any(
            ledger.get("evidence_state") in NON_VERIFYING_EVIDENCE_STATES
            or bool(ledger.get("blocking_gaps"))
            or any(evidence_by_id[evidence_id].get("evidence_state") in NON_VERIFYING_EVIDENCE_STATES
                       for evidence_id in ledger.get("evidence_ids", []) if evidence_id in evidence_by_id)
            for ledger in referenced_ledgers
        )
        if readiness == "public_ready" and (gaps or unresolved):
            errors.append(f"public_ready decision has unresolved or blocked evidence: {decision_id}")

    for dissent in _records(instances, "explorer_dissents", errors):
        dissent_id = dissent.get("dissent_id", "<unknown>")
        if not _nonempty_string(dissent.get("preserved_artifact_ref")):
            errors.append(f"Explorer dissent {dissent_id} lacks preserved artifact reference")
        for field in EXPLORER_FIELDS:
            value = dissent.get(field)
            if value is None or (isinstance(value, (list, str, dict)) and not value):
                errors.append(f"Explorer dissent {dissent_id} has empty {field}")

    for chain in _records(instances, "fiv_chains", errors):
        chain_id = chain.get("chain_id", "<unknown>")
        nodes = chain.get("nodes")
        if not isinstance(nodes, list):
            nodes = [chain.get(kind) for kind in ("finding", "implementation", "verification")]
        if len(nodes) != 3 or any(not isinstance(node, dict) for node in nodes):
            errors.append(f"FIV chain {chain_id} must contain finding, implementation, and verification")
            continue
        for node, kind in zip(nodes, ("finding", "implementation", "verification")):
            if node.get("kind") != kind or not _nonempty_string(node.get("id")):
                errors.append(f"FIV chain {chain_id} has invalid {kind} reference")
            if node.get("evidence_state") not in evidence_states:
                errors.append(f"FIV chain {chain_id} {kind} has invalid evidence state")
            if kind == "verification" and node.get("evidence_state") in NON_VERIFYING_EVIDENCE_STATES:
                errors.append(f"FIV chain {chain_id} verification is not verifiable")
    return errors


validate_artifacts = validate_artifact_instances


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--artifacts", type=Path, help="also validate a concrete artifact-instance JSON document")
    args = parser.parse_args(argv)
    try:
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"registry unreadable: {exc}")
        return 2
    errors = validate_registry(registry)
    if args.artifacts:
        try:
            instances = json.loads(args.artifacts.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"artifact instances unreadable: {exc}")
            return 2
        errors.extend(validate_artifact_instances(registry, instances))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
