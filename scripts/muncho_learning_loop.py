#!/usr/bin/env python3
"""Report-only Muncho operational learning packet tooling.

This script is intentionally outside the Hermes runtime hot path. It creates
reviewable learning packet drafts from explicitly provided evidence summaries
and emits digest reports. It does not mutate skills, memory, gateway config,
Cloud SQL, or any durable knowledge base.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "muncho.learning_packet.v1"
DIGEST_SCHEMA_VERSION = "muncho.learning_digest.v1"

KNOWLEDGE_CLASSES = {
    "case_note_only",
    "business_knowledge",
    "operational_process",
    "team_routing_knowledge",
    "it_access_or_infra",
    "customer_support_pattern",
    "tooling_gap",
    "product_process_improvement",
    "skill_update_candidate",
    "reject",
}

PROMOTION_RECOMMENDATIONS = {
    "case_note_only",
    "pattern_candidate",
    "skill_update_candidate",
    "runbook_update_candidate",
    "durable_knowledge_candidate",
    "access_map_update_candidate",
    "team_channel_map_update_candidate",
    "process_recommendation_candidate",
    "reject",
}

CONFIDENCE_LEVELS = {"low", "medium", "high"}

AUTO_PROMOTION_FORBIDDEN_FOR = {
    "team_channel_routing",
    "access_maps",
    "deploy_runtime_behavior",
    "cloudsql_write_contracts",
    "customer_provider_processes",
    "permissions",
    "production_operations",
}

AUTO_SELF_IMPROVEMENT_ALLOWED_FOR = {
    "workflow_phrasing",
    "report_formatting",
    "evidence_collection_hygiene",
    "runbook_clarity",
    "local_skill_hygiene",
}

REQUIRED_CASE_FIELDS = {
    "case_id",
    "source_refs",
    "requester",
    "involved_people",
    "business_area",
    "knowledge_classes",
    "problem",
    "expected_action",
    "actual_muncho_action",
    "what_went_wrong",
    "what_worked",
    "final_status",
    "evidence_refs",
    "lesson_candidate",
    "promotion_recommendation",
    "confidence",
    "missing_evidence",
}

SECRETISH_RE = re.compile(
    r"(?i)("
    r"api[_-]?key|authorization|bearer\s+[a-z0-9._-]+|password|secret|token"
    r"|-----BEGIN\s+(?:RSA|OPENSSH|PRIVATE)\s+KEY-----"
    r")"
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def stable_packet_id(case_id: str, source_refs: list[dict[str, Any]]) -> str:
    seed = json.dumps(
        {"case_id": case_id, "source_refs": source_refs},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"learning:{digest}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-").lower()
    return slug or "packet"


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def scan_for_secretish_values(value: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if SECRETISH_RE.search(key_text):
                hits.append(child_path)
            hits.extend(scan_for_secretish_values(child, child_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            hits.extend(scan_for_secretish_values(child, f"{path}[{idx}]"))
    elif isinstance(value, str) and SECRETISH_RE.search(value):
        hits.append(path)
    return hits


def validate_source_refs(source_refs: Any) -> None:
    if not isinstance(source_refs, list) or not source_refs:
        raise ValueError("source_refs must be a non-empty list")
    for idx, ref in enumerate(source_refs):
        if not isinstance(ref, dict):
            raise ValueError(f"source_refs[{idx}] must be an object")
        ref_type = ref.get("type")
        if not ref_type:
            raise ValueError(f"source_refs[{idx}].type is required")
        if not any(ref.get(key) for key in ("thread_id", "message_id", "manual_ref", "event_ref", "report_path")):
            raise ValueError(
                f"source_refs[{idx}] requires one of thread_id, message_id, "
                "manual_ref, event_ref, or report_path"
            )


def validate_case_draft(case: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_CASE_FIELDS - set(case))
    if missing:
        raise ValueError(f"case draft missing required fields: {', '.join(missing)}")

    validate_source_refs(case["source_refs"])
    validate_source_refs(case["evidence_refs"])

    classes = case.get("knowledge_classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("knowledge_classes must be a non-empty list")
    invalid_classes = sorted(set(classes) - KNOWLEDGE_CLASSES)
    if invalid_classes:
        raise ValueError(f"unknown knowledge_classes: {', '.join(invalid_classes)}")

    recommendation = case.get("promotion_recommendation")
    if recommendation not in PROMOTION_RECOMMENDATIONS:
        raise ValueError(f"unknown promotion_recommendation: {recommendation}")

    confidence = case.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(f"unknown confidence: {confidence}")

    secretish_hits = scan_for_secretish_values(case)
    if secretish_hits:
        preview = ", ".join(secretish_hits[:5])
        raise ValueError(f"case draft contains secret-like keys or values at: {preview}")


def packet_from_case(case: dict[str, Any], created_at: str | None = None) -> dict[str, Any]:
    validate_case_draft(case)
    created = created_at or now_iso()
    packet = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": stable_packet_id(case["case_id"], case["source_refs"]),
        "created_at": created,
        "status": "draft_report_only",
        "case_id": case["case_id"],
        "source_refs": case["source_refs"],
        "requester": case["requester"],
        "involved_people": case["involved_people"],
        "business_area": case["business_area"],
        "knowledge_classes": case["knowledge_classes"],
        "problem": case["problem"],
        "expected_action": case["expected_action"],
        "actual_muncho_action": case["actual_muncho_action"],
        "what_went_wrong": case["what_went_wrong"],
        "what_worked": case["what_worked"],
        "final_status": case["final_status"],
        "evidence_refs": case["evidence_refs"],
        "lesson_candidate": case["lesson_candidate"],
        "promotion_recommendation": case["promotion_recommendation"],
        "confidence": case["confidence"],
        "missing_evidence": case["missing_evidence"],
        "safety": {
            "report_only": True,
            "runtime_behavior_change": False,
            "durable_promotion_performed": False,
            "standard_hermes_self_improvement_preserved": True,
            "keyword_router_authority": False,
            "auto_promotion_forbidden_for": sorted(AUTO_PROMOTION_FORBIDDEN_FOR),
            "auto_self_improvement_allowed_for": sorted(AUTO_SELF_IMPROVEMENT_ALLOWED_FOR),
        },
        "promotion": {
            "requires_explicit_owner_approval": True,
            "allowed_next_statuses": [
                "case_note_only",
                "pattern_candidate",
                "skill_update_candidate",
                "runbook_update_candidate",
                "durable_knowledge_candidate",
                "reject",
            ],
            "performed": False,
        },
    }
    secretish_hits = scan_for_secretish_values(packet)
    if secretish_hits:
        preview = ", ".join(secretish_hits[:5])
        raise ValueError(f"packet contains secret-like keys or values at: {preview}")
    return packet


def validate_packet(packet: dict[str, Any]) -> None:
    if packet.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid schema_version")
    if packet.get("status") != "draft_report_only":
        raise ValueError("packet must remain draft_report_only")
    safety = packet.get("safety") or {}
    required_safety = {
        "report_only": True,
        "runtime_behavior_change": False,
        "durable_promotion_performed": False,
        "standard_hermes_self_improvement_preserved": True,
        "keyword_router_authority": False,
    }
    for key, expected in required_safety.items():
        if safety.get(key) is not expected:
            raise ValueError(f"safety.{key} must be {expected!r}")
    validate_case_draft({key: packet[key] for key in REQUIRED_CASE_FIELDS})


def load_packets(packet_dir: Path) -> list[dict[str, Any]]:
    packets = []
    for path in sorted(packet_dir.glob("*.json")):
        if path.name.startswith("digest-"):
            continue
        data = load_json(path)
        validate_packet(data)
        packets.append(data)
    return packets


def build_digest(packets: list[dict[str, Any]], created_at: str | None = None) -> dict[str, Any]:
    class_counter: Counter[str] = Counter()
    recommendations: Counter[str] = Counter()
    for packet in packets:
        class_counter.update(packet.get("knowledge_classes", []))
        recommendations.update([packet.get("promotion_recommendation", "unknown")])
    return {
        "schema_version": DIGEST_SCHEMA_VERSION,
        "created_at": created_at or now_iso(),
        "packet_count": len(packets),
        "by_knowledge_class": dict(sorted(class_counter.items())),
        "by_promotion_recommendation": dict(sorted(recommendations.items())),
        "promotion_queue": [
            {
                "packet_id": packet["packet_id"],
                "case_id": packet["case_id"],
                "recommendation": packet["promotion_recommendation"],
                "confidence": packet["confidence"],
            }
            for packet in packets
            if packet["promotion_recommendation"] not in {"case_note_only", "reject"}
        ],
        "boundaries": {
            "runtime_behavior_change": False,
            "durable_promotion_performed": False,
            "standard_hermes_self_improvement_preserved": True,
            "keyword_router_authority": False,
        },
    }


def render_private_digest(digest: dict[str, Any], packets: list[dict[str, Any]]) -> str:
    lines = [
        "# Muncho Learning Loop Digest",
        "",
        f"Generated: {digest['created_at']}",
        f"Packets: {digest['packet_count']}",
        "",
        "## Boundaries",
        "",
        "- Report-only: true",
        "- Runtime behavior changed: false",
        "- Durable promotion performed: false",
        "- Standard Hermes self-improvement preserved: true",
        "- Keyword/router authority introduced: false",
        "",
        "## Packets",
        "",
    ]
    for packet in packets:
        lines.extend(
            [
                f"### {packet['case_id']}",
                "",
                f"- Packet: `{packet['packet_id']}`",
                f"- Business area: {packet['business_area']}",
                f"- Classes: {', '.join(packet['knowledge_classes'])}",
                f"- Recommendation: {packet['promotion_recommendation']}",
                f"- Confidence: {packet['confidence']}",
                f"- Lesson candidate: {packet['lesson_candidate']}",
                f"- Missing evidence: {packet['missing_evidence'] or 'none'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_public_digest(digest: dict[str, Any]) -> str:
    lines = [
        "# Muncho Learning Loop Public-Safe Digest",
        "",
        f"Generated: {digest['created_at']}",
        f"Packets: {digest['packet_count']}",
        "",
        "## Counts",
        "",
    ]
    for key, value in digest["by_knowledge_class"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- No raw case transcripts included.",
            "- No customer/provider/private identifiers included.",
            "- No runtime behavior change.",
            "- No durable knowledge promotion.",
            "- Standard Hermes self-improvement remains untouched.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Muncho Learning Packet",
        "type": "object",
        "required": [
            "schema_version",
            "packet_id",
            "created_at",
            "status",
            *sorted(REQUIRED_CASE_FIELDS),
            "safety",
            "promotion",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "status": {"const": "draft_report_only"},
            "knowledge_classes": {
                "type": "array",
                "items": {"enum": sorted(KNOWLEDGE_CLASSES)},
                "minItems": 1,
            },
            "promotion_recommendation": {"enum": sorted(PROMOTION_RECOMMENDATIONS)},
            "confidence": {"enum": sorted(CONFIDENCE_LEVELS)},
            "safety": {
                "type": "object",
                "properties": {
                    "report_only": {"const": True},
                    "runtime_behavior_change": {"const": False},
                    "durable_promotion_performed": {"const": False},
                    "standard_hermes_self_improvement_preserved": {"const": True},
                    "keyword_router_authority": {"const": False},
                },
            },
        },
    }


def cmd_schema(args: argparse.Namespace) -> int:
    schema = build_schema()
    if args.output:
        write_json(args.output, schema)
    else:
        print(json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    data = load_json(args.input)
    if isinstance(data, list):
        for item in data:
            validate_case_draft(item)
    else:
        if data.get("schema_version") == SCHEMA_VERSION:
            validate_packet(data)
        else:
            validate_case_draft(data)
    print("VERDICT: PASS")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    drafts = load_json(args.input)
    if not isinstance(drafts, list):
        raise ValueError("generate input must be a list of case drafts")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    packets = [packet_from_case(case) for case in drafts]
    for packet in packets:
        filename = f"{slugify(packet['case_id'])}.{packet['packet_id'].split(':', 1)[1]}.json"
        write_json(output_dir / filename, packet)

    digest = build_digest(packets)
    write_json(output_dir / "digest-private.json", digest)
    (output_dir / "digest-private.md").write_text(render_private_digest(digest, packets), encoding="utf-8")
    (output_dir / "digest-public-safe.md").write_text(render_public_digest(digest), encoding="utf-8")
    print("VERDICT: PASS")
    print(f"Packets: {len(packets)}")
    print(f"Output: {output_dir}")
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    packets = load_packets(args.packet_dir.expanduser())
    digest = build_digest(packets)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "digest-private.json", digest)
    (output_dir / "digest-private.md").write_text(render_private_digest(digest, packets), encoding="utf-8")
    (output_dir / "digest-public-safe.md").write_text(render_public_digest(digest), encoding="utf-8")
    print("VERDICT: PASS")
    print(f"Packets: {len(packets)}")
    print(f"Output: {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Muncho report-only operational learning loop")
    sub = parser.add_subparsers(dest="command", required=True)

    schema = sub.add_parser("schema", help="Print or write the learning packet schema")
    schema.add_argument("--output", type=Path)
    schema.set_defaults(func=cmd_schema)

    validate = sub.add_parser("validate", help="Validate a case draft or packet")
    validate.add_argument("--input", type=Path, required=True)
    validate.set_defaults(func=cmd_validate)

    generate = sub.add_parser("generate", help="Generate report-only packets from private case drafts")
    generate.add_argument("--input", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.set_defaults(func=cmd_generate)

    digest = sub.add_parser("digest", help="Generate digest reports from packet JSON files")
    digest.add_argument("--packet-dir", type=Path, required=True)
    digest.add_argument("--output-dir", type=Path, required=True)
    digest.set_defaults(func=cmd_digest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"VERDICT: FAIL\nReason: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
