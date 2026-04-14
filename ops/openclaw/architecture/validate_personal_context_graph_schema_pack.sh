#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: validate_personal_context_graph_schema_pack.sh [--json]

Validate XP-302 personal context graph schema/template pack, fixtures, and provenance linkage rules.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_OUT=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

python3 - "$ROOT" "$JSON_OUT" <<'PY'
import datetime as dt
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
json_out = bool(int(sys.argv[2]))

try:
    import jsonschema
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"dependency_missing:{exc}"}, indent=2))
    raise SystemExit(1)

schema_path = root / "docs/ops/schemas/personal_context_graph_object.schema.json"
manifest_path = root / "ops/openclaw/architecture/personal_context_graph_schema_pack.v1.json"

errors = []
checks = []


def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def add_check(name: str, ok: bool, detail=None):
    row = {"name": name, "ok": ok}
    if detail is not None:
        row["detail"] = detail
    checks.append(row)


for required in (schema_path, manifest_path):
    if not required.exists():
        errors.append(f"missing_required:{required}")

if not errors:
    schema = load_json(schema_path)
    manifest = load_json(manifest_path)

    template_map = manifest.get("pack", {}).get("templates", {})
    fixture_paths = manifest.get("pack", {}).get("fixtures", [])

    # schema + manifest pointers
    try:
        pointer_paths = [manifest.get("pack", {}).get("graph_object_schema")]
        pointer_paths += list(template_map.values())
        pointer_paths += fixture_paths
        for rel in pointer_paths:
            if not rel:
                raise ValueError("missing_pack_pointer")
            resolved = root / rel
            if not resolved.exists():
                raise ValueError(f"broken_pack_pointer:{rel}")
        add_check("schema_pack_manifest_pointers", True)
    except Exception as exc:
        errors.append(f"schema_pack_manifest_pointers:{exc}")
        add_check("schema_pack_manifest_pointers", False, str(exc))

    # templates validate
    try:
        for object_type, rel in template_map.items():
            payload = load_json(root / rel)
            jsonschema.validate(payload, schema)
            if payload.get("object_type") != object_type:
                raise ValueError(f"template_object_type_mismatch:{object_type}:{payload.get('object_type')}")
        add_check("templates_vs_schema", True)
    except Exception as exc:
        errors.append(f"templates_vs_schema:{exc}")
        add_check("templates_vs_schema", False, str(exc))

    # fixtures validate
    fixture_records = []
    try:
        if not fixture_paths:
            raise ValueError("no_fixtures_declared")
        for rel in fixture_paths:
            fixture = load_json(root / rel)
            rows = fixture.get("records") or []
            if not rows:
                raise ValueError(f"fixture_has_no_records:{rel}")
            for row in rows:
                jsonschema.validate(row, schema)
                fixture_records.append(row)
        add_check("fixture_records_vs_schema", True)
    except Exception as exc:
        errors.append(f"fixture_records_vs_schema:{exc}")
        add_check("fixture_records_vs_schema", False, str(exc))

    if fixture_records:
        required_types = {
            "goal",
            "routine",
            "constraint",
            "event",
            "commitment",
            "decision_record",
            "after_action_review",
            "lesson_card",
            "pattern_card",
        }

        by_id = {row["object_id"]: row for row in fixture_records}

        # coverage check
        try:
            present_types = {row.get("object_type") for row in fixture_records}
            missing = sorted(required_types - present_types)
            if missing:
                raise ValueError(f"missing_required_object_types:{missing}")
            add_check("fixture_required_object_types", True)
        except Exception as exc:
            errors.append(f"fixture_required_object_types:{exc}")
            add_check("fixture_required_object_types", False, str(exc))

        # revision/update policy linkage checks
        try:
            seen_hashes = set()
            for row in fixture_records:
                h = row.get("object_hash")
                if h in seen_hashes:
                    raise ValueError(f"duplicate_object_hash:{h}")
                seen_hashes.add(h)

                revision = row.get("revision")
                sup = row.get("supersedes_object_id")
                prev_hash = row.get("previous_object_hash")

                if revision == 1:
                    if sup is not None or prev_hash is not None:
                        raise ValueError(f"revision1_must_not_supersede:{row.get('object_id')}")
                else:
                    if not sup or not prev_hash:
                        raise ValueError(f"revision_ge2_requires_supersede_fields:{row.get('object_id')}")
                    parent = by_id.get(sup)
                    if not parent:
                        raise ValueError(f"missing_superseded_object:{sup}")
                    if parent.get("object_hash") != prev_hash:
                        raise ValueError(
                            f"previous_hash_mismatch:{row.get('object_id')}:{prev_hash}:{parent.get('object_hash')}"
                        )
                    child_ts = dt.datetime.fromisoformat(str(row.get("created_at")).replace("Z", "+00:00"))
                    parent_ts = dt.datetime.fromisoformat(str(parent.get("created_at")).replace("Z", "+00:00"))
                    if child_ts < parent_ts:
                        raise ValueError(f"superseding_created_at_regression:{row.get('object_id')}")
            add_check("revision_update_policy_checks", True)
        except Exception as exc:
            errors.append(f"revision_update_policy_checks:{exc}")
            add_check("revision_update_policy_checks", False, str(exc))

        # decision/review/learning linkage checks
        try:
            for row in fixture_records:
                object_type = row.get("object_type")
                payload = row.get("payload") or {}
                links = row.get("links") or {}

                if object_type == "after_action_review":
                    decision_id = payload.get("decision_record_object_id")
                    decision = by_id.get(decision_id)
                    if not decision or decision.get("object_type") != "decision_record":
                        raise ValueError(f"aar_missing_decision_record:{row.get('object_id')}:{decision_id}")

                if object_type == "lesson_card":
                    for ref in payload.get("derived_from_object_ids", []):
                        if ref not in by_id:
                            raise ValueError(f"lesson_missing_derived_ref:{row.get('object_id')}:{ref}")

                if object_type == "pattern_card":
                    for ref in payload.get("linked_lesson_object_ids", []):
                        target = by_id.get(ref)
                        if not target or target.get("object_type") != "lesson_card":
                            raise ValueError(f"pattern_missing_lesson_ref:{row.get('object_id')}:{ref}")
                    for ref in payload.get("exemplar_object_ids", []):
                        if ref not in by_id:
                            raise ValueError(f"pattern_missing_exemplar_ref:{row.get('object_id')}:{ref}")

                for related in links.get("related_object_ids", []):
                    if related not in by_id:
                        raise ValueError(f"missing_related_object_ref:{row.get('object_id')}:{related}")

            add_check("decision_review_learning_linkage", True)
        except Exception as exc:
            errors.append(f"decision_review_learning_linkage:{exc}")
            add_check("decision_review_learning_linkage", False, str(exc))

        # provenance compatibility with XB + XP risk alignment
        try:
            risk_map = {
                "PX0_INFO": "RG0_LOW",
                "PX1_ASSIST": "RG1_MODERATE",
                "PX2_HIGH_IMPACT": "RG2_HIGH",
                "PX3_SAFETY_CRITICAL": "RG3_CRITICAL",
            }
            for row in fixture_records:
                governance = row.get("governance") or {}
                provenance = row.get("provenance") or {}
                expected_risk = risk_map.get(governance.get("risk_tier"))
                if expected_risk is None:
                    raise ValueError(f"unknown_risk_tier:{row.get('object_id')}:{governance.get('risk_tier')}")
                if provenance.get("risk_class") != expected_risk:
                    raise ValueError(
                        f"risk_alignment_mismatch:{row.get('object_id')}:{provenance.get('risk_class')}:{expected_risk}"
                    )
                if provenance.get("route_class") != "advisory":
                    raise ValueError(f"route_class_mismatch:{row.get('object_id')}")
                if governance.get("advisory_only") is not True:
                    raise ValueError(f"advisory_only_mismatch:{row.get('object_id')}")
            add_check("provenance_risk_alignment", True)
        except Exception as exc:
            errors.append(f"provenance_risk_alignment:{exc}")
            add_check("provenance_risk_alignment", False, str(exc))

result = {
    "ok": len(errors) == 0,
    "checks": checks,
    "errors": errors,
    "schema_path": str(schema_path),
    "manifest_path": str(manifest_path),
}

if json_out:
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("XP-302 PERSONAL CONTEXT GRAPH SCHEMA PACK CHECK")
    print(f"- ok: {result['ok']}")
    print(f"- checks: {len(checks)}")
    print(f"- errors: {errors}")

if errors:
    raise SystemExit(1)
PY
