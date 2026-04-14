#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: validate_health_typed_record_schema_pack.sh [--json]

Validate XH-702 health typed-record schema/template pack and privacy constraint fixtures.
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

schema_path = root / "docs/ops/schemas/health_typed_record.schema.json"
template_path = root / "docs/ops/templates/health_typed_record.template.json"
fixture_path = root / "tests/fixtures/xh/health_typed_record_fixture_v1.json"
pack_fixture_path = root / "tests/fixtures/xh/health_typed_record_pack_fixture_v1.json"
privacy_fixture_path = root / "tests/fixtures/xh/health_privacy_constraint_fixture_v1.json"
pack_manifest_path = root / "ops/openclaw/architecture/health_typed_record_schema_pack.v1.json"

errors = []
checks = []


def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def add_check(name: str, ok: bool, detail=None):
    row = {"name": name, "ok": ok}
    if detail is not None:
        row["detail"] = detail
    checks.append(row)


for required in (
    schema_path,
    template_path,
    fixture_path,
    pack_fixture_path,
    privacy_fixture_path,
    pack_manifest_path,
):
    if not required.exists():
        errors.append(f"missing_required:{required}")

if not errors:
    schema = load_json(schema_path)

    try:
        jsonschema.validate(load_json(template_path), schema)
        add_check("template_vs_schema", True)
    except Exception as exc:
        errors.append(f"template_vs_schema:{exc}")
        add_check("template_vs_schema", False, str(exc))

    try:
        jsonschema.validate(load_json(fixture_path), schema)
        add_check("fixture_vs_schema", True)
    except Exception as exc:
        errors.append(f"fixture_vs_schema:{exc}")
        add_check("fixture_vs_schema", False, str(exc))

    try:
        pack = load_json(pack_fixture_path)
        if pack.get("schema") != "clawd.xh_702.health_typed_record_pack_fixture.v1":
            raise ValueError("pack fixture schema tag mismatch")
        records = pack.get("records") or []
        if len(records) < 4:
            raise ValueError("pack fixture requires at least 4 typed records")

        required_types = {"measurement", "lab_result", "symptom", "protocol"}
        observed_types = {row.get("record_type") for row in records}
        missing_types = sorted(required_types - observed_types)
        if missing_types:
            raise ValueError(f"missing record types in pack fixture: {missing_types}")

        for row in records:
            jsonschema.validate(row, schema)

        add_check("typed_record_pack_fixture", True)
    except Exception as exc:
        errors.append(f"typed_record_pack_fixture:{exc}")
        add_check("typed_record_pack_fixture", False, str(exc))

    try:
        privacy_fixture = load_json(privacy_fixture_path)
        if privacy_fixture.get("schema") != "clawd.xh_702.privacy_constraint_fixture.v1":
            raise ValueError("privacy fixture schema tag mismatch")

        cases = privacy_fixture.get("cases") or []
        if not cases:
            raise ValueError("privacy fixture must include cases")

        case_results = []
        for case in cases:
            case_id = case.get("case_id")
            expect = case.get("expect")
            record = case.get("record")

            ok = False
            error_msg = None
            try:
                jsonschema.validate(record, schema)
                ok = True
            except Exception as exc:
                error_msg = str(exc)
                ok = False

            if expect == "pass" and not ok:
                raise ValueError(f"case {case_id} expected PASS but failed: {error_msg}")
            if expect == "fail" and ok:
                raise ValueError(f"case {case_id} expected FAIL but passed")

            case_results.append({
                "case_id": case_id,
                "expect": expect,
                "observed": "pass" if ok else "fail",
                "status": "pass",
            })

        add_check("privacy_constraint_cases", True, {"case_count": len(case_results)})
    except Exception as exc:
        errors.append(f"privacy_constraint_cases:{exc}")
        add_check("privacy_constraint_cases", False, str(exc))

    try:
        manifest = load_json(pack_manifest_path)
        pointers = [
            manifest.get("pack", {}).get("typed_record", {}).get("schema"),
            manifest.get("pack", {}).get("typed_record", {}).get("template"),
        ] + (manifest.get("pack", {}).get("typed_record", {}).get("fixtures") or [])

        if not pointers:
            raise ValueError("empty schema pack manifest pointers")

        for rel in pointers:
            if not rel:
                raise ValueError("missing schema pack pointer value")
            resolved = root / rel
            if not resolved.exists():
                raise ValueError(f"broken_pack_pointer:{rel}")

        add_check("schema_pack_manifest_pointers", True)
    except Exception as exc:
        errors.append(f"schema_pack_manifest_pointers:{exc}")
        add_check("schema_pack_manifest_pointers", False, str(exc))

result = {
    "ok": len(errors) == 0,
    "checks": checks,
    "errors": errors,
    "pack_path": str(pack_manifest_path),
}

if json_out:
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("XH-702 HEALTH TYPED RECORD SCHEMA PACK CHECK")
    print(f"- ok: {result['ok']}")
    print(f"- checks: {len(checks)}")
    print(f"- errors: {errors}")

if errors:
    raise SystemExit(1)
PY
