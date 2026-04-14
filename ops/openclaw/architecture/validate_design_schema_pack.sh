#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: validate_design_schema_pack.sh [--json]

Validate XD-102 design schema/template pack and representative fixtures.
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


def to_json_safe(value):
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value

try:
    import jsonschema
    import yaml
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"dependency_missing:{exc}"}, indent=2))
    raise SystemExit(1)


def load_frontmatter(md_path: pathlib.Path) -> dict:
    text = md_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"missing_frontmatter_start:{md_path}")
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        raise ValueError(f"missing_frontmatter_end:{md_path}")
    return yaml.safe_load(parts[0].replace("---\n", "", 1))

checks = [
    {
        "name": "token_template_vs_token_schema",
        "payload": root / "ops/openclaw/architecture/templates/design_token_registry.template.json",
        "schema": root / "ops/openclaw/architecture/schemas/design_token_registry.schema.json",
        "loader": "json",
    },
    {
        "name": "token_foundation_vs_token_schema",
        "payload": root / "ops/openclaw/architecture/design_token_registry.foundation.v1.json",
        "schema": root / "ops/openclaw/architecture/schemas/design_token_registry.schema.json",
        "loader": "json",
    },
    {
        "name": "component_template_vs_component_schema",
        "payload": root / "ops/openclaw/architecture/templates/component_spec_template.md",
        "schema": root / "ops/openclaw/architecture/schemas/design_component_spec_frontmatter.schema.json",
        "loader": "frontmatter",
    },
    {
        "name": "interaction_template_vs_interaction_schema",
        "payload": root / "ops/openclaw/architecture/templates/design_interaction_contract.template.yaml",
        "schema": root / "ops/openclaw/architecture/schemas/design_interaction_contract.schema.json",
        "loader": "yaml",
    },
    {
        "name": "token_fixture_vs_token_schema",
        "payload": root / "tests/fixtures/xd/token_registry_fixture_v1.json",
        "schema": root / "ops/openclaw/architecture/schemas/design_token_registry.schema.json",
        "loader": "json",
    },
    {
        "name": "component_fixture_vs_component_schema",
        "payload": root / "tests/fixtures/xd/component_spec_fixture_v1.md",
        "schema": root / "ops/openclaw/architecture/schemas/design_component_spec_frontmatter.schema.json",
        "loader": "frontmatter",
    },
    {
        "name": "interaction_fixture_vs_interaction_schema",
        "payload": root / "tests/fixtures/xd/interaction_contract_fixture_v1.yaml",
        "schema": root / "ops/openclaw/architecture/schemas/design_interaction_contract.schema.json",
        "loader": "yaml",
    },
]

results = []
errors = []
for check in checks:
    payload_path = check["payload"]
    schema_path = check["schema"]
    if not payload_path.exists():
        errors.append(f"missing_payload:{payload_path}")
        results.append({"name": check["name"], "ok": False, "error": f"missing_payload:{payload_path}"})
        continue
    if not schema_path.exists():
        errors.append(f"missing_schema:{schema_path}")
        results.append({"name": check["name"], "ok": False, "error": f"missing_schema:{schema_path}"})
        continue

    try:
        if check["loader"] == "json":
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        elif check["loader"] == "yaml":
            payload = to_json_safe(yaml.safe_load(payload_path.read_text(encoding="utf-8")))
        elif check["loader"] == "frontmatter":
            payload = to_json_safe(load_frontmatter(payload_path))
        else:
            raise ValueError(f"unknown_loader:{check['loader']}")

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(payload, schema)
        results.append({"name": check["name"], "ok": True})
    except Exception as exc:
        msg = f"{check['name']}:{exc}"
        errors.append(msg)
        results.append({"name": check["name"], "ok": False, "error": str(exc)})

pack_path = root / "ops/openclaw/architecture/design_schema_pack.v1.json"
if pack_path.exists():
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    for key in ("token_registry", "component_spec_frontmatter", "interaction_contract"):
        for pointer_field in ("schema", "template"):
            rel = pack.get("pack", {}).get(key, {}).get(pointer_field)
            if not rel:
                errors.append(f"missing_pack_pointer:{key}.{pointer_field}")
                continue
            resolved = root / rel
            if not resolved.exists():
                errors.append(f"broken_pack_pointer:{key}.{pointer_field}:{rel}")

result = {
    "ok": len(errors) == 0,
    "checks": results,
    "errors": errors,
    "pack_path": str(pack_path),
}

if json_out:
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("XD-102 DESIGN SCHEMA PACK CHECK")
    print(f"- ok: {result['ok']}")
    print(f"- checks: {len(results)}")
    print(f"- errors: {errors}")

if errors:
    raise SystemExit(1)
PY
