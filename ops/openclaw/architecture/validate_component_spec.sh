#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
SPEC_PATH="$ROOT/ops/openclaw/architecture/templates/component_spec_template.md"
SCHEMA_PATH="$ROOT/ops/openclaw/architecture/schemas/design_component_spec_frontmatter.schema.json"
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: validate_component_spec.sh [options]

Validate component spec markdown frontmatter against canonical design component schema.

Options:
  --spec <path>     Markdown spec path (default: canonical template)
  --schema <path>   Frontmatter schema path (default: canonical schema)
  --json            JSON output
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spec)
      SPEC_PATH="${2:-}"; shift 2 ;;
    --schema)
      SCHEMA_PATH="${2:-}"; shift 2 ;;
    --json)
      JSON_OUT=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

python3 - "$SPEC_PATH" "$SCHEMA_PATH" "$JSON_OUT" <<'PY'
import datetime as dt
import json
import pathlib
import sys

spec_path = pathlib.Path(sys.argv[1]).resolve()
schema_path = pathlib.Path(sys.argv[2]).resolve()
json_out = bool(int(sys.argv[3]))


def to_json_safe(value):
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value

try:
    import yaml
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"pyyaml_missing:{exc}"}, ensure_ascii=False, indent=2))
    raise SystemExit(1)

try:
    import jsonschema
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"jsonschema_missing:{exc}"}, ensure_ascii=False, indent=2))
    raise SystemExit(1)

errors = []
if not spec_path.exists():
    errors.append(f"missing_spec:{spec_path}")
if not schema_path.exists():
    errors.append(f"missing_schema:{schema_path}")

frontmatter = None
if not errors:
    text = spec_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        errors.append("missing_frontmatter_start")
    else:
        parts = text.split("\n---\n", 1)
        if len(parts) != 2:
            errors.append("missing_frontmatter_end")
        else:
            frontmatter = to_json_safe(yaml.safe_load(parts[0].replace("---\n", "", 1)))

if not errors:
    try:
        schema_obj = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(frontmatter, schema_obj)
    except Exception as exc:
        errors.append(str(exc))

result = {
    "ok": len(errors) == 0,
    "spec_path": str(spec_path),
    "schema_path": str(schema_path),
    "errors": errors,
}

if json_out:
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("DESIGN COMPONENT SPEC CHECK")
    print(f"- ok: {result['ok']}")
    print(f"- spec: {result['spec_path']}")
    print(f"- schema: {result['schema_path']}")
    if errors:
        print(f"- errors: {errors}")

if errors:
    raise SystemExit(1)
PY
