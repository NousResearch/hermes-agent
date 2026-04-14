#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: validate_skill_packaging_schema_pack.sh [--json]

Validate SYS-04 skill packaging schema/template pack, fixture packages, and SKILL.md frontmatter parity.
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

try:
    import yaml
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"dependency_missing:{exc}"}, indent=2))
    raise SystemExit(1)

schema_path = root / "docs/ops/schemas/skill_package_manifest.schema.json"
template_path = root / "docs/ops/templates/skill_package_manifest.template.json"
pack_path = root / "ops/openclaw/architecture/skill_packaging_schema_pack.v1.json"

errors = []
checks = []


def add_check(name: str, ok: bool, detail=None):
    row = {"name": name, "ok": ok}
    if detail is not None:
        row["detail"] = detail
    checks.append(row)


def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_frontmatter(path: pathlib.Path):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("missing_frontmatter_start")
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        raise ValueError("missing_frontmatter_end")
    return yaml.safe_load(parts[0].replace("---\n", "", 1))


for required in (schema_path, template_path, pack_path):
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
        pack = load_json(pack_path)
        fixture_packages = pack.get("pack", {}).get("fixture_packages", [])
        required_frontmatter = pack.get("pack", {}).get("frontmatter_required_fields", [])
        pointers = [
            pack.get("pack", {}).get("manifest_schema"),
            pack.get("pack", {}).get("manifest_template"),
        ] + fixture_packages

        for rel in pointers:
            if not rel:
                raise ValueError("missing_pack_pointer")
            resolved = root / rel
            if not resolved.exists():
                raise ValueError(f"broken_pack_pointer:{rel}")

        add_check("schema_pack_manifest_pointers", True)
    except Exception as exc:
        errors.append(f"schema_pack_manifest_pointers:{exc}")
        add_check("schema_pack_manifest_pointers", False, str(exc))
        fixture_packages = []
        required_frontmatter = []

    for rel in fixture_packages:
        pkg_path = root / rel
        try:
            manifest = load_json(pkg_path)
            jsonschema.validate(manifest, schema)

            package = manifest.get("package", {})
            root_dir = root / package.get("root_dir", "")
            skill_doc = root / package.get("skill_doc", "")
            scripts_dir = root / package.get("scripts_dir", "")
            references_dir = root / package.get("references_dir", "")

            for p in (root_dir, skill_doc, scripts_dir, references_dir):
                if not p.exists():
                    raise ValueError(f"missing_package_path:{p}")

            frontmatter = parse_frontmatter(skill_doc)
            if not isinstance(frontmatter, dict):
                raise ValueError("frontmatter_not_object")

            for field in required_frontmatter:
                if field not in frontmatter:
                    raise ValueError(f"frontmatter_missing_field:{field}")

            parity_fields = {
                "skill_id": manifest.get("skill_id"),
                "version": manifest.get("version"),
                "display_name": manifest.get("display_name"),
                "risk_class": (manifest.get("governance") or {}).get("risk_class"),
                "execution_mode": (manifest.get("runtime") or {}).get("execution_mode"),
                "manifest_path": rel,
            }
            for key, expected in parity_fields.items():
                got = frontmatter.get(key)
                if got != expected:
                    raise ValueError(f"frontmatter_parity_mismatch:{key}:{got}:{expected}")

            add_check(f"fixture_package_valid:{rel}", True)
        except Exception as exc:
            errors.append(f"fixture_package_valid:{rel}:{exc}")
            add_check(f"fixture_package_valid:{rel}", False, str(exc))

result = {
    "ok": len(errors) == 0,
    "checks": checks,
    "errors": errors,
    "schema_path": str(schema_path),
    "pack_path": str(pack_path),
}

if json_out:
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("SYS-04 SKILL PACKAGING SCHEMA PACK CHECK")
    print(f"- ok: {result['ok']}")
    print(f"- checks: {len(checks)}")
    print(f"- errors: {errors}")

if errors:
    raise SystemExit(1)
PY
