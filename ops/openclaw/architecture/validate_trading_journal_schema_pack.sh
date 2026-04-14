#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: validate_trading_journal_schema_pack.sh [--json]

Validate XT-602 trading journal schema/template pack and append-only fixture chain.
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

schema_path = root / "docs/ops/schemas/trading_journal_entry.schema.json"
template_path = root / "docs/ops/templates/trading_journal_entry.template.json"
fixture_path = root / "tests/fixtures/xt/trading_journal_entry_fixture_v1.json"
chain_path = root / "tests/fixtures/xt/trading_journal_append_only_chain_fixture_v1.json"
pack_path = root / "ops/openclaw/architecture/trading_journal_schema_pack.v1.json"

errors = []
checks = []


def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def add_check(name: str, ok: bool, detail=None):
    row = {"name": name, "ok": ok}
    if detail is not None:
        row["detail"] = detail
    checks.append(row)

for required in (schema_path, template_path, fixture_path, chain_path, pack_path):
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
        chain = load_json(chain_path)
        records = chain.get("records") or []
        if chain.get("schema") != "clawd.xt_602.append_only_chain_fixture.v1":
            raise ValueError("chain schema tag mismatch")
        if len(records) < 2:
            raise ValueError("append-only chain must include >=2 revisions")

        revisions = [row.get("revision") for row in records]
        if revisions != sorted(revisions) or revisions != list(range(1, len(records) + 1)):
            raise ValueError("revision sequence must be monotonic and contiguous from 1")

        prev = None
        hashes = set()
        for row in records:
            jsonschema.validate(row, schema)
            h = row.get("entry_hash")
            if h in hashes:
                raise ValueError("duplicate entry_hash in append-only chain")
            hashes.add(h)

            refs = ((row.get("review") or {}).get("review_refs")) or []
            if not refs:
                raise ValueError("review.review_refs must be non-empty")

            if prev is None:
                if row.get("supersedes_entry_id") is not None or row.get("previous_entry_hash") is not None:
                    raise ValueError("revision=1 must not supersede prior entries")
            else:
                if row.get("supersedes_entry_id") != prev.get("entry_id"):
                    raise ValueError("supersedes_entry_id must point to immediate previous entry")
                if row.get("previous_entry_hash") != prev.get("entry_hash"):
                    raise ValueError("previous_entry_hash must equal previous entry_hash")
                ts_now = dt.datetime.fromisoformat(str(row.get("created_at")).replace("Z", "+00:00"))
                ts_prev = dt.datetime.fromisoformat(str(prev.get("created_at")).replace("Z", "+00:00"))
                if ts_now < ts_prev:
                    raise ValueError("created_at must be non-decreasing across revisions")

            prev = row

        add_check("append_only_chain_checks", True)
    except Exception as exc:
        errors.append(f"append_only_chain_checks:{exc}")
        add_check("append_only_chain_checks", False, str(exc))

    try:
        pack = load_json(pack_path)
        pointers = [
            pack.get("pack", {}).get("entry", {}).get("schema"),
            pack.get("pack", {}).get("entry", {}).get("template"),
        ] + (pack.get("pack", {}).get("entry", {}).get("fixtures") or [])

        if not pointers:
            raise ValueError("empty pack pointers")

        for rel in pointers:
            if not rel:
                raise ValueError("missing pack pointer value")
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
    "pack_path": str(pack_path),
}

if json_out:
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("XT-602 TRADING JOURNAL SCHEMA PACK CHECK")
    print(f"- ok: {result['ok']}")
    print(f"- checks: {len(checks)}")
    print(f"- errors: {errors}")

if errors:
    raise SystemExit(1)
PY
