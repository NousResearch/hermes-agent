#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python_bin=""
for candidate in \
  "$ROOT/.venv/bin/python" \
  "$ROOT/venv/bin/python" \
  "$HOME/.hermes/hermes-agent/.venv/bin/python" \
  "$HOME/.hermes/hermes-agent/venv/bin/python" \
  "$(command -v python3 || true)"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    python_bin="$candidate"
    break
  fi
done

if [ -z "$python_bin" ]; then
  echo "error: no Python interpreter found for smoke checks" >&2
  exit 1
fi

runtime_root="${HERMES_FACTORY_RUNTIME_DIR:-}"
if [ -n "$runtime_root" ]; then
  mkdir -p "$runtime_root"
  smoke_home="$(mktemp -d "$runtime_root/hermes-smoke.XXXXXX")"
else
  smoke_home="$(mktemp -d "${TMPDIR:-/tmp}/hermes-smoke.XXXXXX")"
fi
cleanup() {
  rm -rf "$smoke_home"
}
trap cleanup EXIT

# Run with a deliberately tiny environment so the smoke path cannot depend on
# user credentials or live profile state. PYTHONPATH points at this checkout so
# editable installs are not required for the baseline repository checks.
env -i \
  PATH="$PATH" \
  HOME="${HOME:-}" \
  TMPDIR="${TMPDIR:-/tmp}" \
  HERMES_HOME="$smoke_home" \
  PYTHONPATH="$ROOT" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$python_bin" - <<'PY'
from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path

root = Path.cwd()
pyproject = root / "pyproject.toml"
if not pyproject.exists():
    raise SystemExit("pyproject.toml missing from repository root")

project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
if project.get("name") != "hermes-agent":
    raise SystemExit(f"unexpected project name: {project.get('name')!r}")
print("ok: pyproject -> hermes-agent")

import hermes_constants  # noqa: F401
from hermes_cli.config import load_config_readonly, validate_config_structure
from hermes_state import SessionDB

home = Path(os.environ["HERMES_HOME"]).resolve()
tmp_root = Path(tempfile.gettempdir()).resolve()
factory_runtime = (root / ".factory" / "runtime").resolve()
if not (home == tmp_root or tmp_root in home.parents or factory_runtime in home.parents):
    raise SystemExit(f"refusing to smoke against non-temporary HERMES_HOME: {home}")

config = load_config_readonly()
issues = validate_config_structure(config)
if issues:
    raise SystemExit(f"default config validation failed: {issues!r}")
print("ok: default config validates")

db = SessionDB()
row = db._conn.execute("PRAGMA quick_check").fetchone()
value = row[0] if row else None
if value != "ok":
    raise SystemExit(f"state.db quick_check failed: {value!r}")
print(f"ok: state.db quick_check -> {value}")
PY

echo "ok: smoke passed"
