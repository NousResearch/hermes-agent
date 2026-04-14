#!/usr/bin/env bash
set -euo pipefail

# HL Terminal FAST gate (few minutes, fail fast)
# Exit 0 = PASS
# Exit 2 = FAIL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HL_TERMINAL_ROOT:-$(pwd)}"
TERMINAL="$ROOT/terminal"
BACKEND="$ROOT/backend"

PAGE_TSX="$TERMINAL/src/app/page.tsx"
SPARKLINE_INLINE="$TERMINAL/src/components/InlineSparkline.tsx"
SPARKLINE_TRAJ="$TERMINAL/src/components/TrajectorySparkline.tsx"
SPARKLINE_TSX="$SPARKLINE_INLINE"
if [[ ! -f "$SPARKLINE_TSX" && -f "$SPARKLINE_TRAJ" ]]; then
  SPARKLINE_TSX="$SPARKLINE_TRAJ"
fi
THROTTLE_FILE="$TERMINAL/src/hooks/useRealtimeInvalidation.ts"
SPEC_SUMMARY="$ROOT/autopilot_artifacts/spec/spec_backlog_summary.md"
P0_PROGRESS_JSON="$ROOT/autopilot_artifacts/p0_progress.json"

warns=()

fail() {
  echo "FAIL: $*" >&2
  exit 2
}

warn() {
  local msg="$*"
  warns+=("$msg")
  echo "WARN: $msg" >&2
}

run_named() {
  local name="$1"
  shift
  echo "::step::$name" >&2
  "$@" || fail "$name"
  echo "OK: $name" >&2
}

run_node_test_if_present() {
  local label="$1"
  local workdir="$2"
  local rel_test_path="$3"

  if [[ ! -f "$workdir/$rel_test_path" ]]; then
    warn "$label skipped (missing $rel_test_path)"
    return 0
  fi

  echo "::step::$label" >&2
  if bash -lc "cd '$workdir' && node --test '$rel_test_path'"; then
    echo "OK: $label" >&2
  else
    fail "$label failed"
  fi
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    echo "OK: $label ($path)" >&2
  else
    fail "$label missing ($path)"
  fi
}

require_grep() {
  local label="$1"
  local pattern="$2"
  local path="$3"
  if [[ ! -f "$path" ]]; then
    fail "$label check file missing ($path)"
  fi
  if grep -Eq "$pattern" "$path"; then
    echo "OK: $label" >&2
  else
    fail "$label not found in $path"
  fi
}

check_spec_and_p0_tracker() {
  require_file "spec-sync:backlog-summary" "$SPEC_SUMMARY"
  require_file "spec-sync:p0-progress-json" "$P0_PROGRESS_JSON"

  python3 - "$P0_PROGRESS_JSON" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])

try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"FAIL: p0_progress.json invalid JSON: {exc}", file=sys.stderr)
    raise SystemExit(1)

if not isinstance(data, dict):
    print("FAIL: p0_progress.json must be a JSON object", file=sys.stderr)
    raise SystemExit(1)

required = ["completed", "in_progress", "remaining", "verified", "evidence_refs", "validation_refs"]
missing = [k for k in required if k not in data]
if missing:
    print(f"FAIL: p0_progress.json missing required keys: {', '.join(missing)}", file=sys.stderr)
    raise SystemExit(1)

verified = data.get("verified")
if not isinstance(verified, list):
    print("FAIL: p0_progress.json key 'verified' must be a list", file=sys.stderr)
    raise SystemExit(1)

validation_refs = data.get("validation_refs")


def item_key(item):
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("id", "key", "item", "title", "name", "slug"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return str(item).strip()


def refs_nonempty(refs):
    if isinstance(refs, str):
        return bool(refs.strip())
    if isinstance(refs, list):
        return len(refs) > 0
    return False

refs_by_key = {}
if isinstance(validation_refs, dict):
    for k, v in validation_refs.items():
        refs_by_key[str(k).strip()] = v
elif isinstance(validation_refs, list):
    for entry in validation_refs:
        if not isinstance(entry, dict):
            continue
        key = None
        for field in ("id", "key", "item", "title", "name", "slug"):
            val = entry.get(field)
            if isinstance(val, str) and val.strip():
                key = val.strip()
                break
        if not key:
            continue
        refs = entry.get("validation_refs")
        if refs is None:
            refs = entry.get("refs")
        if refs is None:
            refs = entry.get("evidence")
        refs_by_key[key] = refs
else:
    print("FAIL: p0_progress.json key 'validation_refs' must be object or list", file=sys.stderr)
    raise SystemExit(1)

missing_validation = []
for item in verified:
    key = item_key(item)
    refs = refs_by_key.get(key)
    if refs is None:
        refs = refs_by_key.get(key.lower())
    if refs is None:
        refs = refs_by_key.get(key.upper())
    if not refs_nonempty(refs):
        missing_validation.append(key)

if missing_validation:
    print(
        "FAIL: verified items missing validation_refs entries: " + ", ".join(missing_validation),
        file=sys.stderr,
    )
    raise SystemExit(1)

remaining = data.get("remaining")
remaining_count = None
if isinstance(remaining, list):
    remaining_count = len(remaining)
elif remaining is None:
    remaining_count = None
else:
    try:
        remaining_count = len(remaining)
    except Exception:
        remaining_count = None

if remaining_count == 0:
    print(
        "INFO: p0_progress.remaining is empty; FAST gate still enforces all checks and masterpiece gate must still run FULL.",
        file=sys.stderr,
    )

print("OK: p0_progress consistency + arbiter validation")
PY
}

echo "== HL Terminal FAST Gate ==" >&2
echo "ROOT=$ROOT" >&2

run_named "daily_smoke" env HL_TERMINAL_ROOT="$ROOT" bash "$SCRIPT_DIR/hl_terminal_daily_smoke.sh"

echo "== Trust Gate minimal checks ==" >&2
run_node_test_if_present "trust:asof-terminal" "$TERMINAL" "tests/phase19-asof-safety.test.mjs"
run_node_test_if_present "trust:asof-gating-terminal" "$TERMINAL" "tests/phase65-asof-gating.test.mjs"
run_node_test_if_present "trust:evidence-lock-terminal" "$TERMINAL" "tests/arbiters-evidence-lock.test.mjs"
run_node_test_if_present "trust:evidence-api-terminal" "$TERMINAL" "tests/phase65-evidence-api.test.mjs"
run_node_test_if_present "trust:abstain-backend" "$BACKEND" "tests/phase34-coverage-abstain.test.mjs"
run_node_test_if_present "trust:freshness-watermark-terminal" "$TERMINAL" "tests/phase30-news-context.test.mjs"
run_node_test_if_present "trust:freshness-watermark-backend" "$BACKEND" "tests/phase30-news-ingest.test.mjs"

echo "== Market Overview spec smoke ==" >&2
require_file "trajectory:sparkline-component" "$SPARKLINE_TSX"
require_file "trajectory:overview-page" "$PAGE_TSX"
require_grep "trajectory:overview imports sparkline" "import[[:space:]]+(InlineSparkline|TrajectorySparkline)[[:space:]]+from" "$PAGE_TSX"
require_grep "trajectory:overview has trajectory source" "(/api/candles\\?|appendTrajectoryPoint\\()" "$PAGE_TSX"

if grep -Eq "href=\"/system/coverage\"|<Link href=\"/system/coverage\"" "$PAGE_TSX"; then
  echo "OK: coverage diagnostics offloaded away from overview (linked to /system/coverage)" >&2
elif grep -Eq "Coverage status" "$PAGE_TSX"; then
  fail "coverage block still appears on overview without explicit offload marker to /system/coverage"
else
  warn "coverage/offload marker not detected; unable to prove violation"
fi

echo "== Performance spec smoke ==" >&2
require_file "perf:repaint-throttle-config-file" "$THROTTLE_FILE"
require_grep "perf:repaint-throttle-config" "throttleMs|normalizeInvalidationWindowMs|windowMs" "$THROTTLE_FILE"

if grep -Eq "<canvas|getContext\\(['\"]2d['\"]\\)|CanvasRenderingContext2D" "$SPARKLINE_TSX"; then
  echo "OK: sparkline rendering path is canvas-based" >&2
else
  fail "sparkline implementation is not canvas-based in $SPARKLINE_TSX"
fi

if command -v rg >/dev/null 2>&1; then
  if rg -n --hidden --glob '!.git' -e 'LCP|INP|web-vitals|PerformanceObserver' "$TERMINAL/src" "$TERMINAL/tests" >/dev/null 2>&1; then
    echo "OK: found LCP/INP-related instrumentation hooks" >&2
  else
    warn "LCP/INP measurement tooling not found; skipping hard perf measurement for now"
  fi
else
  warn "ripgrep not available; cannot probe LCP/INP tooling"
fi

echo "== P0 tracker arbiter checks ==" >&2
run_named "p0_progress_consistency" check_spec_and_p0_tracker

if ((${#warns[@]} > 0)); then
  echo "WARNINGS (${#warns[@]}):" >&2
  for w in "${warns[@]}"; do
    echo "- $w" >&2
  done
fi

echo "OK: hl-terminal FAST gate passed" >&2
