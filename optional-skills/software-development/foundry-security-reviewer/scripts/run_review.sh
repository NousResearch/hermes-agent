#!/usr/bin/env bash
# Run a Foundry security-review baseline and write review_output.md.

set -u
set -o pipefail

PROJECT_ROOT="${1:-$PWD}"

if ! command -v forge >/dev/null 2>&1; then
  printf 'Error: forge CLI was not found on PATH. Install Foundry from https://getfoundry.sh and retry.\n' >&2
  exit 127
fi

if [ ! -d "$PROJECT_ROOT" ]; then
  printf 'Error: project root does not exist: %s\n' "$PROJECT_ROOT" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
if [ ! -f "$PROJECT_ROOT/foundry.toml" ]; then
  printf 'Error: foundry.toml was not found in %s. Run this script from a Foundry project root.\n' "$PROJECT_ROOT" >&2
  exit 2
fi

OUTPUT_FILE="$PROJECT_ROOT/review_output.md"
LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/foundry-security-review.XXXXXX")"
trap 'rm -rf "$LOG_DIR"' EXIT

BUILD_LOG="$LOG_DIR/build.log"
TEST_LOG="$LOG_DIR/test.log"
COVERAGE_LOG="$LOG_DIR/coverage.log"
SNAPSHOT_LOG="$LOG_DIR/snapshot.log"
SLITHER_LOG="$LOG_DIR/slither.log"
ADERYN_LOG="$LOG_DIR/aderyn.log"

BUILD_STATUS=0
TEST_STATUS=0
COVERAGE_STATUS=0
SNAPSHOT_STATUS=0
SLITHER_STATUS=0
ADERYN_STATUS=0
HAS_SLITHER=0
HAS_ADERYN=0

run_capture() {
  local log_file="$1"
  local status
  shift

  "$@" >"$log_file" 2>&1
  status=$?
  return "$status"
}

append_log() {
  local log_file="$1"
  if [ -s "$log_file" ]; then
    printf '\n```text\n'
    cat "$log_file"
    printf '```\n'
  else
    printf '\n_No output produced._\n'
  fi
}

append_coverage_flags() {
  local flagged
  flagged="$(awk -F'|' '
    /\.sol[[:space:]]*\|/ {
      low = 0
      for (i = 3; i <= NF; i++) {
        cell = $i
        if (cell ~ /%/) {
          value = cell
          sub(/%.*/, "", value)
          gsub(/[^0-9.]/, "", value)
          if (value != "" && value + 0 < 80) low = 1
        }
      }
      if (low) {
        file = $2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", file)
        print "- [ ] " file " — below 80% in the coverage summary"
      }
    }
  ' "$COVERAGE_LOG")"

  if [ -n "$flagged" ]; then
    printf '\n### Modules below 80%%\n%s\n' "$flagged"
  else
    printf '\n### Modules below 80%%\n- [x] No module below 80%% was detected in the summary.\n'
  fi
}

append_filtered_findings() {
  local log_file="$1"
  local label="$2"
  local pattern="$3"
  local findings

  findings="$(grep -Eis "$pattern" "$log_file" || true)"
  if [ -n "$findings" ]; then
    printf '\n### %s\n```text\n%s\n```\n' "$label" "$findings"
  else
    printf '\n### %s\n- [x] No matching findings were reported.\n' "$label"
  fi
}

printf 'Running Foundry security review in %s\n' "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

run_capture "$BUILD_LOG" forge build || BUILD_STATUS=$?
run_capture "$TEST_LOG" forge test -vv || TEST_STATUS=$?
run_capture "$COVERAGE_LOG" forge coverage --report summary || COVERAGE_STATUS=$?
run_capture "$SNAPSHOT_LOG" forge snapshot || SNAPSHOT_STATUS=$?

if command -v slither >/dev/null 2>&1; then
  HAS_SLITHER=1
  run_capture "$SLITHER_LOG" slither . || SLITHER_STATUS=$?
fi

if command -v aderyn >/dev/null 2>&1; then
  HAS_ADERYN=1
  run_capture "$ADERYN_LOG" aderyn . || ADERYN_STATUS=$?
fi

{
  printf '# Foundry Security Review\n\n'
  printf -- '- **Project:** `%s`\n' "$PROJECT_ROOT"
  printf -- '- **Generated:** `%s`\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf '\n## Test Results\n'
  if [ "$BUILD_STATUS" -eq 0 ]; then
    printf -- '- [x] `forge build` succeeded.\n'
  else
    printf -- '- [ ] `forge build` failed (exit %s). Resolve compilation errors before relying on this review.\n' "$BUILD_STATUS"
  fi
  append_log "$BUILD_LOG"
  if [ "$TEST_STATUS" -eq 0 ]; then
    printf '\n- [x] `forge test -vv` passed.\n'
  else
    printf '\n- [ ] `forge test -vv` failed (exit %s). Failing tests and traces follow.\n' "$TEST_STATUS"
  fi
  append_log "$TEST_LOG"

  printf '\n## Coverage\n'
  if [ "$COVERAGE_STATUS" -eq 0 ]; then
    printf -- '- [x] `forge coverage --report summary` completed.\n'
  else
    printf -- '- [ ] `forge coverage --report summary` failed (exit %s).\n' "$COVERAGE_STATUS"
  fi
  append_log "$COVERAGE_LOG"
  append_coverage_flags

  printf '\n## Gas Snapshot\n'
  if [ "$SNAPSHOT_STATUS" -eq 0 ]; then
    printf -- '- [x] `forge snapshot` captured the gas baseline.\n'
  else
    printf -- '- [ ] `forge snapshot` failed (exit %s).\n' "$SNAPSHOT_STATUS"
  fi
  append_log "$SNAPSHOT_LOG"

  printf '\n## Security Findings\n'
  if [ "$HAS_SLITHER" -eq 1 ]; then
    if [ "$SLITHER_STATUS" -eq 0 ]; then
      printf -- '- [x] `slither .` completed.\n'
    else
      printf -- '- [ ] `slither .` exited %s; inspect the output before dismissing findings.\n' "$SLITHER_STATUS"
    fi
    append_filtered_findings "$SLITHER_LOG" 'Slither HIGH/MEDIUM findings' 'high|medium'
  else
    printf -- '- [ ] Slither was not installed; skipped `slither .`.\n'
  fi
  if [ "$HAS_ADERYN" -eq 1 ]; then
    if [ "$ADERYN_STATUS" -eq 0 ]; then
      printf -- '- [x] `aderyn .` completed.\n'
    else
      printf -- '- [ ] `aderyn .` exited %s; inspect the output before dismissing findings.\n' "$ADERYN_STATUS"
    fi
    append_filtered_findings "$ADERYN_LOG" 'Aderyn critical findings' 'critical'
  else
    printf -- '- [ ] Aderyn was not installed; skipped `aderyn .`.\n'
  fi

  printf '\n## Recommendations\n'
  printf -- '- [ ] Validate every reported finding against the contract source and reachable call paths.\n'
  printf -- '- [ ] Add a Forge regression or fuzz test for every confirmed vulnerability.\n'
  printf -- '- [ ] Compare the change against the bundled common-vulnerabilities reference before approving it.\n'
  printf -- '- [ ] Record intentional Slither suppressions (`# slither-disable`) in the PR.\n'
} >"$OUTPUT_FILE"

printf 'Review written to %s\n' "$OUTPUT_FILE"

if [ "$BUILD_STATUS" -ne 0 ] || [ "$TEST_STATUS" -ne 0 ] || [ "$COVERAGE_STATUS" -ne 0 ] || [ "$SNAPSHOT_STATUS" -ne 0 ]; then
  exit 1
fi
