#!/usr/bin/env bash
set -euo pipefail

# HL Terminal masterpiece quality gate
# Tier orchestration:
#  - FAST gate: daily smoke + deterministic local checks
#  - HYGIENE gate: UX/security contract checks
#  - UI_POLISH gate: deterministic UX polish checks (a11y, console, keyboard, perf, tokens)
#  - STABILITY gate: deterministic repeat-run checks
#  - RELIABILITY MEASUREMENT gate: measurement-spec compliance checks
#  - FULL gate: weekly health suite
#
# Exit 0 = PASS
# Exit 2 = FAIL (real gate failures)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HL_TERMINAL_ROOT:-$(pwd)}"
FAST_GATE="$SCRIPT_DIR/hl_terminal_fast_gate.sh"
HYGIENE_GATE="$SCRIPT_DIR/hl_terminal_hygiene_gate.sh"
UI_POLISH_GATE="$SCRIPT_DIR/hl_terminal_ui_polish_gate.sh"
STABILITY_GATE="$SCRIPT_DIR/hl_terminal_stability_gate.sh"
RELIABILITY_MEASUREMENT_GATE="$SCRIPT_DIR/hl_terminal_reliability_measurement_gate.sh"
FULL_GATE="$SCRIPT_DIR/hl_terminal_full_gate.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 2
}

echo "== HL Terminal Masterpiece Gate (FAST -> HYGIENE -> UI_POLISH -> STABILITY -> RELIABILITY_MEASUREMENT -> FULL) ==" >&2
echo "ROOT=$ROOT" >&2

[[ -f "$FAST_GATE" ]] || fail "FAST gate script missing ($FAST_GATE)"
[[ -f "$HYGIENE_GATE" ]] || fail "HYGIENE gate script missing ($HYGIENE_GATE)"
[[ -f "$UI_POLISH_GATE" ]] || fail "UI_POLISH gate script missing ($UI_POLISH_GATE)"
[[ -f "$STABILITY_GATE" ]] || fail "STABILITY gate script missing ($STABILITY_GATE)"
[[ -f "$RELIABILITY_MEASUREMENT_GATE" ]] || fail "RELIABILITY MEASUREMENT gate script missing ($RELIABILITY_MEASUREMENT_GATE)"
[[ -f "$FULL_GATE" ]] || fail "FULL gate script missing ($FULL_GATE)"

run_gate() {
  local label="$1"
  local script="$2"

  if env HL_TERMINAL_ROOT="$ROOT" bash "$script"; then
    echo "OK: $label gate passed" >&2
  else
    local rc=$?
    echo "FAIL: $label gate failed; stopping orchestration" >&2
    exit "$rc"
  fi
}

run_gate "FAST" "$FAST_GATE"
run_gate "HYGIENE" "$HYGIENE_GATE"
run_gate "UI_POLISH" "$UI_POLISH_GATE"
run_gate "STABILITY" "$STABILITY_GATE"
run_gate "RELIABILITY_MEASUREMENT" "$RELIABILITY_MEASUREMENT_GATE"
run_gate "FULL" "$FULL_GATE"

echo "OK: hl-terminal masterpiece gate passed (FAST + HYGIENE + UI_POLISH + STABILITY + RELIABILITY_MEASUREMENT + FULL)" >&2
