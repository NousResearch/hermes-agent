#!/usr/bin/env bash
# morning-chain.sh — M5 morning workflow: journal → perf-coach brief → hermes deliver
#
# Usage:
#   morning-chain.sh           # run all three steps
#   morning-chain.sh --dry-run # print each command without executing
#
# Locking: uses flock (Linux / homebrew util-linux) or shlock (macOS /usr/bin/shlock)
# to ensure at most one invocation of each step runs at a time.
# Lock files: logs/morning-chain-step{1,2,3}.lock (in MORNING_CHAIN_LOCK_DIR)
#
# Environment overrides (for testing):
#   MORNING_CHAIN_STEP1      override step-1 command (default: bin/journal-morning-run.sh)
#   MORNING_CHAIN_STEP2      override step-2 command (default: python3 .../export_brief.py)
#   MORNING_CHAIN_STEP3      override step-3 command (default: hermes brief ...)
#   MORNING_CHAIN_LOG_DIR    override log directory   (default: <repo>/logs)
#   MORNING_CHAIN_LOCK_DIR   override lock directory  (default: <repo>/logs)

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (the directory containing this deploy/ tree)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KILL_SWITCH="${REPO_ROOT}/deploy/.morning-chain-disabled"
LOG_DIR="${MORNING_CHAIN_LOG_DIR:-${REPO_ROOT}/logs}"
LOCK_DIR="${MORNING_CHAIN_LOCK_DIR:-${REPO_ROOT}/logs}"
DATE="$(date +%Y-%m-%d)"
LOG_FILE="${LOG_DIR}/morning-chain-${DATE}.log"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

# ---------------------------------------------------------------------------
# Step commands (overridable for tests)
# ---------------------------------------------------------------------------
STEP1_CMD="${MORNING_CHAIN_STEP1:-${REPO_ROOT}/bin/journal-morning-run.sh}"
STEP2_CMD="${MORNING_CHAIN_STEP2:-python3 ${HOME}/perf-coach/scripts/export_brief.py}"
STEP3_CMD="${MORNING_CHAIN_STEP3:-hermes brief compose --deliver --target 06:00}"

STEP1_LOCK="${LOCK_DIR}/morning-chain-step1.lock"
STEP2_LOCK="${LOCK_DIR}/morning-chain-step2.lock"
STEP3_LOCK="${LOCK_DIR}/morning-chain-step3.lock"

# ---------------------------------------------------------------------------
# Kill-switch check — exits 0 silently when the disable file exists
# ---------------------------------------------------------------------------
if [[ -f "${KILL_SWITCH}" ]]; then
  exit 0
fi

# ---------------------------------------------------------------------------
# Dry-run mode — print each command and exit
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[dry-run] Step 1: ${STEP1_CMD}"
  echo "[dry-run] Step 2: ${STEP2_CMD}"
  echo "[dry-run] Step 3: ${STEP3_CMD}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" | tee -a "${LOG_FILE}"
}

log "morning-chain starting (pid=$$)"

# ---------------------------------------------------------------------------
# _try_lock — attempt non-blocking lock acquisition on a lock file
#   Uses flock (Linux / homebrew util-linux) when available,
#   falls back to shlock (macOS /usr/bin/shlock) otherwise.
#   Returns 0 if lock acquired, non-zero if already held.
# ---------------------------------------------------------------------------
_try_lock() {
  local lock_file="$1"
  if command -v flock &>/dev/null; then
    flock -n "${lock_file}" true
  elif command -v shlock &>/dev/null; then
    # shlock creates a PID file; returns 0 if lock acquired, 1 if held by live PID
    shlock -f "${lock_file}" -p "$$"
  else
    # Portable POSIX fallback: atomic directory creation
    mkdir "${lock_file}.lk" 2>/dev/null
  fi
}

# ---------------------------------------------------------------------------
# _release_lock — release lock file after step completes (non-flock paths)
# ---------------------------------------------------------------------------
_release_lock() {
  local lock_file="$1"
  if command -v flock &>/dev/null; then
    : # flock releases automatically when the fd/subprocess exits
  elif command -v shlock &>/dev/null; then
    rm -f "${lock_file}"
  else
    rmdir "${lock_file}.lk" 2>/dev/null || true
  fi
}

# ---------------------------------------------------------------------------
# run_step — execute one step under a per-step lock; abort chain on failure
# ---------------------------------------------------------------------------
run_step() {
  local step_num="$1"
  local lock_file="$2"
  local cmd="$3"

  log "Step ${step_num}: acquiring lock ${lock_file}"

  if ! _try_lock "${lock_file}"; then
    log "Step ${step_num}: lock busy — another invocation is running; skipping"
    exit 0
  fi

  log "Step ${step_num}: running: ${cmd}"

  local exit_code=0
  if command -v flock &>/dev/null; then
    # Hold the lock for the duration of the step subprocess
    flock -n "${lock_file}" bash -c "${cmd}" >> "${LOG_FILE}" 2>&1 || exit_code=$?
  else
    bash -c "${cmd}" >> "${LOG_FILE}" 2>&1 || exit_code=$?
    _release_lock "${lock_file}"
  fi

  if [[ ${exit_code} -ne 0 ]]; then
    log "Step ${step_num}: FAILED (exit ${exit_code}) — aborting chain"
    exit ${exit_code}
  fi

  log "Step ${step_num}: OK"
}

# ---------------------------------------------------------------------------
# Execute the chain
# ---------------------------------------------------------------------------
run_step 1 "${STEP1_LOCK}" "${STEP1_CMD}"
run_step 2 "${STEP2_LOCK}" "${STEP2_CMD}"
run_step 3 "${STEP3_LOCK}" "${STEP3_CMD}"

log "morning-chain complete"
