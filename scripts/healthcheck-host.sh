#!/usr/bin/env bash
# Basic host resource health check for disk and memory pressure.
#
# This is intentionally scoped to host resource pressure. Application
# liveness/readiness checks should stay in the Hermes runtime healthcheck
# path so container/orchestrator probes can call the app check and this script
# independently.
# Environment overrides:
#   DISK_USAGE_THRESHOLD=90    # fail if any checked filesystem is >= this % used
#   MEMORY_USAGE_THRESHOLD=90  # fail if memory usage is >= this % used
#   HERMES_HEALTHCHECK_MEMINFO=/proc/meminfo  # test override for meminfo path

set -euo pipefail

DISK_USAGE_THRESHOLD="${DISK_USAGE_THRESHOLD:-90}"
MEMORY_USAGE_THRESHOLD="${MEMORY_USAGE_THRESHOLD:-90}"
MEMINFO_PATH="${HERMES_HEALTHCHECK_MEMINFO:-/proc/meminfo}"

failures=0

is_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

validate_threshold() {
  local name="$1"
  local value="$2"

  if ! is_integer "$value" || (( value < 1 || value > 100 )); then
    echo "ERROR: $name must be an integer between 1 and 100 (got '$value')" >&2
    exit 2
  fi
}

check_disk_space() {
  echo "Disk space:"

  local df_output
  if ! df_output="$(df -P -x tmpfs -x devtmpfs -x squashfs)"; then
    echo "  FAIL unable to read filesystem usage"
    failures=$((failures + 1))
    return
  fi

  # Skip pseudo/temporary filesystems so the signal is focused on real mounted
  # storage. Parse from the right so mount points containing spaces are kept.
  while IFS= read -r line; do
    [[ "$line" == Filesystem* ]] && continue
    [[ -z "$line" ]] && continue

    local source used_pct target used
    read -r source _ _ _ used_pct target <<< "$line"
    used="${used_pct%%%}"

    if ! is_integer "$used" || [[ -z "$target" ]]; then
      echo "  FAIL unable to parse filesystem usage line: ${line}"
      failures=$((failures + 1))
      continue
    fi

    if (( used >= DISK_USAGE_THRESHOLD )); then
      echo "  FAIL ${target}: ${used}% used (${source})"
      failures=$((failures + 1))
    else
      echo "  OK   ${target}: ${used}% used (${source})"
    fi
  done <<< "$df_output"
}

check_memory() {
  echo "Memory:"

  local mem_total_kb mem_available_kb mem_used_pct
  mem_total_kb="$(awk '/^MemTotal:/ {print $2}' "$MEMINFO_PATH" 2>/dev/null || true)"
  mem_available_kb="$(awk '/^MemAvailable:/ {print $2}' "$MEMINFO_PATH" 2>/dev/null || true)"

  if ! is_integer "$mem_total_kb" || ! is_integer "$mem_available_kb" || (( mem_total_kb == 0 || mem_available_kb > mem_total_kb )); then
    echo "  FAIL unable to read memory information from ${MEMINFO_PATH}"
    failures=$((failures + 1))
    return
  fi

  mem_used_pct=$(( (mem_total_kb - mem_available_kb) * 100 / mem_total_kb ))

  if (( mem_used_pct >= MEMORY_USAGE_THRESHOLD )); then
    echo "  FAIL ${mem_used_pct}% used (threshold: ${MEMORY_USAGE_THRESHOLD}%)"
    failures=$((failures + 1))
  else
    echo "  OK   ${mem_used_pct}% used (threshold: ${MEMORY_USAGE_THRESHOLD}%)"
  fi
}

main() {
  validate_threshold "DISK_USAGE_THRESHOLD" "$DISK_USAGE_THRESHOLD"
  validate_threshold "MEMORY_USAGE_THRESHOLD" "$MEMORY_USAGE_THRESHOLD"

  echo "Healthcheck thresholds: disk >= ${DISK_USAGE_THRESHOLD}%, memory >= ${MEMORY_USAGE_THRESHOLD}%"
  check_disk_space
  check_memory

  if (( failures > 0 )); then
    echo "Healthcheck failed: ${failures} issue(s) found"
    exit 1
  fi

  echo "Healthcheck passed"
}

main "$@"
