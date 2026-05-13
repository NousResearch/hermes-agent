#!/usr/bin/env bash
# Safe starter template for IT automation lab shell scripts.

set -euo pipefail

DRY_RUN=0
TARGET=""

usage() {
  cat <<'USAGE'
Usage: bash-script-template.sh [--dry-run] --target <target>

Options:
  --dry-run          Print actions without changing state.
  --target <target>  Target host, path, service, or identifier.
  -h, --help         Show this help text.
USAGE
}

log() {
  printf '[%s] %s
' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

fail() {
  log "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

run_cmd() {
  if [ "$DRY_RUN" -eq 1 ]; then
    log "DRY-RUN: $*"
  else
    log "RUN: $*"
    "$@"
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --target)
      TARGET="${2:-}"
      [ -n "$TARGET" ] || fail "--target requires a value"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

[ -n "$TARGET" ] || fail "--target is required"

main() {
  require_command date
  log "Starting automation for target: $TARGET"
  log "Replace this template action with a safe, idempotent operation."
  run_cmd true
  log "Completed. Add verification commands before using this in production."
}

main "$@"
