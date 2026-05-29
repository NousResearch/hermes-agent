#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

CONFIG_PATH=""
RUN_HERMES=0

usage() {
  cat >&2 <<'USAGE'
Usage: healthcheck.sh [--config PATH] [--run-hermes-version]

Performs local, non-live checks for the Dobby package templates and scripts.
--run-hermes-version is optional and only runs when --config is also provided.
It does not start the gateway, contact Discord, or call model providers.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      if [ "$#" -lt 2 ]; then
        usage
        exit 2
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --run-hermes-version)
      RUN_HERMES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

failures=0

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  failures=$((failures + 1))
}

pass() {
  printf 'OK: %s\n' "$1"
}

check_file() {
  file="$1"
  if [ -f "$file" ]; then
    pass "found $file"
  else
    fail "missing $file"
  fi
}

check_file "$PACKAGE_ROOT/config/.env.example"
check_file "$PACKAGE_ROOT/config/config.example.yaml"
check_file "$PACKAGE_ROOT/config/SOUL.example.md"
check_file "$PACKAGE_ROOT/config/tool-policy.example.yaml"
check_file "$PACKAGE_ROOT/scripts/preflight.sh"
check_file "$PACKAGE_ROOT/scripts/healthcheck.sh"
check_file "$PACKAGE_ROOT/scripts/redaction-check.sh"
check_file "$PACKAGE_ROOT/README.md"

for script in "$PACKAGE_ROOT"/scripts/*.sh; do
  if [ -f "$script" ]; then
    if grep -q '^set -euo pipefail$' "$script"; then
      pass "$(basename "$script") uses strict shell mode"
    else
      fail "$(basename "$script") is missing set -euo pipefail"
    fi

    if bash -n "$script"; then
      pass "$(basename "$script") parses with bash -n"
    else
      fail "$(basename "$script") failed bash -n"
    fi
  fi
done

if [ -n "$CONFIG_PATH" ]; then
  if [ -f "$CONFIG_PATH" ]; then
    pass "explicit config path exists: $CONFIG_PATH"
  else
    fail "explicit config path missing: $CONFIG_PATH"
  fi
fi

if [ "$RUN_HERMES" -eq 1 ]; then
  if [ -z "$CONFIG_PATH" ]; then
    fail "--run-hermes-version requires --config PATH"
  elif command -v hermes >/dev/null 2>&1; then
    temp_home="$(mktemp -d "${TMPDIR:-/tmp}/dobby-healthcheck-home.XXXXXX")"
    trap 'rm -rf "$temp_home"' EXIT
    ln -s "$CONFIG_PATH" "$temp_home/config.yaml"
    if HERMES_HOME="$temp_home" hermes version >/dev/null 2>&1; then
      pass "hermes version command ran with isolated HERMES_HOME"
    else
      fail "hermes version command failed with isolated HERMES_HOME"
    fi
  else
    fail "hermes command not found"
  fi
fi

if [ "$failures" -ne 0 ]; then
  printf 'healthcheck: %s check(s) failed\n' "$failures" >&2
  exit 1
fi

printf 'healthcheck: local dry-run checks passed\n'
