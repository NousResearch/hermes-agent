#!/usr/bin/env bash
# Offline test orchestrator for the talaria skill suite.
#
# Runs every tier and fails on first red. Safe by construction:
#   - dead-proxy egress trap: any accidental real-egress request dies on
#     closed loopback port 9; no_proxy keeps 127.0.0.1/localhost flowing
#     to the mocks
#   - the mock stack binds 127.0.0.1 on the 7 default service ports only
#   - no live *arr or Jellyfin API is ever contacted
#
# Toolchain: python3 only (P6 stdlib-only ruling — jq is GONE from v2;
# bash lives only in this orchestrator).
#
# Usage:   tests/run_tests.sh          (from the py/ dir, or anywhere)
# Exits:   0 all green, non-zero on first failure.

set -euo pipefail
cd "$(dirname "$0")/.."

# --- dead-proxy egress trap (loopback-only test harness) --------------------
export http_proxy="http://127.0.0.1:9"
export https_proxy="http://127.0.0.1:9"
export all_proxy="http://127.0.0.1:9"
export HTTP_PROXY="http://127.0.0.1:9"
export HTTPS_PROXY="http://127.0.0.1:9"
export ALL_PROXY="http://127.0.0.1:9"
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# --- toolchain precondition -------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "FAIL: python3 (>= 3.8) is required" >&2
  exit 1
fi

echo "== fixtures: regenerate deterministically (route cross-check) =="
python3 tests/gen_fixtures.py

echo "== TP0 + TP1 + TP2 + TP2w + TP2j: structural, CLI sweep, mock E2E =="
python3 -m unittest discover -s tests -t . -p 'test_*.py'

echo
echo "ALL GREEN: TP0 structural + TP1 CLI + TP2/TP2w/TP2j E2E passed inside the dead-proxy harness."
