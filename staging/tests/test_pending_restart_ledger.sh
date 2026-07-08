#!/usr/bin/env bash
# Tests for the A1b pending-restart ledger in fleet-gateway-restart.sh.
# Lib-mode source (functions only), drives fgr_ledger_add / fgr_ledger_clear
# against a temp HERMES + a stubbed runtime tree. No launchctl, no real restart.
#
# Run: bash staging/tests/test_pending_restart_ledger.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FGR="$HERE/../fleet/fleet-gateway-restart.sh"
PASS="✅"; FAIL="❌"; OK=1

check() { if eval "$2"; then echo "  $PASS $1"; else echo "  $FAIL $1"; OK=0; fi; }

TMP="$(mktemp -d -t fgrledger.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/state" "$TMP/runtime/hermes-agent"
# a real git repo so fgr_target_sha resolves a sha
( cd "$TMP/runtime/hermes-agent" && git init -q && git config user.email t@t.t \
  && git config user.name t && echo x > f && git add -A && git commit -q -m x )

export HERMES_HOME_ROOT="$TMP"
export FGR_RUNTIME_TREE="$TMP/runtime/hermes-agent"
export FGR_LEDGER="$TMP/state/pending-gateway-restart.json"
export FGR_LIB=1
# shellcheck disable=SC1090
source "$FGR"

echo ""
echo "TEST 1 — fgr_ledger_add creates an entry with target_sha + first_skipped"
fgr_ledger_add "ai.hermes.gateway-aegis" "12345"
check "ledger file created" "[ -f '$FGR_LEDGER' ]"
check "entry present" "python3 -c 'import json;d=json.load(open(\"$FGR_LEDGER\"));assert \"ai.hermes.gateway-aegis\" in d'"
check "target_sha is a real sha" "python3 -c 'import json;d=json.load(open(\"$FGR_LEDGER\"));assert len(d[\"ai.hermes.gateway-aegis\"][\"target_sha\"])==40'"
check "since_epoch recorded" "python3 -c 'import json;d=json.load(open(\"$FGR_LEDGER\"));assert d[\"ai.hermes.gateway-aegis\"][\"since_epoch\"]==\"12345\"'"

echo ""
echo "TEST 2 — re-add preserves first_skipped (idempotent update)"
FIRST=$(python3 -c 'import json;print(json.load(open("'"$FGR_LEDGER"'"))["ai.hermes.gateway-aegis"]["first_skipped"])')
sleep 1
fgr_ledger_add "ai.hermes.gateway-aegis" "99999"
check "first_skipped preserved" "python3 -c 'import json;d=json.load(open(\"$FGR_LEDGER\"));assert str(d[\"ai.hermes.gateway-aegis\"][\"first_skipped\"])==\"$FIRST\"'"
check "last_skipped advanced" "python3 -c 'import json;d=json.load(open(\"$FGR_LEDGER\"));e=d[\"ai.hermes.gateway-aegis\"];assert int(e[\"last_skipped\"])>=int(e[\"first_skipped\"])'"

echo ""
echo "TEST 3 — a second label coexists"
fgr_ledger_add "ai.hermes.gateway-argus" "55555"
check "both labels present" "python3 -c 'import json;d=json.load(open(\"$FGR_LEDGER\"));assert len(d)==2'"

echo ""
echo "TEST 4 — fgr_ledger_clear removes only its label"
fgr_ledger_clear "ai.hermes.gateway-aegis"
check "aegis cleared" "python3 -c 'import json;d=json.load(open(\"$FGR_LEDGER\"));assert \"ai.hermes.gateway-aegis\" not in d'"
check "argus retained" "python3 -c 'import json;d=json.load(open(\"$FGR_LEDGER\"));assert \"ai.hermes.gateway-argus\" in d'"

echo ""
echo "TEST 5 — clear on absent label is a no-op (no crash, ledger intact)"
fgr_ledger_clear "ai.hermes.gateway-nonexistent"
check "ledger still valid + argus present" "python3 -c 'import json;d=json.load(open(\"$FGR_LEDGER\"));assert \"ai.hermes.gateway-argus\" in d'"

echo ""
echo "TEST 6 — clear when no ledger file exists is a no-op (fail-safe)"
rm -f "$FGR_LEDGER"
fgr_ledger_clear "ai.hermes.gateway-argus"; rc=$?
check "clear on missing ledger returns 0" "[ $rc -eq 0 ]"

echo ""
if [ "$OK" = "1" ]; then echo "ALL LEDGER TESTS PASSED $PASS"; exit 0; else echo "SOME TESTS FAILED $FAIL"; exit 1; fi
