#!/bin/bash
# test_aggregate_cron_health.sh — Tests for aggregate_cron_health.sh
# Usage: bash test_aggregate_cron_health.sh

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT
mkdir -p "$TMPDIR/cron/output"
cp aggregate_cron_health.sh "$TMPDIR/"

PASS=0
TOTAL=7

ok() { PASS=$((PASS+1)); echo "✅ $1"; }
fail() { echo "❌ $1"; }

# T1: All healthy
echo '{"jobs":[{"id":"a1","name":"ok","last_status":"ok","no_agent":true,"paused_at":null}]}' > "$TMPDIR/cron/jobs.json"
out=$(HERMES_HOME="$TMPDIR" TODAY="2026-07-13" bash "$TMPDIR/aggregate_cron_health.sh" 2>&1)
echo "$out" | python3 -c "import json,sys; d=json.loads(sys.stdin.readline()); assert d['success_rate']==1.0" 2>/dev/null && ok "T1 all-healthy" || fail "T1"

# T2: Detect failure
echo '{"jobs":[{"id":"f1","name":"fail","last_status":"error","no_agent":true,"paused_at":null},{"id":"o1","name":"ok","last_status":"ok","no_agent":true,"paused_at":null}]}' > "$TMPDIR/cron/jobs.json"
out=$(HERMES_HOME="$TMPDIR" TODAY="2026-07-13" bash "$TMPDIR/aggregate_cron_health.sh" 2>&1)
echo "$out" | grep -q "CRITICAL\|fail_cron\|failure" && ok "T2 detect-failure" || fail "T2"

# T3: LLM thin output
mkdir -p "$TMPDIR/cron/output/thin_01"
echo -e "line1\nline2" > "$TMPDIR/cron/output/thin_01/latest.md"
echo '{"jobs":[{"id":"thin_01","name":"thin","last_status":"ok","no_agent":false,"paused_at":null}]}' > "$TMPDIR/cron/jobs.json"
out=$(HERMES_HOME="$TMPDIR" TODAY="2026-07-13" bash "$TMPDIR/aggregate_cron_health.sh" 2>&1)
echo "$out" | grep -q "WARNING" && ok "T3 llm-thin-output" || fail "T3"

# T4: SLO <95% alert
echo '{"jobs":[{"id":"a1","last_status":"ok","no_agent":true,"paused_at":null},{"id":"a2","last_status":"ok","no_agent":true,"paused_at":null},{"id":"a3","last_status":"error","no_agent":true,"paused_at":null}]}' > "$TMPDIR/cron/jobs.json"
out=$(HERMES_HOME="$TMPDIR" TODAY="2026-07-13" bash "$TMPDIR/aggregate_cron_health.sh" 2>&1)
echo "$out" | grep -q "CRITICAL\|66%" && ok "T4 slo-alert" || fail "T4"

# T5: Paused cron excluded
echo '{"jobs":[{"id":"p1","name":"paused","last_status":"error","no_agent":true,"paused_at":"2026-01-01"}]}' > "$TMPDIR/cron/jobs.json"
out=$(HERMES_HOME="$TMPDIR" TODAY="2026-07-13" bash "$TMPDIR/aggregate_cron_health.sh" 2>&1)
cat "$TMPDIR/logs/cron_health_2026-07-13.json" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['total_crons']==0" 2>/dev/null && ok "T5 paused-excluded" || fail "T5"

# T6: JSON output file exists
echo '{"jobs":[{"id":"o1","name":"ok","last_status":"ok","no_agent":true,"paused_at":null}]}' > "$TMPDIR/cron/jobs.json"
HERMES_HOME="$TMPDIR" TODAY="2026-07-13" bash "$TMPDIR/aggregate_cron_health.sh" > /dev/null 2>&1
[[ -f "$TMPDIR/logs/cron_health_2026-07-13.json" ]] && ok "T6 json-output" || fail "T6"

# T7: Exit codes (0/1/2)
echo '{"jobs":[{"id":"o1","name":"ok","last_status":"ok","no_agent":true,"paused_at":null}]}' > "$TMPDIR/cron/jobs.json"
HERMES_HOME="$TMPDIR" TODAY="2026-07-13" bash "$TMPDIR/aggregate_cron_health.sh" > /dev/null 2>&1; e=$?
[[ $e -eq 0 ]] && ok "T7 exit-code-healthy" || fail "T7"

echo ""
echo "$PASS/$TOTAL PASS"
[[ $PASS -eq $TOTAL ]]
