#!/bin/bash
# End-to-end test of the v2.1 CRON LEG + v2.2 BOT LEG of session_coord.py.
# Everything runs on scratch DB + scratch manifests + scratch cron stores.
SC="${SC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/session_coord.py}"
D=$(mktemp -d "${TMPDIR:-/tmp}/cronleg.XXXXXX")
export HERMES_COORD_DB="$D/board.db"
export HERMES_COORD_CRON_MANIFEST="$D/manifest.json"
export HERMES_COORD_CRON_JOBS="$D/jobs.json"
# v2.2: the radar also globs <profiles>/*/cron/jobs.json (Bot Mode bots keep
# their Routines in per-profile stores) — point it at scratch so the suite is
# hermetic vs any real profiles directory on the machine.
export HERMES_COORD_PROFILES_DIR="$D/profiles"
PASS=0; FAIL=0
ok(){ PASS=$((PASS+1)); echo "ok:   $1"; }
bad(){ FAIL=$((FAIL+1)); echo "FAIL: $1"; }
co(){ python3 "$SC" "$@"; }

# --- fixtures: one job firing in 10 min (backup, critical, wait-policy),
#     one firing in 5 min (watchdog, skip-policy), one unguarded in 20 min
NEXT10=$(python3 -c "from datetime import datetime,timedelta,timezone; print((datetime.now().astimezone()+timedelta(minutes=10)).isoformat())")
NEXT5=$(python3 -c "from datetime import datetime,timedelta,timezone; print((datetime.now().astimezone()+timedelta(minutes=5)).isoformat())")
NEXT20=$(python3 -c "from datetime import datetime,timedelta,timezone; print((datetime.now().astimezone()+timedelta(minutes=20)).isoformat())")
cat > "$D/jobs.json" <<EOF
{"jobs":[
 {"id":"backupjob0001","name":"nightly backup","enabled":true,"next_run_at":"$NEXT10","last_run_at":null},
 {"id":"watchdogjob02","name":"fleet watchdog","enabled":true,"next_run_at":"$NEXT5","last_run_at":null},
 {"id":"unguardedjob3","name":"legacy sweeper","enabled":true,"next_run_at":"$NEXT20","last_run_at":null}
]}
EOF
cat > "$D/manifest.json" <<EOF
{"jobs":{
 "backupjob0001":{"name":"nightly backup","resources":["file:$D/tree"],"policy":"wait","critical":true},
 "watchdogjob02":{"name":"fleet watchdog","resources":["fleet-key"],"policy":"skip","critical":false},
 "unguardedjob3":{"name":"legacy sweeper","resources":["file:$D/tree/sub"],"policy":"unguarded","critical":false}
}}
EOF
mkdir -p "$D/tree/sub"

# ============ 1. session claim overlapping manifested crons -> advisories
S=$(co register --task "editing tree" | head -1)
OUT=$(co claim --id "$S" --res "file:$D/tree" --task edit 2>&1)
echo "$OUT" | grep -q "CLAIMED" && ok "1a. claim succeeds (advisory never blocks)" || bad "1a. claim blocked: $OUT"
echo "$OUT" | grep -q "CRITICAL cron 'nightly backup'.*fires in ~[0-9]*m" && ok "1b. critical advisory present w/ ETA" || bad "1b. no critical advisory: $OUT"
echo "$OUT" | grep -q "ASK THE USER to pause or trigger it early" && ok "1c. critical decision options offered" || bad "1c. options missing"
echo "$OUT" | grep -q "wait-for-cron --job backupjob000" && ok "1d. wait-for-cron hint names the job" || bad "1d. no wait-for-cron hint"
echo "$OUT" | grep -q "does NOT check the board (unguarded)" && ok "1e. unguarded job warned (overlaps via dir cover)" || bad "1e. no unguarded warning: $OUT"
echo "$OUT" | grep -q "inside your claim's 90m TTL window" && ok "1f. TTL-overlap callout" || bad "1f. no TTL callout"

# non-overlapping resource -> no advisory
OUT=$(co claim --id "$S" --res "other-key" --task edit2 2>&1)
echo "$OUT" | grep -q "CRON ADVISORY" && bad "1g. false-positive advisory on unrelated key" || ok "1g. no advisory for unrelated key"

# ============ 2. check shows advisory too
OUT=$(co check --res "fleet-key" 2>&1)
echo "$OUT" | grep -q "FREE" && ok "2a. check reports free" || bad "2a. check not free"
echo "$OUT" | grep -q "cron 'fleet watchdog'.*fires in ~[0-9]*m" && ok "2b. check carries advisory" || bad "2b. no advisory on check: $OUT"
echo "$OUT" | grep -q "guard will skip that" && ok "2c. non-critical wording = expect skip" || bad "2c. wrong wording"

# ============ 3. status lists upcoming manifested fires
OUT=$(co status 2>&1)
echo "$OUT" | grep -q "scheduled cron jobs w/ declared resources" && ok "3a. status has cron section" || bad "3a. no cron section: $OUT"
echo "$OUT" | grep -q "CONFLICTS with currently HELD" && ok "3b. status flags conflict with live claim" || bad "3b. no conflict flag: $OUT"

# ============ 4. cron-guard defers (skip) when session holds; holder notified
co claim --id "$S" --res "fleet-key" --task "pre-warm" >/dev/null 2>&1
OUT=$(co cron-guard --job watchdogjob02 2>&1); RC=$?
[ $RC -eq 75 ] && ok "4a. guard defers rc 75 while session holds" || bad "4a. rc=$RC out=$OUT"
echo "$OUT" | grep -q "DEFER (skipped)" && ok "4b. stderr says skipped" || bad "4b. $OUT"
INB=$(co inbox --id "$S" 2>&1)
echo "$INB" | grep -q "cron_defer.*fleet watchdog\|fleet watchdog.*politely skipped" && ok "4c. holder got cron_defer note" || bad "4c. inbox: $INB"
echo "$INB" | grep -q "CRITICAL job, consider asking the user to re-run" && ok "4d. note carries re-run guidance" || bad "4d. no guidance"

# ============ 5. cron-guard acquires when free; stdout is ONLY the id; done releases
co release --id "$S" --res "fleet-key" >/dev/null 2>&1
GOUT=$(co cron-guard --job watchdogjob02 2>"$D/gerr.txt"); RC=$?
[ $RC -eq 0 ] && ok "5a. guard acquires rc 0" || bad "5a. rc=$RC"
[ "$(echo "$GOUT" | wc -l | tr -d ' ')" = "1" ] && [[ "$GOUT" =~ ^[0-9a-f]{12}$ ]] && ok "5b. stdout = exactly the 12-hex guard id" || bad "5b. stdout polluted: '$GOUT'"
grep -q "CLAIMED" "$D/gerr.txt" && ok "5c. claim chatter went to stderr" || bad "5c. stderr empty"
OUT=$(co status 2>&1)
echo "$OUT" | grep -q "cron: fleet watchdog" && ok "5d. board shows cron session label" || bad "5d. $OUT"
# a session bumping into it sees CRON JOB wording
S2=$(co register --task "wants fleet" | head -1)
OUT=$(co claim --id "$S2" --res "fleet-key" --task x 2>&1); RC=$?
[ $RC -eq 75 ] && echo "$OUT" | grep -q "CRON JOB" && ok "5e. HELD names CRON JOB holder" || bad "5e. rc=$RC $OUT"
echo "$OUT" | grep -q "crons cannot checkpoint/pause\|Crons cannot checkpoint/pause" && ok "5f. cron-holder guidance present" || bad "5f. $OUT"
# preempt refuses vs cron even with user rank
co prioritize --session "$S2" --rank 1 >/dev/null 2>&1
OUT=$(co preempt --id "$S2" --res "fleet-key" 2>&1); RC=$?
[ $RC -eq 75 ] && echo "$OUT" | grep -q "cannot checkpoint/pause" && ok "5g. preempt refused vs cron w/ explanation" || bad "5g. rc=$RC $OUT"
co done --id "$GOUT" >/dev/null 2>&1
OUT=$(co claim --id "$S2" --res "fleet-key" --task x 2>&1)
echo "$OUT" | grep -q "CLAIMED" && ok "5h. after guard done, session claims fine" || bad "5h. $OUT"
co done --id "$S2" >/dev/null 2>&1
co done --id "$S" >/dev/null 2>&1   # S releases tree/other-key from sections 1-4

# ============ 6. cron-guard --policy wait blocks until release, then acquires
S3=$(co register --task "short edit" | head -1)
OUT=$(co claim --id "$S3" --res "file:$D/tree" --task edit 2>&1)
echo "$OUT" | grep -q "CLAIMED" || bad "6-pre. S3 could not claim tree: $OUT"
( sleep 6; co release --id "$S3" --res "file:$D/tree" >/dev/null 2>&1 ) &
T0=$(date +%s)
GOUT=$(co cron-guard --job backupjob0001 --policy wait --timeout 30 2>"$D/gerr2.txt"); RC=$?
T1=$(date +%s)
[ $RC -eq 0 ] && ok "6a. wait-policy guard acquired after release" || bad "6a. rc=$RC $(cat $D/gerr2.txt)"
[ $((T1-T0)) -ge 4 ] && ok "6b. guard actually waited (${T1}-${T0}=$((T1-T0))s)" || bad "6b. too fast: $((T1-T0))s"
wait
co done --id "$GOUT" >/dev/null 2>&1

# ============ 7. wait-for-cron: session steps aside, cron runs, session resumes
NEXT20S=$(python3 -c "from datetime import datetime,timedelta; print((datetime.now().astimezone()+timedelta(seconds=20)).isoformat())")
python3 - "$D" "$NEXT20S" <<'PYEOF'
import json, sys
d, ts = sys.argv[1], sys.argv[2]
j = json.load(open(f"{d}/jobs.json"))
for job in j["jobs"]:
    if job["id"] == "backupjob0001":
        job["next_run_at"] = ts
json.dump(j, open(f"{d}/jobs.json", "w"))
PYEOF
S4=$(co register --task "wants to edit tree around backup" | head -1)
# simulate the cron firing shortly: background guard acquires then finishes
( sleep 5; G=$(co cron-guard --job backupjob0001 2>/dev/null); sleep 3; co done --id "$G" >/dev/null 2>&1 ) &
OUT=$(co wait-for-cron --id "$S4" --job backupjob0001 --timeout 60 2>&1); RC=$?
[ $RC -eq 0 ] && echo "$OUT" | grep -q "CRON RAN" && ok "7a. wait-for-cron saw the tick complete (rc 0)" || bad "7a. rc=$RC $OUT"
wait
# deferred branch: session HOLDS the resource, cron skips, waiter told rc 2
co claim --id "$S4" --res "file:$D/tree" --task edit >/dev/null 2>&1
( sleep 4; co cron-guard --job backupjob0001 --policy skip >/dev/null 2>&1 ) &
OUT=$(co wait-for-cron --id "$S4" --job backupjob0001 --timeout 45 2>&1); RC=$?
[ $RC -eq 2 ] && echo "$OUT" | grep -q "CRON DEFERRED" && ok "7b. wait-for-cron reports deferral rc 2 (you still hold)" || bad "7b. rc=$RC $OUT"
wait
# beyond-timeout ETA branch: job fires in ~10m but timeout is 60s -> fast refusal
OUT=$(co wait-for-cron --id "$S4" --job unguardedjob3 --timeout 60 2>&1); RC=$?
[ $RC -eq 75 ] && echo "$OUT" | grep -q "beyond --timeout" && ok "7c. far-future fire refused fast w/ guidance" || bad "7c. rc=$RC $OUT"

# ============ 8. cron-note lifecycle + done-warning for unresolved pause
OUT=$(co cron-note --id "$S4" --job backupjob0001 --action paused --reason "editing tree, user approved pause" 2>&1)
echo "$OUT" | grep -q "NOTED.*paused" && ok "8a. cron-note paused recorded" || bad "8a. $OUT"
echo "$OUT" | grep -q "REMINDER: a paused job does not fire AT ALL" && ok "8b. pause reminder shown" || bad "8b. $OUT"
OUT=$(co done --id "$S4" 2>&1)
echo "$OUT" | grep -q "UNRESOLVED PAUSED CRON.*nightly backup" && ok "8c. done warns about never-resumed paused cron" || bad "8c. $OUT"
# resumed case: no warning
S5=$(co register --task another | head -1)
co cron-note --id "$S5" --job backupjob0001 --action paused >/dev/null 2>&1
co cron-note --id "$S5" --job backupjob0001 --action resumed >/dev/null 2>&1
OUT=$(co done --id "$S5" 2>&1)
echo "$OUT" | grep -q "UNRESOLVED PAUSED CRON" && bad "8d. false warning after resume" || ok "8d. resumed pause not flagged"

# ============ 9. resolve by name fragment + ambiguity
OUT=$(co cron-note --id "$S5" --job "fleet" --action triggered 2>&1)
echo "$OUT" | grep -q "NOTED: cron 'fleet watchdog' triggered" && ok "9a. name-fragment resolution" || bad "9a. $OUT"
OUT=$(co cron-note --id "$S5" --job "job" --action paused 2>&1); RC=$?
[ $RC -eq 1 ] && echo "$OUT" | grep -qi "ambiguous\|no cron job" && ok "9b. ambiguous fragment rejected" || bad "9b. rc=$RC $OUT"

# ============ 10. no manifest/store -> everything inert, still works
export HERMES_COORD_CRON_MANIFEST="$D/nope.json" HERMES_COORD_CRON_JOBS="$D/nope2.json"
S6=$(co register --task plain | head -1)
OUT=$(co claim --id "$S6" --res "file:$D/tree" --task t 2>&1); RC=$?
[ $RC -eq 0 ] && ! echo "$OUT" | grep -q "CRON" && ok "10a. absent manifest: claims clean, no cron noise" || bad "10a. rc=$RC $OUT"
OUT=$(co cron-guard --job whatever 2>&1); RC=$?
[ $RC -eq 1 ] && ok "10b. guard errors cleanly w/o store (rc 1)" || bad "10b. rc=$RC"
OUT=$(co cron-guard --name adhoc --res "adhoc-key" 2>/dev/null); RC=$?
[ $RC -eq 0 ] && ok "10c. guard works manifest-less with explicit --res" || bad "10c. rc=$RC"
co done --id "$OUT" >/dev/null 2>&1

# ============ 11. v2.2 BOT LEG: per-profile cron stores (Bot Mode routines)
# A Bot Mode bot is a Hermes profile; its Routines live in the profile's OWN
# cron store. The radar must merge every profile store with the default one,
# attribute profile jobs as "[bot:<profile>]", survive broken stores, and
# resolve id collisions default-store-first.
export HERMES_COORD_CRON_MANIFEST="$D/manifest2.json" HERMES_COORD_CRON_JOBS="$D/jobs2.json"
NEXT8=$(python3 -c "from datetime import datetime,timedelta; print((datetime.now().astimezone()+timedelta(minutes=8)).isoformat())")
NEXT12=$(python3 -c "from datetime import datetime,timedelta; print((datetime.now().astimezone()+timedelta(minutes=12)).isoformat())")
mkdir -p "$D/profiles/researcher/cron" "$D/profiles/broken/cron"
# Default store: one job + a colliding id (the default copy must win).
cat > "$D/jobs2.json" <<JSON
{"jobs":[
 {"id":"defaultjob001","name":"default sweep","enabled":true,"next_run_at":"$NEXT12"},
 {"id":"sharedid00001","name":"default owner","enabled":true,"next_run_at":"$NEXT12"}
]}
JSON
# Researcher bot store: routine on a shared box, a pre-tagged name (must not
# get double-tagged), and a duplicate of the colliding id (must lose).
cat > "$D/profiles/researcher/cron/jobs.json" <<JSON
{"jobs":[
 {"id":"researcherjb1","name":"morning digest","enabled":true,"next_run_at":"$NEXT8"},
 {"id":"researcherjb2","name":"[bot:researcher] tagged already","enabled":true,"next_run_at":"$NEXT12"},
 {"id":"sharedid00001","name":"bot impostor","enabled":true,"next_run_at":"$NEXT12"}
]}
JSON
# A corrupt store must contribute nothing without breaking the scan.
echo "NOT JSON {" > "$D/profiles/broken/cron/jobs.json"
cat > "$D/manifest2.json" <<JSON
{"jobs":{
 "researcherjb1":{"name":"morning digest","resources":["box:gpu1"],"policy":"skip","critical":false},
 "researcherjb2":{"name":"tagged already","resources":["res:tagcheck"],"policy":"skip","critical":false},
 "defaultjob001":{"name":"default sweep","resources":["res:defaultkey"],"policy":"skip","critical":false}
}}
JSON
# 11a. status radar sees the bot routine, tagged with its owning bot
OUT=$(co status 2>&1)
echo "$OUT" | grep -q "\[bot:researcher\] morning digest" && ok "11a. radar shows bot routine w/ [bot:] tag" || bad "11a. $OUT"
# 11b. claim advisory on the shared box names the bot job
S7=$(co register --task "gpu work" | head -1)
OUT=$(co claim --id "$S7" --res "box:gpu1" --task gpu 2>&1)
echo "$OUT" | grep -q "CRON ADVISORY.*\[bot:researcher\] morning digest" && ok "11b. claim advisory names bot routine" || bad "11b. $OUT"
# 11c. cron-guard resolves a bot-store job id and defers vs the live claim
OUT=$(co cron-guard --job researcherjb1 2>&1); RC=$?
[ $RC -eq 75 ] && echo "$OUT" | grep -q "DEFER" && ok "11c. guard resolves bot job + defers vs holder" || bad "11c. rc=$RC $OUT"
co done --id "$S7" >/dev/null 2>&1
# ...and acquires once free
GOUT=$(co cron-guard --job researcherjb1 2>/dev/null); RC=$?
[ $RC -eq 0 ] && [[ "$GOUT" =~ ^[0-9a-f]{12}$ ]] && ok "11d. guard acquires bot job once free" || bad "11d. rc=$RC '$GOUT'"
co done --id "$GOUT" >/dev/null 2>&1
# 11e-g. load the module straight from $SC (portable — no installed paths)
# and inspect the merged view: tag hygiene, collision winner, attribution.
OUT=$(python3 - "$SC" <<'PYMOD'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("sc", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
j = m.cron_store_jobs()
print(j["researcherjb2"]["name"])
print(j["sharedid00001"]["name"])
print(j["researcherjb1"]["profile"])
PYMOD
)
echo "$OUT" | sed -n 1p | grep -qx "\[bot:researcher\] tagged already" && ok "11e. pre-tagged name not double-tagged" || bad "11e. $OUT"
# 11f. id collision across stores: default store wins
echo "$OUT" | sed -n 2p | grep -qx "default owner" && ok "11f. id collision -> default store wins" || bad "11f. $OUT"
# 11g. profile attribution recorded; broken store contributed nothing (no crash)
echo "$OUT" | sed -n 3p | grep -qx "researcher" && ok "11g. profile field carries bot name (broken store inert)" || bad "11g. $OUT"

# --- 12. bot ENROLLMENT audit: status flags persona-bearing profiles whose
#     SOUL.md lacks the enrollment marker; profiles with no persona are never
#     flagged (nothing proves they are bots); adding the marker clears it.
cat > "$D/profiles/researcher/SOUL.md" <<'EOF'
# Researcher
You are the researcher bot. (No coordination blurb yet.)
EOF
OUT=$(co status 2>&1)
echo "$OUT" | grep -q "UNENROLLED bot profiles (1)" && echo "$OUT" | grep -q "researcher" \
  && ok "12a. status flags unenrolled persona profile (SOUL-less 'broken' not flagged)" || bad "12a. $OUT"
JOUT=$(co status --json 2>/dev/null)
echo "$JOUT" | grep -q '"unenrolled_bot_profiles".*researcher' && ok "12b. --json carries unenrolled_bot_profiles" || bad "12b. $JOUT"
printf '\nEnrollment marker: session-coord (bot-wire v1) — installer idempotence + board audit.\n' >> "$D/profiles/researcher/SOUL.md"
OUT=$(co status 2>&1)
echo "$OUT" | grep -q "UNENROLLED" && bad "12c. marker should clear the flag: $OUT" || ok "12c. enrollment marker clears the flag"

# --- 13. non-bot PROFILE audit: a SOUL-less profile whose own memory store
#     exists without the standing rule is flagged UNWIRED; the rule clears
#     it; a profile with no memory store yet is never flagged (no evidence).
mkdir -p "$D/profiles/plainprofile/memories" "$D/profiles/freshprofile"
echo "some unrelated memory entry" > "$D/profiles/plainprofile/memories/MEMORY.md"
OUT=$(co status 2>&1)
echo "$OUT" | grep -q "UNWIRED profiles (1)" && echo "$OUT" | grep -q "plainprofile" \
  && ok "13a. status flags unwired non-bot profile (store-less 'freshprofile' not flagged)" || bad "13a. $OUT"
JOUT=$(co status --json 2>/dev/null)
echo "$JOUT" | grep -q '"unwired_profiles".*plainprofile' && ok "13b. --json carries unwired_profiles" || bad "13b. $JOUT"
printf '\nSTANDING RULE — session-coord (wire v1): ALWAYS deconflict via the coordination board.\n' >> "$D/profiles/plainprofile/memories/MEMORY.md"
OUT=$(co status 2>&1)
echo "$OUT" | grep -q "UNWIRED" && bad "13c. wire marker should clear the flag: $OUT" || ok "13c. standing-rule marker clears the flag"

echo
echo "RESULT: $PASS passed, $FAIL failed"
rm -rf "$D"
exit $([ $FAIL -eq 0 ] && echo 0 || echo 1)
