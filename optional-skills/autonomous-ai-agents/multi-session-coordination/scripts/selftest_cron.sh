#!/bin/bash
# End-to-end test of the v2.1 CRON LEG + v2.2 BOT LEG of session_coord.py.
# Everything runs on scratch DB + scratch manifests + scratch cron stores.
SC="${SC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/session_coord.py}"
SCRIPT_DIR=$(cd "$(dirname "$SC")" && pwd)
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
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
if echo "$OUT" | grep -q "CLAIMED"; then ok "1a. claim succeeds (advisory never blocks)"; else bad "1a. claim blocked: $OUT"; fi
if echo "$OUT" | grep -q "CRITICAL cron 'nightly backup'.*fires in ~[0-9]*m"; then ok "1b. critical advisory present w/ ETA"; else bad "1b. no critical advisory: $OUT"; fi
if echo "$OUT" | grep -q "ASK THE USER to pause or trigger it early"; then ok "1c. critical decision options offered"; else bad "1c. options missing"; fi
if echo "$OUT" | grep -q "wait-for-cron --job backupjob000"; then ok "1d. wait-for-cron hint names the job"; else bad "1d. no wait-for-cron hint"; fi
if echo "$OUT" | grep -q "does NOT check the board (unguarded)"; then ok "1e. unguarded job warned (overlaps via dir cover)"; else bad "1e. no unguarded warning: $OUT"; fi
if echo "$OUT" | grep -q "inside your claim's 90m TTL window"; then ok "1f. TTL-overlap callout"; else bad "1f. no TTL callout"; fi

# non-overlapping resource -> no advisory
OUT=$(co claim --id "$S" --res "other-key" --task edit2 2>&1)
if echo "$OUT" | grep -q "CRON ADVISORY"; then bad "1g. false-positive advisory on unrelated key"; else ok "1g. no advisory for unrelated key"; fi

# ============ 2. check shows advisory too
OUT=$(co check --res "fleet-key" 2>&1)
if echo "$OUT" | grep -q "FREE"; then ok "2a. check reports free"; else bad "2a. check not free"; fi
if echo "$OUT" | grep -q "cron 'fleet watchdog'.*fires in ~[0-9]*m"; then ok "2b. check carries advisory"; else bad "2b. no advisory on check: $OUT"; fi
if echo "$OUT" | grep -q "guard will skip that"; then ok "2c. non-critical wording = expect skip"; else bad "2c. wrong wording"; fi

# ============ 3. status lists upcoming manifested fires
OUT=$(co status 2>&1)
if echo "$OUT" | grep -q "scheduled cron jobs w/ declared resources"; then ok "3a. status has cron section"; else bad "3a. no cron section: $OUT"; fi
if echo "$OUT" | grep -q "CONFLICTS with currently HELD"; then ok "3b. status flags conflict with live claim"; else bad "3b. no conflict flag: $OUT"; fi

# ============ 4. cron-guard defers (skip) when session holds; holder notified
co claim --id "$S" --res "fleet-key" --task "pre-warm" >/dev/null 2>&1
OUT=$(co cron-guard --job watchdogjob02 2>&1); RC=$?
if [ $RC -eq 75 ]; then ok "4a. guard defers rc 75 while session holds"; else bad "4a. rc=$RC out=$OUT"; fi
if echo "$OUT" | grep -q "DEFER (skipped)"; then ok "4b. stderr says skipped"; else bad "4b. $OUT"; fi
INB=$(co inbox --id "$S" 2>&1)
if echo "$INB" | grep -q "cron_defer.*fleet watchdog\|fleet watchdog.*politely skipped"; then ok "4c. holder got cron_defer note"; else bad "4c. inbox: $INB"; fi
if echo "$INB" | grep -q "CRITICAL job, consider asking the user to re-run"; then ok "4d. note carries re-run guidance"; else bad "4d. no guidance"; fi

# ============ 5. cron-guard acquires when free; stdout is ONLY the id; done releases
co release --id "$S" --res "fleet-key" >/dev/null 2>&1
GOUT=$(co cron-guard --job watchdogjob02 2>"$D/gerr.txt"); RC=$?
if [ $RC -eq 0 ]; then ok "5a. guard acquires rc 0"; else bad "5a. rc=$RC"; fi
if [ "$(echo "$GOUT" | wc -l | tr -d ' ')" = "1" ] && [[ "$GOUT" =~ ^[0-9a-f]{12}$ ]]; then ok "5b. stdout = exactly the 12-hex guard id"; else bad "5b. stdout polluted: '$GOUT'"; fi
if grep -q "CLAIMED" "$D/gerr.txt"; then ok "5c. claim chatter went to stderr"; else bad "5c. stderr empty"; fi
OUT=$(co status 2>&1)
if echo "$OUT" | grep -q "cron: fleet watchdog"; then ok "5d. board shows cron session label"; else bad "5d. $OUT"; fi
# a session bumping into it sees CRON JOB wording
S2=$(co register --task "wants fleet" | head -1)
OUT=$(co claim --id "$S2" --res "fleet-key" --task x 2>&1); RC=$?
if [ $RC -eq 75 ] && echo "$OUT" | grep -q "CRON JOB"; then ok "5e. HELD names CRON JOB holder"; else bad "5e. rc=$RC $OUT"; fi
if echo "$OUT" | grep -q "crons cannot checkpoint/pause\|Crons cannot checkpoint/pause"; then ok "5f. cron-holder guidance present"; else bad "5f. $OUT"; fi
# preempt refuses vs cron even with user rank
co prioritize --session "$S2" --rank 1 >/dev/null 2>&1
OUT=$(co preempt --id "$S2" --res "fleet-key" 2>&1); RC=$?
if [ $RC -eq 75 ] && echo "$OUT" | grep -q "cannot checkpoint/pause"; then ok "5g. preempt refused vs cron w/ explanation"; else bad "5g. rc=$RC $OUT"; fi
co "done" --id "$GOUT" >/dev/null 2>&1
OUT=$(co claim --id "$S2" --res "fleet-key" --task x 2>&1)
if echo "$OUT" | grep -q "CLAIMED"; then ok "5h. after guard done, session claims fine"; else bad "5h. $OUT"; fi
co "done" --id "$S2" >/dev/null 2>&1
co "done" --id "$S" >/dev/null 2>&1   # S releases tree/other-key from sections 1-4

# ============ 6. cron-guard --policy wait blocks until release, then acquires
S3=$(co register --task "short edit" | head -1)
OUT=$(co claim --id "$S3" --res "file:$D/tree" --task edit 2>&1)
echo "$OUT" | grep -q "CLAIMED" || bad "6-pre. S3 could not claim tree: $OUT"
( sleep 6; co release --id "$S3" --res "file:$D/tree" >/dev/null 2>&1 ) &
T0=$(date +%s)
GOUT=$(co cron-guard --job backupjob0001 --policy wait --timeout 30 2>"$D/gerr2.txt"); RC=$?
T1=$(date +%s)
if [ $RC -eq 0 ]; then ok "6a. wait-policy guard acquired after release"; else bad "6a. rc=$RC $(cat "$D"/gerr2.txt)"; fi
if [ $((T1-T0)) -ge 4 ]; then ok "6b. guard actually waited (${T1}-${T0}=$((T1-T0))s)"; else bad "6b. too fast: $((T1-T0))s"; fi
wait
co "done" --id "$GOUT" >/dev/null 2>&1

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
( sleep 5; G=$(co cron-guard --job backupjob0001 2>/dev/null); sleep 3; co "done" --id "$G" >/dev/null 2>&1 ) &
OUT=$(co wait-for-cron --id "$S4" --job backupjob0001 --timeout 60 2>&1); RC=$?
if [ $RC -eq 0 ] && echo "$OUT" | grep -q "CRON RAN"; then ok "7a. wait-for-cron saw the tick complete (rc 0)"; else bad "7a. rc=$RC $OUT"; fi
wait
# deferred branch: session HOLDS the resource, cron skips, waiter told rc 2
co claim --id "$S4" --res "file:$D/tree" --task edit >/dev/null 2>&1
( sleep 4; co cron-guard --job backupjob0001 --policy skip >/dev/null 2>&1 ) &
OUT=$(co wait-for-cron --id "$S4" --job backupjob0001 --timeout 45 2>&1); RC=$?
if [ $RC -eq 2 ] && echo "$OUT" | grep -q "CRON DEFERRED"; then ok "7b. wait-for-cron reports deferral rc 2 (you still hold)"; else bad "7b. rc=$RC $OUT"; fi
wait
# beyond-timeout ETA branch: job fires in ~10m but timeout is 60s -> fast refusal
OUT=$(co wait-for-cron --id "$S4" --job unguardedjob3 --timeout 60 2>&1); RC=$?
if [ $RC -eq 75 ] && echo "$OUT" | grep -q "beyond --timeout"; then ok "7c. far-future fire refused fast w/ guidance"; else bad "7c. rc=$RC $OUT"; fi

# ============ 8. cron-note lifecycle + done-warning for unresolved pause
OUT=$(co cron-note --id "$S4" --job backupjob0001 --action paused --reason "editing tree, user approved pause" 2>&1)
if echo "$OUT" | grep -q "NOTED.*paused"; then ok "8a. cron-note paused recorded"; else bad "8a. $OUT"; fi
if echo "$OUT" | grep -q "REMINDER: a paused job does not fire AT ALL"; then ok "8b. pause reminder shown"; else bad "8b. $OUT"; fi
OUT=$(co "done" --id "$S4" 2>&1)
if echo "$OUT" | grep -q "UNRESOLVED PAUSED CRON.*nightly backup"; then ok "8c. done warns about never-resumed paused cron"; else bad "8c. $OUT"; fi
# resumed case: no warning
S5=$(co register --task another | head -1)
co cron-note --id "$S5" --job backupjob0001 --action paused >/dev/null 2>&1
co cron-note --id "$S5" --job backupjob0001 --action resumed >/dev/null 2>&1
OUT=$(co "done" --id "$S5" 2>&1)
if echo "$OUT" | grep -q "UNRESOLVED PAUSED CRON"; then bad "8d. false warning after resume"; else ok "8d. resumed pause not flagged"; fi

# ============ 9. resolve by name fragment + ambiguity
OUT=$(co cron-note --id "$S5" --job "fleet" --action triggered 2>&1)
if echo "$OUT" | grep -q "NOTED: cron 'fleet watchdog' triggered"; then ok "9a. name-fragment resolution"; else bad "9a. $OUT"; fi
OUT=$(co cron-note --id "$S5" --job "job" --action paused 2>&1); RC=$?
if [ $RC -eq 1 ] && echo "$OUT" | grep -qi "ambiguous\|no cron job"; then ok "9b. ambiguous fragment rejected"; else bad "9b. rc=$RC $OUT"; fi

# ============ 10. no manifest/store -> everything inert, still works
export HERMES_COORD_CRON_MANIFEST="$D/nope.json" HERMES_COORD_CRON_JOBS="$D/nope2.json"
S6=$(co register --task plain | head -1)
OUT=$(co claim --id "$S6" --res "file:$D/tree" --task t 2>&1); RC=$?
if [ $RC -eq 0 ] && ! echo "$OUT" | grep -q "CRON"; then ok "10a. absent manifest: claims clean, no cron noise"; else bad "10a. rc=$RC $OUT"; fi
OUT=$(co cron-guard --job whatever 2>&1); RC=$?
if [ $RC -eq 1 ]; then ok "10b. guard errors cleanly w/o store (rc 1)"; else bad "10b. rc=$RC"; fi
OUT=$(co cron-guard --name adhoc --res "adhoc-key" 2>/dev/null); RC=$?
if [ $RC -eq 0 ]; then ok "10c. guard works manifest-less with explicit --res"; else bad "10c. rc=$RC"; fi
co "done" --id "$OUT" >/dev/null 2>&1

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
if echo "$OUT" | grep -q "\[bot:researcher\] morning digest"; then ok "11a. radar shows bot routine w/ [bot:] tag"; else bad "11a. $OUT"; fi
# 11b. claim advisory on the shared box names the bot job
S7=$(co register --task "gpu work" | head -1)
OUT=$(co claim --id "$S7" --res "box:gpu1" --task gpu 2>&1)
if echo "$OUT" | grep -q "CRON ADVISORY.*\[bot:researcher\] morning digest"; then ok "11b. claim advisory names bot routine"; else bad "11b. $OUT"; fi
# 11c. cron-guard resolves a bot-store job id and defers vs the live claim
OUT=$(co cron-guard --job researcherjb1 2>&1); RC=$?
if [ $RC -eq 75 ] && echo "$OUT" | grep -q "DEFER"; then ok "11c. guard resolves bot job + defers vs holder"; else bad "11c. rc=$RC $OUT"; fi
co "done" --id "$S7" >/dev/null 2>&1
# ...and acquires once free
GOUT=$(co cron-guard --job researcherjb1 2>/dev/null); RC=$?
if [ $RC -eq 0 ] && [[ "$GOUT" =~ ^[0-9a-f]{12}$ ]]; then ok "11d. guard acquires bot job once free"; else bad "11d. rc=$RC '$GOUT'"; fi
co "done" --id "$GOUT" >/dev/null 2>&1
# 11e-g. load the module straight from $SC (portable — no installed paths)
# and inspect the merged view: tag hygiene, collision winner, attribution.
OUT=$(python3 - "$SC" <<'PYMOD'
import importlib.util, os, sys
# A file loader does not add its directory to sys.path; use Python-native argv.
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[1])))
spec = importlib.util.spec_from_file_location("sc", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
j = m.cron_store_jobs()
print(j["researcherjb2"]["name"])
print(j["sharedid00001"]["name"])
print(j["researcherjb1"]["profile"])
PYMOD
)
if echo "$OUT" | sed -n 1p | grep -qx "\[bot:researcher\] tagged already"; then ok "11e. pre-tagged name not double-tagged"; else bad "11e. $OUT"; fi
# 11f. id collision across stores: default store wins
if echo "$OUT" | sed -n 2p | grep -qx "default owner"; then ok "11f. id collision -> default store wins"; else bad "11f. $OUT"; fi
# 11g. profile attribution recorded; broken store contributed nothing (no crash)
if echo "$OUT" | sed -n 3p | grep -qx "researcher"; then ok "11g. profile field carries bot name (broken store inert)"; else bad "11g. $OUT"; fi

# --- 12. bot ENROLLMENT audit: status flags persona-bearing profiles whose
#     SOUL.md lacks the enrollment marker; profiles with no persona are never
#     flagged (nothing proves they are bots); only the exact managed block clears it.
cat > "$D/profiles/researcher/SOUL.md" <<'EOF'
# Researcher
You are the researcher bot. (No coordination blurb yet.)
EOF
OUT=$(co status 2>&1)
if echo "$OUT" | grep -q "UNENROLLED bot profiles (1)" && echo "$OUT" | grep -q "researcher"; then ok "12a. status flags unenrolled persona profile (SOUL-less 'broken' not flagged)"; else bad "12a. $OUT"; fi
JOUT=$(co status --json 2>/dev/null)
if echo "$JOUT" | grep -q '"unenrolled_bot_profiles".*researcher'; then ok "12b. --json carries unenrolled_bot_profiles"; else bad "12b. $JOUT"; fi
cat >> "$D/profiles/researcher/SOUL.md" <<'EOF'

<!-- BEGIN session-coord managed bot-board-v2 -->
## Shared-resource coordination (Hermes co-worker protocol)

You share this machine with interactive sessions, other bots, and scheduled
jobs. Before mutating anything shared, consult the coordination board.

Shared resources include remote machines (`box:<host>`), the desktop UI
(`ui:desktop`), singleton apps, shared project directories (`file:~/...`),
shared skills/scripts, and the machine's main agent memory store.

The only claim-free exemption is the profile's own internal memory, sessions,
and cron store. Files or directories created in shared space are still shared;
register and claim them. Profile memory is not the machine's main memory store.

```bash
SC="<session_coord_path>"
CID=$(python3 "$SC" register --task "<task>" --surface "bot:researcher" | head -1)
python3 "$SC" claim --id "$CID" --res "<key>" --res "<key2>" --task "<task>"
# Continue only after CLAIMED. If HELD/QUEUED, do not mutate the requested resources.
# A shell actor that must stay alive may use one bounded --wait call.
# ... do the whole task ...
python3 "$SC" done --id "$CID"
```

Rules:
- Claim every resource for the task up front in one atomic call. Hold the set
  for the whole task and release it with `done`.
- Exit 75 means another actor is ahead. Do not mutate the requested resources.
  Use `check`/`inbox`, wait for release, or use a bounded `--wait` shell call.
- On a preempt/priority request, finish the current atomic step, save progress,
  run `pause --id "$CID" --note "<progress file>"`, and report the pause.
- Priorities are user-set. Never rank, preempt, or `steal` on your own.
- Chat can negotiate an ETA but never authorizes mutation; only CLAIMED does.
- If the board errors, report that coordination is unavailable and do not
  mutate shared resources blind.

Enrollment marker: session-coord (bot-board-wire v2).
<!-- END session-coord managed bot-board-v2 -->
EOF
OUT=$(co status 2>&1)
if echo "$OUT" | grep -q "UNENROLLED"; then bad "12c. managed block should clear the flag: $OUT"; else ok "12c. managed block clears the flag"; fi

# --- 13. non-bot PROFILE audit: a SOUL-less profile whose own memory store
#     exists without the standing rule is flagged UNWIRED; the rule clears
#     it; a profile with no memory store yet is never flagged (no evidence).
mkdir -p "$D/profiles/plainprofile/memories" "$D/profiles/freshprofile"
echo "some unrelated memory entry" > "$D/profiles/plainprofile/memories/MEMORY.md"
OUT=$(co status 2>&1)
if echo "$OUT" | grep -q "UNWIRED profiles (1)" && echo "$OUT" | grep -q "plainprofile"; then ok "13a. status flags unwired non-bot profile (store-less 'freshprofile' not flagged)"; else bad "13a. $OUT"; fi
JOUT=$(co status --json 2>/dev/null)
if echo "$JOUT" | grep -q '"unwired_profiles".*plainprofile'; then ok "13b. --json carries unwired_profiles"; else bad "13b. $JOUT"; fi
cat >> "$D/profiles/plainprofile/memories/MEMORY.md" <<'EOF'
<!-- BEGIN session-coord managed board-v2 -->
STANDING RULE — session-coord (board-wire v2): Before mutating shared resources (files, skills, memory, cron store, remote boxes, or desktop UI), first run `python3 /fixture/session_coord.py status`. Register once per task with `ID=$(python3 /fixture/session_coord.py register --task '...' --surface <surface>)`, then atomically claim every required resource with `python3 /fixture/session_coord.py claim --id $ID --res <key> [--res <key2>]`. Continue only after CLAIMED; when HELD/QUEUED, do not mutate the requested resources—use `check`, `inbox`, or one bounded `--wait` shell call and wait for release. Run `python3 /fixture/session_coord.py done --id $ID` at task end. Priorities are user-set; never `steal` without explicit approval. Full protocol: skill multi-session-coordination. Off-switch: `python3 /fixture/session_coord.py disable`.
<!-- END session-coord managed board-v2 -->
EOF
OUT=$(co status 2>&1)
if echo "$OUT" | grep -q "UNWIRED"; then bad "13c. managed rule should clear the flag: $OUT"; else ok "13c. managed standing rule clears the flag"; fi

echo
echo "RESULT: $PASS passed, $FAIL failed"
rm -rf "$D"
if [ "$FAIL" -eq 0 ]; then exit 0; else exit 1; fi
