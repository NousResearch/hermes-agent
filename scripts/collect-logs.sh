#!/usr/bin/env bash
# collect-logs - rebuilds <drive>/logs/ and <drive>/logs.zip (ledger snapshot + git state + manifest + completeness self-check). Run any time by hand, and at the end of every agent task that touched the repo or the ledger.
#
# POSIX/bash mirror of collect-logs.ps1 - identical output layout and checks. Use whichever fits
# your shell; they produce the same D:\logs\ tree and the same D:\logs.zip.
#
#   scripts/collect-logs.sh  [--out <dir>]  [--repo <dir>]  [--quiet]
#
# Builds:
#   <out>/ledger/             verbatim copy of <repo>/logs/ledger/  (the tracked project ledger)
#   <out>/repo-state.txt      git snapshot - HEAD, branch, ahead/behind origin & upstream, uncommitted, recent log
#   <out>/HANDOFF-INDEX.md    generated manifest + completeness-check results
#   <out>/redaction-report.txt what the mandatory redaction pass touched
#   <out>/repo-runtime-logs/  <repo>/logs/*.log|*.jsonl  - EXCLUDED by default; --include-runtime-logs to include
#   <out>/*.md                LEFT UNTOUCHED by staging - session reports humans drop here (redaction still scans them)
#
# Before zipping, EVERY text file in <out> is run through the agent's production credential
# redactor (scripts/redact_handoff.py -> agent.redact). If a likely secret SURVIVES, the zip
# is NOT created: the run fails, names the file:line, and renames any previous zip to <out>.zip.stale.
# A Python that can import agent.redact is REQUIRED (repo .venv / sibling -venv / PATH).
#
# then zips <out> to <out>.zip (default D:/logs.zip), overwriting, writes <out>.zip.sha256,
# and prints a PASS/WARN/FAIL report.  Exit 0 = no FAIL (WARN ok); exit 1 = a FAIL (bundle incomplete).
# Needs: bash, coreutils, python. Uses git if on PATH (else snapshot skipped, noted). Zips via zip|python.
set -euo pipefail

QUIET=0; OUT=""; REPO=""; INCLUDE_RT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out)   OUT="$2"; shift 2 ;;
    --repo)  REPO="$2"; shift 2 ;;
    --quiet) QUIET=1; shift ;;
    --include-runtime-logs) INCLUDE_RT=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -n "$REPO" ] || REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(cd "$REPO" && pwd)"

LEDGER_SRC="$REPO/logs/ledger"
if [ ! -d "$LEDGER_SRC" ]; then
  echo "No ledger at '$LEDGER_SRC' - is --repo correct? Cannot build a handoff without it." >&2
  exit 1
fi

if [ -z "$OUT" ]; then
  # parent of the repo = its drive root in practice: /d/north-forge-agent -> /d/logs ; D:/north-forge-agent -> D:/logs
  OUT="$(dirname "$REPO")/logs"
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
ZIP="${OUT%/}.zip"

case "$OUT/" in
  "$REPO"/*) echo "--out '$OUT' is inside the repo. Pick a location outside '$REPO' (default D:/logs)." >&2; exit 1 ;;
esac

STAMP_ISO="$(date '+%Y-%m-%d %H:%M:%S %z')"
TODAY="$(date '+%Y-%m-%d')"
HOST="$(hostname 2>/dev/null || echo '?')"

CHECKS=()   # each entry: "LEVEL<TAB>text"
add_check() { CHECKS+=("$1"$'\t'"$2"); }

# --- 1. refresh ledger copy --------------------------------------------
rm -rf "$OUT/ledger"
mkdir -p "$OUT/ledger"
( cd "$LEDGER_SRC" && tar cf - . ) | ( cd "$OUT/ledger" && tar xf - )

missing=""
while IFS= read -r -d '' f; do
  rel="${f#"$LEDGER_SRC"/}"
  if [ ! -f "$OUT/ledger/$rel" ] || [ "$(wc -c <"$f")" != "$(wc -c <"$OUT/ledger/$rel")" ]; then
    missing="$missing $rel"
  fi
done < <(find "$LEDGER_SRC" -type f -print0)
src_count="$(find "$LEDGER_SRC" -type f | wc -l | tr -d ' ')"
if [ -z "$missing" ]; then
  add_check OK "ledger copied in full - $src_count file(s) under $(basename "$OUT")/ledger/"
else
  add_check FAIL "ledger copy incomplete - missing/size-mismatch:$missing"
fi

bad=""
for n in README.md INDEX.md CHANGELOG.md errors/ERROR-LOG.md decisions/DECISION-LOG.md; do
  [ -s "$OUT/ledger/$n" ] || bad="$bad $n"
done
audit_count="$(find "$OUT/ledger/audits" -name 'AUDIT-*.md' 2>/dev/null | wc -l | tr -d ' ')"
tmpl_count="$(find "$OUT/ledger/templates" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')"
[ -z "$bad" ] || add_check FAIL "ledger missing/empty:$bad"
if [ "$audit_count" -lt 1 ]; then add_check WARN "ledger has no AUDIT- file"; else add_check OK "ledger carries $audit_count audit(s), $tmpl_count template(s)"; fi
[ "$tmpl_count" -ge 4 ] || add_check WARN "ledger templates/ has only $tmpl_count file(s) - expected 4"

# --- 2. repo runtime logs (EXCLUDED by default) ---------------------
rm -rf "$OUT/repo-runtime-logs"
rt_avail=0; rt_count=0
if [ -d "$REPO/logs" ]; then
  while IFS= read -r -d '' p; do
    rt_avail=$((rt_avail+1))
    if [ "$INCLUDE_RT" -eq 1 ]; then
      [ "$rt_count" -eq 0 ] && mkdir -p "$OUT/repo-runtime-logs"
      cp -f "$p" "$OUT/repo-runtime-logs/"
      rt_count=$((rt_count+1))
    fi
  done < <(find "$REPO/logs" -maxdepth 1 -type f \( -name '*.log' -o -name '*.jsonl' \) -print0 2>/dev/null)
fi
if [ "$INCLUDE_RT" -eq 1 ]; then
  if [ "$rt_count" -gt 0 ]; then add_check WARN "--include-runtime-logs: copied $rt_count repo runtime log(s) - high-risk content, still redacted before zip"
  else add_check OK "--include-runtime-logs set, but no repo runtime *.log / *.jsonl exist"; fi
elif [ "$rt_avail" -gt 0 ]; then
  add_check OK "$rt_avail repo runtime log(s) EXCLUDED (default) - pass --include-runtime-logs to include"
fi

# --- 3. drop legacy nested bundles --------------------------------
for z in "$OUT"/north-forge-agent-logs-*.zip; do
  [ -e "$z" ] || continue
  rm -f "$z"; add_check OK "removed superseded nested bundle $(basename "$z") (its content is now ledger/)"
done

# --- 4. git snapshot -> repo-state.txt ---------------------------
RS="$OUT/repo-state.txt"
ahead=""; behind=""; dirty_n=0; dirty_list=""; GIT_OK=0; HEAD_SHA=""; BRANCH=""
{
  echo "repo-state.txt   generated $STAMP_ISO"
  echo "host $HOST   by ${USER:-?}   via scripts/collect-logs.sh"
  echo "repo $REPO"
  printf '%.0s=' $(seq 1 72); echo
} >"$RS"
if command -v git >/dev/null 2>&1; then
  g() { git -C "$REPO" "$@" 2>&1 || true; }
  HEAD_SHA="$(g rev-parse HEAD)"
  BRANCH="$(g rev-parse --abbrev-ref HEAD)"
  GIT_OK=1
  {
    echo "HEAD    $HEAD_SHA"
    echo "branch  $BRANCH"
    echo
    echo "--- git status -sb ---";  g status -sb
    echo
    echo "--- vs origin/main  (left=local ahead, right=behind) ---"
    g rev-list --left-right --count 'HEAD...origin/main'
    echo "--- vs upstream/main ---"; g rev-list --left-right --count 'HEAD...upstream/main'
    echo
    echo "--- git log --oneline -15 ---"; g log --oneline -15
    echo
    echo "--- remotes ---"; g remote -v
    echo
    echo "--- uncommitted (git status --porcelain) ---"
    porc="$(g status --porcelain)"
    if [ -n "$porc" ]; then echo "$porc"; else echo "(working tree clean)"; fi
  } >>"$RS"
  ro="$(git -C "$REPO" rev-list --left-right --count 'HEAD...origin/main' 2>/dev/null || echo '')"
  if printf '%s' "$ro" | grep -qE '^[0-9]+[[:space:]]+[0-9]+$'; then
    ahead="$(printf '%s' "$ro" | awk '{print $1}')"; behind="$(printf '%s' "$ro" | awk '{print $2}')"
  fi
  porc="$(git -C "$REPO" status --porcelain 2>/dev/null || true)"
  if [ -n "$porc" ]; then
    dirty_n="$(printf '%s\n' "$porc" | grep -c . || true)"
    dirty_list="$(printf '%s\n' "$porc" | sed 's/^[[:space:]]*//' | head -12 | paste -sd'~' - | sed 's/~/; /g')"
    [ "$dirty_n" -gt 12 ] && dirty_list="$dirty_list ..."
  fi
else
  echo "git not found on PATH - snapshot skipped." >>"$RS"
fi

if [ "$GIT_OK" -eq 0 ]; then
  add_check WARN "git not on PATH - repo-state.txt has no snapshot; verify branch state by hand"
else
  { [ -n "$behind" ] && [ "$behind" -gt 0 ]; } && add_check WARN "local HEAD is $behind commit(s) BEHIND origin/main - checkout may be stale"
  if [ -n "$ahead" ]; then
    if [ "$ahead" -gt 0 ]; then add_check OK "$ahead commit(s) held on local main, not pushed (expected - review-before-push)"
    else add_check OK "local main is level with origin/main"; fi
  fi
  if [ "$dirty_n" -gt 0 ]; then
    add_check WARN "working tree has $dirty_n uncommitted change(s): $dirty_list - commit intended edits before treating this bundle as final"
  else
    add_check OK "working tree clean"
  fi
fi

# --- 5. session-report freshness -----------------------------
report_max=""; report_n=0
for f in "$OUT"/*.md; do
  [ -e "$f" ] || continue
  b="$(basename "$f")"; [ "$b" = "HANDOFF-INDEX.md" ] && continue
  report_n=$((report_n+1))
  d="$(printf '%s' "$b" | grep -oE '20[0-9]{2}-[0-9]{2}-[0-9]{2}' | sort | tail -1 || true)"
  [ -n "$d" ] && { [ -z "$report_max" ] || [ "$d" \> "$report_max" ]; } && report_max="$d"
done
ledger_max="$(grep -rhoE '(CHG|ERR|DECISION|AUDIT|RUN)-20[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{3}' "$OUT/ledger" 2>/dev/null | grep -oE '20[0-9]{2}-[0-9]{2}-[0-9]{2}' | sort | tail -1 || true)"
if [ "$report_n" -eq 0 ]; then
  add_check WARN "no session-report *.md in $OUT - a handoff normally carries a plain-language write-up"
else
  add_check OK "$report_n session report(s) present (newest dated ${report_max:-?})"
fi
if [ -n "$ledger_max" ] && [ -n "$report_max" ] && [ "$ledger_max" \> "$report_max" ]; then
  add_check WARN "ledger has entries dated $ledger_max but the newest session report is $report_max - a run may not have left its handoff note"
fi
if [ -n "$ledger_max" ] && [ -z "$report_max" ]; then
  add_check WARN "ledger active ($ledger_max) but no dated session report found"
fi

# --- 5b. resolve a Python that can import agent.redact ------
REDACT_PY="$REPO/scripts/redact_handoff.py"
PYRED=""
sib="$(dirname "$REPO")/$(basename "$REPO")-venv"
for cand in \
    "$REPO/.venv/Scripts/python.exe" "$REPO/.venv/bin/python" \
    "$sib/Scripts/python.exe" "$sib/bin/python" \
    python3 python; do
  case "$cand" in
    */*) [ -x "$cand" ] || continue ;;
    *)   command -v "$cand" >/dev/null 2>&1 || continue ;;
  esac
  if "$cand" -c "import sys; sys.path.insert(0, r'$REPO'); import agent.redact" >/dev/null 2>&1; then
    PYRED="$cand"; break
  fi
done

# --- 5c. build a throwaway STAGING COPY and redact IT (not <out>) ---
# The durable session reports in <out> are left byte-for-byte intact; only the copy
# that goes into the zip is redacted. Fails closed: no redactor, or a survivor => no zip.
redaction_failed=0
STAGE_PARENT="$(mktemp -d 2>/dev/null || { d="${TMPDIR:-/tmp}/nf-handoff.$$"; mkdir -p "$d"; printf '%s' "$d"; })"
STAGE="$STAGE_PARENT/$(basename "$OUT")"
mkdir -p "$STAGE"
( cd "$OUT" && find . -maxdepth 1 -mindepth 1 \
    ! -name '*.zip' ! -name '*.zip.*' ! -name 'HANDOFF-INDEX.md' ! -name 'redaction-report.txt' \
    -exec cp -R {} "$STAGE/" \; )
if [ -z "$PYRED" ]; then
  add_check FAIL "handoff redaction could not run - no Python that imports agent.redact. Run scripts/bootstrap-north-forge.ps1."
  redaction_failed=1
else
  set +e
  "$PYRED" "$REDACT_PY" "$STAGE"
  rc=$?
  set -e
  summary=""
  if [ -f "$STAGE/redaction-report.txt" ]; then
    cp -f "$STAGE/redaction-report.txt" "$OUT/redaction-report.txt"   # keep a record in <out>
    fr="$(grep -oE 'files redacted[[:space:]]*:[[:space:]]*[0-9]+' "$STAGE/redaction-report.txt" | grep -oE '[0-9]+$' || true)"
    sc="$(grep -oE 'text files scanned[[:space:]]*:[[:space:]]*[0-9]+' "$STAGE/redaction-report.txt" | grep -oE '[0-9]+$' || true)"
    [ -n "$fr" ] && [ -n "$sc" ] && summary="$fr of $sc file(s) redacted"
  fi
  if [ "$rc" -eq 0 ]; then
    add_check OK "handoff redaction: ${summary:-done}, 0 survivors (agent.redact; backstop only; <out> sources untouched)"
  elif [ "$rc" -eq 3 ]; then
    add_check FAIL "handoff redaction: likely secret(s) SURVIVED - see redaction-report.txt. Zip NOT created; fix the named file:line in <out>."
    redaction_failed=1
  else
    add_check FAIL "handoff redaction failed (exit $rc) - see stderr. Zip NOT created."
    redaction_failed=1
  fi
fi

# --- 6. HANDOFF-INDEX.md (into the staging copy) -----------
IDX="$STAGE/HANDOFF-INDEX.md"
{
  echo "# Handoff bundle - $TODAY"
  echo
  echo "Generated $STAMP_ISO by \`scripts/collect-logs.sh\` on $HOST."
  echo "This folder is zipped to \`$ZIP\` - that zip is the single file to hand a reviewer."
  echo
  echo "## Git coordinates"
  echo
  if [ "$GIT_OK" -eq 1 ]; then
    echo "- HEAD \`$HEAD_SHA\` on \`$BRANCH\`"
    [ -n "$ahead" ] && echo "- vs \`origin/main\`: $ahead ahead / $behind behind"
    echo "- uncommitted files: $dirty_n"
  else
    echo "- (git unavailable when this ran - see repo-state.txt)"
  fi
  echo
  echo "## Contents"
  echo
  ( cd "$STAGE" && find . -type f ! -name 'HANDOFF-INDEX.md' | sed 's#^\./##' | sort | while IFS= read -r r; do
      sz="$(wc -c <"$STAGE/$r" | tr -d ' ')"
      echo "- \`$r\`  ($sz bytes)"
    done )
  echo
  echo "## Redaction"
  echo
  echo "Every text file in this bundle was run through the agent's production credential"
  echo "redactor (\`agent.redact\`) on a throwaway copy before zipping - the source files in"
  echo "<out> are left intact. See \`redaction-report.txt\` for what was touched."
  echo "Pattern-matching is a backstop, not a guarantee - treat the bundle as sensitive."
  echo
  echo "## Completeness self-check"
  echo
  echo "_(zip-integrity checks run after this file is written - see the console output of the run.)_"
  echo
  for c in "${CHECKS[@]}"; do
    lvl="${c%%$'\t'*}"; txt="${c#*$'\t'}"
    printf -- "- [%-4s] %s\n" "$lvl" "$txt"
  done
  echo
  ok_n=0; warn_n=0; fail_n=0
  for c in "${CHECKS[@]}"; do case "${c%%$'\t'*}" in OK) ok_n=$((ok_n+1));; WARN) warn_n=$((warn_n+1));; FAIL) fail_n=$((fail_n+1));; esac; done
  if [ "$fail_n" -gt 0 ]; then echo "**INCOMPLETE** - $ok_n OK, $warn_n WARN, $fail_n FAIL"; else echo "**COMPLETE** - $ok_n OK, $warn_n WARN, $fail_n FAIL"; fi
} >"$IDX"

# --- 7. zip the STAGING COPY  (skipped if the redaction gate failed) ---
base="$(basename "$OUT")"
if [ "$redaction_failed" -eq 1 ]; then
  if [ -f "$ZIP" ]; then
    mv -f "$ZIP" "$ZIP.stale"
    rm -f "$ZIP.sha256"
    add_check FAIL "previous zip preserved as $(basename "$ZIP").stale - it is NOT the current handoff"
  fi
else
  rm -f "$ZIP" "$ZIP.stale"
  if command -v zip >/dev/null 2>&1; then
    ( cd "$STAGE_PARENT" && zip -rq "$ZIP" "$base" )
  elif command -v python3 >/dev/null 2>&1; then
    ( cd "$STAGE_PARENT" && python3 -m zipfile -c "$ZIP" "$base" )
  elif command -v python >/dev/null 2>&1; then
    ( cd "$STAGE_PARENT" && python -m zipfile -c "$ZIP" "$base" )
  else
    add_check FAIL "no zip/python available to build $ZIP"
  fi
fi

sha=""
if [ -f "$ZIP" ]; then
  if command -v sha256sum >/dev/null 2>&1; then sha="$(sha256sum "$ZIP" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then sha="$(shasum -a 256 "$ZIP" | awk '{print $1}')"; fi
  [ -n "$sha" ] && printf '%s  %s\n' "$sha" "$(basename "$ZIP")" >"$ZIP.sha256"
fi

# --- 8. post-zip verification ----------------------------
if [ -f "$ZIP" ]; then
  if command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
    PY="$(command -v python3 || command -v python)"
    zip_list="$("$PY" - "$ZIP" "$base" <<'PY'
import sys, zipfile
zp, base = sys.argv[1], sys.argv[2]
out = []
for n in zipfile.ZipFile(zp).namelist():
    if n.endswith('/'):
        continue
    n = n.replace('\\', '/')
    if n.startswith(base + '/'):
        n = n[len(base) + 1:]
    out.append(n)
sys.stdout.write('\n'.join(out))
PY
)"
  else
    zip_list="$(cd "$parent" && unzip -Z1 "$ZIP" | sed "s#^$base/##" | grep -v '/$' || true)"
  fi
  zip_sorted="$(printf '%s\n' "$zip_list" | tr -d '\r' | sed '/^$/d' | sort)"
  stage_sorted="$(cd "$STAGE" && find . -type f | sed 's#^\./##' | tr -d '\r' | sed '/^$/d' | sort)"
  only_stage="$(comm -23 <(printf '%s\n' "$stage_sorted") <(printf '%s\n' "$zip_sorted"))"
  only_zip="$(comm -13 <(printf '%s\n' "$stage_sorted") <(printf '%s\n' "$zip_sorted"))"
  if [ -z "$only_stage" ] && [ -z "$only_zip" ]; then
    add_check OK "$(basename "$ZIP") contains every staged file ($(printf '%s\n' "$zip_sorted" | wc -l | tr -d ' ') entries)"
  else
    [ -n "$only_stage" ] && add_check FAIL "not in zip: $(printf '%s' "$only_stage" | tr '\n' ' ')"
    [ -n "$only_zip" ]   && add_check FAIL "in zip but not staged: $(printf '%s' "$only_zip" | tr '\n' ' ')"
  fi
  bytes="$(wc -c <"$ZIP" | tr -d ' ')"
  add_check OK "$(basename "$ZIP") = $bytes bytes${sha:+, SHA256 ${sha:0:16}...}"
fi

# --- cleanup the throwaway staging copy -----------------
[ -n "${STAGE_PARENT:-}" ] && rm -rf "$STAGE_PARENT"

# --- 9. report -------------------------------------------
ok_n=0; warn_n=0; fail_n=0
for c in "${CHECKS[@]}"; do case "${c%%$'\t'*}" in OK) ok_n=$((ok_n+1));; WARN) warn_n=$((warn_n+1));; FAIL) fail_n=$((fail_n+1));; esac; done
if [ "$QUIET" -eq 0 ]; then
  printf '\ncollect-logs - %s\n' "$STAMP_ISO"
  printf '  repo : %s\n  out  : %s\n  zip  : %s\n\n' "$REPO" "$OUT" "$ZIP"
  for c in "${CHECKS[@]}"; do printf '  [%-4s] %s\n' "${c%%$'\t'*}" "${c#*$'\t'}"; done
  echo
  if [ "$fail_n" -gt 0 ]; then printf '  INCOMPLETE - %s FAIL, %s WARN\n\n' "$fail_n" "$warn_n"
  else printf '  COMPLETE - 0 FAIL, %s WARN\n\n' "$warn_n"; fi
fi
[ "$fail_n" -eq 0 ]
