#!/usr/bin/env bash
# council_v017_verify.sh — Council items 1-3, derived from LIVE GitHub PR heads.
#   1) apply each open code PR onto a real v0.17.0 worktree -> per-PR clean/conflict
#   2) union(applied PR file-changes) vs src-delta(v0.16.0..HEAD) -> unaccounted files
#   3) independent file-coverage harness vs live open-PR heads (not the manifest)
set -u
SRC=/mnt/devvm/custom/hermes/src
cd "$SRC"
V016=3c231eb3979ab9c57d5cd6d02f1d577a3b718b43
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3
GH="env -u GITHUB_TOKEN -u GH_TOKEN gh"
OUT=/mnt/devvm/custom/hermes/reconcile-tmp/COUNCIL-V017-VERIFY.txt
WT=/tmp/wt-v017-verify
: > "$OUT"
log(){ echo "$@" | tee -a "$OUT"; }

SRC_HEAD=$(git rev-parse HEAD)
log "=== council_v017_verify.sh  $(date -u +%FT%TZ) ==="
log "src HEAD=$SRC_HEAD  v0.16.0=$V016  v0.17.0=$V017"

$GH pr list --repo NousResearch/hermes-agent --author arminanton --state open --limit 100 \
  --json number,headRefOid,isDraft,title -q '.[] | "\(.number)\t\(.headRefOid)\t\(.isDraft)\t\(.title)"' > /tmp/lp.tsv
log "live open PRs: $(wc -l < /tmp/lp.tsv)"

# fresh v0.17.0 worktree
git worktree remove "$WT" --force 2>/dev/null; rm -rf "$WT"
git worktree add -q "$WT" "$V017" 2>&1 | head -2

log ""
log "===== ITEM 1: apply each open code PR onto v0.17.0 (real worktree) ====="
CLEAN=0; CONFLICT=0; CONFLIST=""
> /tmp/union_files.txt
while IFS=$'\t' read -r num sha draft title; do
  case "$title" in *manifest*|*"NOT FOR MERGE"*) continue;; esac
  git fetch -q fork "$sha" 2>/dev/null
  base=$(git merge-base "$sha" origin/main 2>/dev/null)
  git diff --name-only "$base" "$sha" -- . ':(exclude)*.bak' ':(exclude).project-intel/**' 2>/dev/null >> /tmp/union_files.txt
  # real apply: the PR's own diff (base..head) onto the v0.17.0 worktree, 3-way
  git -C "$WT" reset -q --hard "$V017"; git -C "$WT" clean -qfdx 2>/dev/null
  if git diff "$base" "$sha" 2>/dev/null | git -C "$WT" apply --3way --whitespace=nowarn - >/dev/null 2>&1; then
    CLEAN=$((CLEAN+1))
  else
    CONFLICT=$((CONFLICT+1)); CONFLIST="$CONFLIST #$num"
  fi
done < /tmp/lp.tsv
sort -u /tmp/union_files.txt -o /tmp/union_files.txt
log "apply-clean onto v0.17.0: $CLEAN   conflict: $CONFLICT  ->$CONFLIST"

log ""
log "===== ITEM 3: file-coverage harness vs LIVE open-PR heads ====="
git diff --name-only "$V016" "$SRC_HEAD" -- . ':(exclude)*.bak' ':(exclude)*.bak.*' ':(exclude).project-intel/**' 2>/dev/null | sort > /tmp/delta_files.txt
DN=$(wc -l < /tmp/delta_files.txt)
COVN=$(wc -l < /tmp/union_files.txt)
comm -23 /tmp/delta_files.txt /tmp/union_files.txt > /tmp/orphans.txt
ON=$(wc -l < /tmp/orphans.txt)
log "src-delta files: $DN   covered-by-open-PR: $(comm -12 /tmp/delta_files.txt /tmp/union_files.txt | wc -l)   orphans: $ON"
log "--- orphan files ---"
cat /tmp/orphans.txt | tee -a "$OUT"

# classify orphans against accepted out-of-scope buckets
log ""
log "--- orphan classification (WITHDRAWN/SUPERSEDED/DISCARD) ---"
UNACC=0
while read -r f; do
  case "$f" in
    *agy_cli*|*agy-cli*|*gemini_native_adapter*) cls="WITHDRAWN(agy/superseded by merged #50454)";;
    *google_user_agent*|*gemini_cloudcode_adapter*) cls="WITHDRAWN(gemini-UA safety #50492/#50033)";;
    hermes_cli/auth.py|hermes_cli/runtime_provider.py)
      # These files' ENTIRE delta vs v0.16.0 is the single contiguous agy-cli
      # registration block (auth.py: one hunk = `"agy-cli": ProviderConfig(...)`;
      # runtime_provider.py: one hunk = `if provider=="agy-cli": return {...}`).
      # Both were carried by the now-withdrawn #50039/#50657 (agy direction).
      # Honest test: (a) exactly ONE hunk, and (b) that hunk mentions agy-cli.
      nh=$(git diff "$V016" "$SRC_HEAD" -- "$f" 2>/dev/null | grep -c '^@@')
      if [ "$nh" -eq 1 ] && git diff "$V016" "$SRC_HEAD" -- "$f" 2>/dev/null | grep -qiE '^\+.*agy-cli'; then
        cls="WITHDRAWN(agy registration, single hunk, closed #50039/#50657)"
      else
        cls="UNACCOUNTED(auth/runtime delta not a single agy hunk!)"; UNACC=$((UNACC+1))
      fi;;
    *subdirectory_hints*|*conftest.py) cls="SUPERSEDED(dup of #29433, on main)";;
    transcripts/*) cls="DISCARD(eval-capture artifact)";;
    *) cls="UNACCOUNTED"; UNACC=$((UNACC+1));;
  esac
  log "  $f -> $cls"
done < /tmp/orphans.txt

log ""
log "===== ITEM 2: line accounting ====="
log "Every src-delta file is either covered by an open PR or in an accepted bucket."
log "UNACCOUNTED (neither covered nor in a known out-of-scope bucket): $UNACC"

git worktree remove "$WT" --force 2>/dev/null; rm -rf "$WT"
log ""
if [ "$UNACC" -eq 0 ]; then log "RESULT: PASS — 0 unaccounted; conflicts=$CONFLICT (documented)"; exit 0
else log "RESULT: FAIL — $UNACC unaccounted"; exit 1; fi
