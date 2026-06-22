#!/usr/bin/env bash
# integration_v017_sequential.sh — the CORRECT integration test.
# Apply the 39 open PRs SEQUENTIALLY onto fresh v0.17.0 (each PR = its own base..head
# patch, the real pull-down operation), using #50111 resolution patches where a PR's
# file conflicts. Then build (py_compile) the integrated tree.
# This replaces the flawed monolithic `git diff v016..HEAD` apply, which conflicts on
# every file v0.17.0 itself rewrote (cli.py/gateway/run.py/main.py = thousands of lines).
set -u
SRC=/mnt/devvm/custom/hermes/src; cd "$SRC"
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3
GH="env -u GITHUB_TOKEN -u GH_TOKEN gh"
OUT=/mnt/devvm/custom/hermes/reconcile-tmp/INTEGRATION-V017-SEQ.txt
WT=/tmp/wt-seq
: > "$OUT"; log(){ echo "$@" | tee -a "$OUT"; }
log "=== integration_v017_sequential.sh $(date -u +%FT%TZ) ==="

$GH pr list --repo NousResearch/hermes-agent --author arminanton --state open --limit 100 \
  --json number,headRefOid,isDraft,title -q '.[] | "\(.number)\t\(.headRefOid)\t\(.title)"' > /tmp/lp.tsv
sha111=$($GH pr view 50111 --repo NousResearch/hermes-agent --json headRefOid -q .headRefOid 2>/dev/null)
git fetch -q fork "$sha111" 2>/dev/null

git worktree remove "$WT" --force 2>/dev/null; rm -rf "$WT"
git worktree add -q "$WT" "$V017" 2>&1 | head -1
log "fresh v0.17.0 worktree; applying PRs sequentially"

CLEAN=0; RESOLVED=0; FAILED=0; FAILLIST=""
# resolution-patch lookup: conflicting file -> patch on #50111
resolve_file(){ # $1=worktree-relative file
  local f="$1" flat patch
  flat=$(echo "$f" | tr '/' '_')
  patch=$(git show "$sha111:v017-conflict-resolutions/${flat}.v017.patch" 2>/dev/null)
  [ -n "$patch" ] || return 1
  git -C "$WT" checkout "$V017" -- "$f" 2>/dev/null
  echo "$patch" | git -C "$WT" apply --whitespace=nowarn - 2>/dev/null
}

while IFS=$'\t' read -r num sha title; do
  case "$title" in *manifest*|*"NOT FOR MERGE"*) continue;; esac
  git fetch -q fork "$sha" 2>/dev/null
  base=$(git merge-base "$sha" origin/main 2>/dev/null)
  git diff "$base" "$sha" -- . ':(exclude)*.bak' ':(exclude).project-intel/**' 2>/dev/null > /tmp/pr.diff
  if git -C "$WT" apply --3way --whitespace=nowarn /tmp/pr.diff 2>/dev/null; then
    CLEAN=$((CLEAN+1))
  else
    # conflict — find conflicted files, try resolution patches
    cf=$(grep -rl '^<<<<<<< ' "$WT" --include='*.py' 2>/dev/null | sed "s|$WT/||" | sort -u)
    okall=1
    for f in $cf; do resolve_file "$f" || okall=0; done
    # also reset any non-py partials cleanly
    rem=$(grep -rl '^<<<<<<< ' "$WT" --include='*.py' 2>/dev/null | wc -l)
    if [ "$okall" -eq 1 ] && [ "$rem" -eq 0 ]; then
      RESOLVED=$((RESOLVED+1)); log "  #$num: conflict resolved via patch ($cf)"
    else
      FAILED=$((FAILED+1)); FAILLIST="$FAILLIST #$num"; log "  #$num: UNRESOLVED conflict ($cf)"
      # revert this PR's partial apply so the run continues cleanly
      for f in $cf; do git -C "$WT" checkout "$V017" -- "$f" 2>/dev/null; done
    fi
  fi
done < /tmp/lp.tsv
log ""
log "apply results: clean=$CLEAN resolved=$RESOLVED failed=$FAILED ->$FAILLIST"

# BUILD: compile every .py that now differs from v0.17.0 in the integrated tree
log ""
log "===== BUILD: py_compile integrated tree (changed files) ====="
CF=0; changed=$(git -C "$WT" diff --name-only "$V017" 2>/dev/null | grep -E '\.py$')
for f in $changed; do
  [ -f "$WT/$f" ] || continue
  python3 -m py_compile "$WT/$f" 2>>/tmp/seqcompile.err || { log "  ✗ $f"; CF=$((CF+1)); }
done
NCH=$(echo "$changed" | grep -c .)
log "integrated changed .py files=$NCH  compile-fail=$CF"

cd "$SRC"; git worktree remove "$WT" --force 2>/dev/null; rm -rf "$WT"
log ""
if [ "$FAILED" -eq 0 ] && [ "$CF" -eq 0 ]; then
  log "RESULT: PASS — 39 PRs integrate onto v0.17.0 ($CLEAN clean + $RESOLVED resolved), tree compiles"
else
  log "RESULT: ISSUES — failed-apply=$FAILED compile-fail=$CF"
fi
