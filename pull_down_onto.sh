#!/usr/bin/env bash
# pull_down_onto.sh — SELF-CONTAINED downstream-consumer script.
# Reproduces "apply the full open-PR set onto a target release" with ZERO reliance on
# comment threads. A consumer runs:  bash pull_down_onto.sh <TARGET_REF> [<repo_checkout>]
#
#   <TARGET_REF>      a release commit/tag/branch to pull the PRs onto
#                     (default = v0.17.0 commit 2bd1977d8fad185c9b4be47884f7e87f1add0ce3)
#   <repo_checkout>   a checkout of NousResearch/hermes-agent with a `fork` remote
#                     pointing at arminanton/hermes-agent (default = $PWD)
#
# It fetches the open arminanton PRs + the #50111 resolution patches itself, checks out
# a detached worktree at TARGET_REF (asserting HEAD==TARGET), applies every PR's net diff
# sequentially, auto-applies the committed resolution patches for drifted files, then
# compiles the integrated tree. Exit 0 iff every PR lands with 0 unresolved conflicts.
set -u
TARGET="${1:-2bd1977d8fad185c9b4be47884f7e87f1add0ce3}"
REPO="${2:-$PWD}"
cd "$REPO" || { echo "repo checkout not found: $REPO"; exit 2; }
GH="env -u GITHUB_TOKEN -u GH_TOKEN gh"
PY="$(command -v python3)"
WT="$(mktemp -d /tmp/pulldown.XXXX)"

echo "=== pull_down_onto.sh  $(date -u +%FT%TZ) ==="
TREF=$(git rev-parse "$TARGET^{commit}" 2>/dev/null) || { echo "cannot resolve TARGET_REF=$TARGET (fetch it first)"; exit 2; }
echo "TARGET_REF=$TARGET -> $TREF"
echo "  (= $(git log -1 --format='%h %s' "$TREF" 2>/dev/null))"

# the resolution-patch source: #50111 manifest branch head on the fork
SHA111=$($GH pr view 50111 --repo NousResearch/hermes-agent --json headRefOid -q .headRefOid 2>/dev/null)
git fetch -q fork "$SHA111" 2>/dev/null
echo "resolution patches from #50111 @ ${SHA111:0:9}"

# live open PRs (number, head, title). The 40th open PR is the #50111 MANIFEST itself
# (NOT FOR MERGE) — excluded from apply; the 39 code PRs are applied.
$GH pr list --repo NousResearch/hermes-agent --author arminanton --state open --limit 100 \
  --json number,headRefOid,title -q '.[] | "\(.number)\t\(.headRefOid)\t\(.title)"' > "$WT/prs.tsv"
TOTAL_OPEN=$(wc -l < "$WT/prs.tsv")
MANIFEST=$(grep -cE 'manifest|NOT FOR MERGE' "$WT/prs.tsv")
CODE=$((TOTAL_OPEN-MANIFEST))
echo "open PRs: $TOTAL_OPEN  =  $CODE code PRs  +  $MANIFEST manifest/#50111 (excluded from apply)"

# detached worktree pinned at TARGET; assert
git worktree add -q --detach "$WT/tree" "$TREF" 2>&1 | head -1
H=$(git -C "$WT/tree" rev-parse HEAD)
[ "$H" = "$TREF" ] && echo "✓ ASSERT worktree HEAD == TARGET_REF" || { echo "✗ HEAD!=TARGET"; exit 2; }

resolve(){ local f="$1" flat patch; flat=$(echo "$f"|tr '/' '_')
  patch=$(git show "$SHA111:v017-conflict-resolutions/${flat}.v017.patch" 2>/dev/null)
  [ -n "$patch" ] || return 1
  git -C "$WT/tree" checkout "$TREF" -- "$f" 2>/dev/null
  echo "$patch" | git -C "$WT/tree" apply --whitespace=nowarn - 2>/dev/null; }

CLEAN=0; RES=0; FAIL=0; FAILLIST=""
while IFS=$'\t' read -r num sha title; do
  case "$title" in *manifest*|*"NOT FOR MERGE"*) continue;; esac
  git fetch -q fork "$sha" 2>/dev/null
  base=$(git merge-base "$sha" origin/main 2>/dev/null)
  git diff "$base" "$sha" -- . ':(exclude)*.bak' ':(exclude).project-intel/**' 2>/dev/null > "$WT/pr.diff"
  if git -C "$WT/tree" apply --3way --whitespace=nowarn "$WT/pr.diff" 2>/dev/null; then CLEAN=$((CLEAN+1))
  else
    cf=$(grep -rl '^<<<<<<< ' "$WT/tree" --include='*.py' 2>/dev/null | sed "s|$WT/tree/||" | sort -u)
    ok=1; for f in $cf; do resolve "$f" || ok=0; done
    rem=$(grep -rl '^<<<<<<< ' "$WT/tree" --include='*.py' 2>/dev/null | wc -l)
    if [ "$ok" -eq 1 ] && [ "$rem" -eq 0 ]; then RES=$((RES+1))
    else FAIL=$((FAIL+1)); FAILLIST="$FAILLIST #$num($cf)"; for f in $cf; do git -C "$WT/tree" checkout "$TREF" -- "$f" 2>/dev/null; done; fi
  fi
done < "$WT/prs.tsv"
APPLIED=$((CLEAN+RES))
echo "apply onto $TARGET: clean=$CLEAN resolved=$RES failed=$FAIL (applied=$APPLIED/$CODE) ->$FAILLIST"

# build: compile changed .py
CF=0
for f in $(git -C "$WT/tree" diff --name-only "$TREF" 2>/dev/null | grep -E '\.py$'); do
  [ -f "$WT/tree/$f" ] && { "$PY" -m py_compile "$WT/tree/$f" 2>/dev/null || { echo "  compile-fail: $f"; CF=$((CF+1)); }; }
done
echo "compile-fail=$CF"

git worktree remove "$WT/tree" --force 2>/dev/null; rm -rf "$WT"
echo ""
if [ "$FAIL" -eq 0 ] && [ "$CF" -eq 0 ]; then
  echo "RESULT: PASS — $APPLIED/$CODE code PRs pull down onto $TARGET ($CLEAN clean + $RES resolved), tree compiles"
  exit 0
else
  echo "RESULT: FAIL — failed-apply=$FAIL compile-fail=$CF"
  exit 1
fi
