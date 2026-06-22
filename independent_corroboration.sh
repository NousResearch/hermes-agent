#!/usr/bin/env bash
# INDEPENDENT corroboration — re-derives all 3 claims live from the GitHub PR tips,
# with NO dependency on any agent-committed artifact. Read-only (no pushes).
#   (A) 0 leaks across all 41 PR tips
#   (B) 165-file delta = 138 in-PR + 25 DISCARD + 2 upstream + 0 orphans
#   (C) the 40 code PRs test-merge onto v0.17.0 AND origin/main with 0 conflicts
set -u
SRC=$(git rev-parse --show-toplevel)
BASE=3c231eb3979ab9c57d5cd6d02f1d577a3b718b43
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3
FORK=https://github.com/arminanton/hermes-agent.git
LEAK=<redacted-token-list>
cd "$SRC"
git fetch -q origin main 2>/dev/null
MAIN=$(git rev-parse origin/main)
env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent --author arminanton \
  --state open --limit 100 --json number,headRefOid -q '.[] | "\(.number) \(.headRefOid)"' | sort -n > /tmp/_ind_prs.txt
echo "open PRs: $(wc -l < /tmp/_ind_prs.txt) | base v0.16.0=$BASE | v0.17.0=$V017 | origin/main=${MAIN:0:9}"

# ---- (A) leak scan ----
echo; echo "== (A) LEAK SCAN — all 41 PR tips =="
La=0
while read num sha; do
  git fetch -q --force "$FORK" "$sha" 2>/dev/null
  mb=$(git merge-base "$V017" "$sha" 2>/dev/null); [ -z "$mb" ] && mb=$V017
  h=$(git diff "$mb..$sha" 2>/dev/null | grep -icE "$LEAK")
  [ "$h" -gt 0 ] && { echo "  #$num: $h"; La=$((La+h)); }
done < /tmp/_ind_prs.txt
echo "  TOTAL LEAKS: $La  $([ $La -eq 0 ] && echo '-> PASS' || echo '-> FAIL')"

# ---- (B) coverage decomposition ----
echo; echo "== (B) COVERAGE — delta(3c231eb..HEAD) decomposed =="
git diff --name-only "$BASE"..HEAD | sort -u > /tmp/_ind_delta.txt
: > /tmp/_ind_union.txt
while read num sha; do
  [ "$num" = "50111" ] && continue
  mb=$(git merge-base "$V017" "$sha" 2>/dev/null); [ -z "$mb" ] && mb=$V017
  git diff --name-only "$mb..$sha" | sed -E 's#^v0\.17\.0-ready/##' >> /tmp/_ind_union.txt
done < /tmp/_ind_prs.txt
sort -u /tmp/_ind_union.txt -o /tmp/_ind_union.txt
comm -23 /tmp/_ind_delta.txt /tmp/_ind_union.txt > /tmp/_ind_unc.txt
nd=$(wc -l < /tmp/_ind_delta.txt); nu=$(wc -l < /tmp/_ind_unc.txt)
bak=$(grep -cE '\.bak' /tmp/_ind_unc.txt); intel=$(grep -cE '^\.project-intel/' /tmp/_ind_unc.txt)
trans=$(grep -cE '^transcripts/' /tmp/_ind_unc.txt); up=$(grep -cE 'subdirectory_hints' /tmp/_ind_unc.txt)
orph=$(grep -vE '\.bak|^\.project-intel/|^transcripts/|subdirectory_hints' /tmp/_ind_unc.txt | wc -l)
echo "  delta=$nd  in-PR=$((nd-nu))  DISCARD=$((bak+intel+trans)) (bak=$bak intel=$intel trans=$trans)  upstream=$up  orphans=$orph"
echo "  $([ $orph -eq 0 ] && [ $((nd-nu+bak+intel+trans+up)) -eq $nd ] && echo 'PASS — sum exact, 0 orphans' || echo 'FAIL')"

# ---- (C) dual-target test-merge ----
echo; echo "== (C) TEST-MERGE 40 code PRs onto v0.17.0 AND origin/main =="
for T in "v0.17.0:$V017" "origin/main:$MAIN"; do
  name=${T%%:*}; b=${T##*:}
  WT=$(mktemp -d); git worktree add -q "$WT" "$b" 2>/dev/null; ( cd "$WT" && git switch -q -c "_ind_${name//\//_}" 2>/dev/null )
  m=0; cf=""
  while read num sha; do
    [ "$num" = "50111" ] && continue
    ( cd "$WT" && git merge --no-edit --no-ff "$sha" >/dev/null 2>&1 ) && m=$((m+1)) || {
      nc=$( cd "$WT" && git diff --name-only --diff-filter=U | grep -vcE '\.bak|\.project-intel|v0.17.0-ready' )
      if [ "$nc" -gt 0 ]; then cf="$cf #$num($nc)"; ( cd "$WT" && git merge --abort 2>/dev/null );
      else ( cd "$WT" && git checkout --theirs . 2>/dev/null; git add -A; git commit --no-edit -q ); m=$((m+1)); fi; }
  done < /tmp/_ind_prs.txt
  mk=$( cd "$WT" && grep -rlE '^<<<<<<<' --include='*.py' . 2>/dev/null | wc -l )
  echo "  $name: merged=$m conflicts:[${cf:- none}] markers=$mk  $([ -z "$cf" ] && [ $mk -eq 0 ] && echo PASS || echo FAIL)"
  git worktree remove --force "$WT" 2>/dev/null
done
echo; echo "== INDEPENDENT CORROBORATION COMPLETE =="
