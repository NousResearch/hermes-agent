#!/usr/bin/env bash
# Reproducible combine + per-PR verification against PINNED base v0.17.0.
# Self-contained: fresh clone, pins every PR to its CURRENT head SHA (incl the
# force-pushed #50457), runs (A) per-PR individual clean-apply and (B) ordered
# 3-way-merge combine. Emits a SHA-pinned manifest + logs. No reuse of prior state.
#   Run: bash combine_and_verify_v017.sh
set -u
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3
WORK=<WORKDIR>
FORK=https://github.com/arminanton/hermes-agent.git
UPSTREAM=https://github.com/NousResearch/hermes-agent.git
OUTDIR="$WORK/out"; mkdir -p "$OUTDIR"
SHAMAN="$OUTDIR/PINNED-SHAS.txt"; PERPR="$OUTDIR/PER-PR-CLEAN.txt"; COMBINE="$OUTDIR/COMBINE.txt"
: > "$SHAMAN"; : > "$PERPR"; : > "$COMBINE"

rm -rf "$WORK/repo"; mkdir -p "$WORK/repo"; cd "$WORK/repo" || exit 1
git init -q; git remote add origin "$UPSTREAM"; git remote add fork "$FORK"
git fetch -q --depth 200 origin "$V017" || { echo "FETCH v017 FAILED"; exit 1; }
git checkout -q "$V017"
echo "base v0.17.0 = $(git rev-parse --short HEAD)  pre-markers=$(grep -rlE '^<<<<<<<' --include='*.py' . 2>/dev/null|wc -l)"

# enumerate open PRs (exclude manifest #50111) -> pin each to CURRENT head SHA
env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent \
  --author arminanton --state open --limit 100 \
  --json number,headRefName,headRefOid,isDraft \
  -q '.[] | "\(.number) \(.headRefName) \(.headRefOid) \(.isDraft)"' \
  | sort -n > "$WORK/prs.txt"
echo "open PRs: $(wc -l < "$WORK/prs.txt")"

# fetch every pinned SHA
while read -r num br sha draft; do
  git fetch -q --force fork "$br" 2>/dev/null
  git fetch -q --force fork "$sha" 2>/dev/null || git fetch -q --depth 200 fork "$br" 2>/dev/null
  printf "#%-7s %-50s %s draft=%s\n" "$num" "$br" "$sha" "$draft" >> "$SHAMAN"
done < "$WORK/prs.txt"

# ---- (A) per-PR individual clean-apply on pristine v0.17.0, pinned to head SHA ----
echo "== A: per-PR individual apply (pinned SHAs) ==" 
cleanA=0; confA=0
while read -r num br sha draft; do
  [ "$num" = "50111" ] && { echo "#$num MANIFEST (docs)" >> "$PERPR"; }
  git reset -q --hard "$V017"; git clean -fdq
  mb=$(git merge-base "$V017" "$sha" 2>/dev/null)
  git diff "$mb..$sha" > "$WORK/d_$num.diff" 2>/dev/null
  out=$(git apply --3way "$WORK/d_$num.diff" 2>&1); rc=$?
  mk=$(grep -rlE '^<<<<<<<' --include='*.py' . 2>/dev/null|wc -l)
  cf=0; for f in $(git diff --name-status | awk '$1!="D"{print $2}'); do case "$f" in *.py) [ -f "$f" ] && { python3 -m py_compile "$f" 2>/dev/null || cf=$((cf+1)); };; esac; done
  if [ $rc -eq 0 ] && [ "$mk" -eq 0 ]; then printf "#%-7s CLEAN @%s compile_fail=%s\n" "$num" "${sha:0:9}" "$cf" >> "$PERPR"; cleanA=$((cleanA+1))
  else printf "#%-7s CONFLICT mk=%s @%s\n" "$num" "$mk" "${sha:0:9}" >> "$PERPR"; confA=$((confA+1)); fi
done < "$WORK/prs.txt"
git reset -q --hard "$V017"; git clean -fdq
echo "A-TOTALS: clean=$cleanA conflict=$confA" >> "$PERPR"

# ---- (B) ordered 3-way-merge combine (ascending PR#), pinned SHAs ----
echo "== B: ordered 3-way combine ==" 
git checkout -q -B _integration "$V017"
merged=0; conflictprs=""
while read -r num br sha draft; do
  case "$num" in 50111) continue;; esac
  git merge --no-edit --no-ff "$sha" > "$WORK/m_$num.out" 2>&1
  if [ $? -eq 0 ]; then merged=$((merged+1))
  else
    nconf=$(git diff --name-only --diff-filter=U | grep -vcE '\.bak|\.project-intel|v0.17.0-ready')
    if [ "$nconf" -gt 0 ]; then conflictprs="$conflictprs $num($nconf)"; git merge --abort 2>/dev/null
    else git checkout --theirs . 2>/dev/null; git add -A; git commit --no-edit -q; merged=$((merged+1)); fi
  fi
done < "$WORK/prs.txt"
mk=$(grep -rlE '^<<<<<<<' --include='*.py' . 2>/dev/null|wc -l)
cf=0; for f in $(git diff --name-status "$V017" | awk '$1!="D"{print $2}'); do case "$f" in *.py) [ -f "$f" ] && { python3 -m py_compile "$f" 2>/dev/null || cf=$((cf+1)); };; esac; done
{
echo "B-COMBINE: merged=$merged conflicted=$(echo $conflictprs|wc -w)"
echo "conflicting:$conflictprs"
echo "final conflict markers: $mk"
echo "real compile failures (skip deleted): $cf"
echo "files changed vs v0.17.0: $(git diff --name-only "$V017"|wc -l)"
echo "integration tree SHA: $(git rev-parse HEAD)"
} >> "$COMBINE"

echo "=== DONE ==="; echo "--- PINNED SHAS ---"; cat "$SHAMAN" | head -5; echo "..."
echo "--- A (per-PR) ---"; tail -2 "$PERPR"
echo "--- B (combine) ---"; cat "$COMBINE"
echo "logs in: $OUTDIR"
