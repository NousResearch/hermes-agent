#!/usr/bin/env bash
# delta_coverage_report.sh v2 — file-scoped line-level coverage.
# For each (file + added-line) in git diff v0.16.0..HEAD -- *.py, attribute to:
#   - the UNION of open-PR-head diffs (same file, same added line), OR
#   - the Bucket C residual patch (same file, same added line).
# Report per-file totals + grand reconciliation + any unattributed (file,line) pairs.
set -u
SRC="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
V016=3c231eb3979ab9c57d5cd6d02f1d577a3b718b43
GH="env -u GITHUB_TOKEN -u GH_TOKEN gh"
PATCH="${PATCH:-./RESIDUAL-NOT-IN-ANY-PR.patch}"
cd "$SRC" || exit 1
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

# helper: emit "FILE\tADDEDLINE" pairs from a unified diff on stdin
pairs() {
  awk '
    /^diff --git / { f=$0; sub(/^diff --git a\//,"",f); sub(/ b\/.*$/,"",f); next }
    /^\+\+\+ /     { ff=$0; sub(/^\+\+\+ b\//,"",ff); if(ff!="/dev/null") f=ff; next }
    /^\+[^+]/      { l=substr($0,2); gsub(/^[ \t]+|[ \t]+$/,"",l); if(l!="") print f"\t"l }
  '
}

# 1. full delta added (file,line) pairs
git diff "$V016" HEAD -- '*.py' | pairs | sort -u > "$WORK/delta.tsv"
TOTAL=$(wc -l < "$WORK/delta.tsv")

# 2. union of open-PR heads
OPEN_PRS=$($GH pr list --repo NousResearch/hermes-agent --author arminanton --state open --limit 100 \
  --json number,headRefName -q '.[] | .headRefName' 2>/dev/null)
> "$WORK/pr.tsv"
for br in $OPEN_PRS; do
  git fetch fork "$br" >/dev/null 2>&1
  git diff "$V016"...fork/"$br" -- '*.py' 2>/dev/null | pairs >> "$WORK/pr.tsv"
done
sort -u "$WORK/pr.tsv" -o "$WORK/pr.tsv"

# 3. Bucket C patch pairs
[ -f "$PATCH" ] && pairs < "$PATCH" | sort -u > "$WORK/bucketc.tsv" || : > "$WORK/bucketc.tsv"

# 4. coverage
cat "$WORK/pr.tsv" "$WORK/bucketc.tsv" | sort -u > "$WORK/covered.tsv"
comm -23 "$WORK/delta.tsv" "$WORK/covered.tsv" > "$WORK/unattr.tsv"
UNATTR=$(wc -l < "$WORK/unattr.tsv")
INPR=$(comm -12 "$WORK/delta.tsv" "$WORK/pr.tsv" | wc -l)
INBC=$(comm -23 <(comm -12 "$WORK/delta.tsv" "$WORK/bucketc.tsv") "$WORK/pr.tsv" | wc -l)

echo "=== FILE-SCOPED LINE COVERAGE (git diff v0.16.0..HEAD -- *.py) ==="
echo "total delta (file,line) pairs : $TOTAL"
echo "  covered by an OPEN PR        : $INPR"
echo "  covered ONLY by Bucket C     : $INBC"
echo "  UNATTRIBUTED                 : $UNATTR"
echo ""
if [ "$UNATTR" -gt 0 ]; then
  echo "=== UNATTRIBUTED pairs by file (count) ==="
  cut -f1 "$WORK/unattr.tsv" | sort | uniq -c | sort -rn
  echo ""
  echo "=== first 30 unattributed (file<TAB>line) ==="
  head -30 "$WORK/unattr.tsv"
  cp "$WORK/unattr.tsv" /tmp/delta_unattr.tsv
fi
