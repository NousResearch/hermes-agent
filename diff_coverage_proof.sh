#!/usr/bin/env bash
# Reproducible diff-coverage proof against v0.16.0 (3c231eb).
#   delta  = `git diff 3c231eb..HEAD --name-only`
#   union  = files covered by the 40 code PRs (#50111 manifest excluded), each PR's
#            diff measured vs its merge-base with v0.17.0 (the PR target)
#   PROVE: delta - union  ==  exactly the documented non-contributable buckets, 0 orphans:
#            .bak backups + .project-intel generated + transcripts eval-captures + subdirectory_hints(upstream #29433)
set -u
BASE=3c231eb3979ab9c57d5cd6d02f1d577a3b718b43      # v0.16.0
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3      # v0.17.0 (PR target)
FORK=https://github.com/arminanton/hermes-agent.git
TMP=$(mktemp -d)

git diff --name-only "$BASE"..HEAD | sort -u > "$TMP/delta.txt"
ndelta=$(wc -l < "$TMP/delta.txt")

: > "$TMP/union.txt"
env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent \
  --author arminanton --state open --limit 100 --json number,headRefOid \
  -q '.[] | "\(.number) \(.headRefOid)"' | sort -n > "$TMP/prs.txt"
while read -r num sha; do
  [ "$num" = "50111" ] && continue
  git fetch -q --force "$FORK" "$sha" 2>/dev/null
  mb=$(git merge-base "$V017" "$sha" 2>/dev/null); [ -z "$mb" ] && mb=$V017
  git diff --name-only "$mb..$sha" | sed -E 's#^v0\.17\.0-ready/##' >> "$TMP/union.txt"
done < "$TMP/prs.txt"
sort -u "$TMP/union.txt" -o "$TMP/union.txt"

comm -23 "$TMP/delta.txt" "$TMP/union.txt" > "$TMP/uncovered.txt"
nuncov=$(wc -l < "$TMP/uncovered.txt")

bak=$(grep -cE '\.bak' "$TMP/uncovered.txt")
intel=$(grep -cE '^\.project-intel/' "$TMP/uncovered.txt")
trans=$(grep -cE '^transcripts/' "$TMP/uncovered.txt")
upstream=$(grep -cE 'subdirectory_hints' "$TMP/uncovered.txt")
orphan=$(grep -vE '\.bak|^\.project-intel/|^transcripts/|subdirectory_hints' "$TMP/uncovered.txt" | wc -l)
explained=$((bak+intel+trans+upstream))

echo "=== REPRODUCIBLE DIFF-COVERAGE PROOF (base $BASE) ==="
echo "delta (3c231eb..HEAD):              $ndelta files"
echo "delta - union of 40 code PRs:       $nuncov files uncovered"
echo "  .bak backups (DISCARD):           $bak"
echo "  .project-intel generated (DISCARD): $intel"
echo "  transcripts eval-captures (DISCARD): $trans"
echo "  subdirectory_hints (upstream #29433): $upstream"
echo "  UNEXPLAINED ORPHANS:              $orphan"
echo "  in-PR (delta covered):            $((ndelta - nuncov))"
echo
if [ "$orphan" -eq 0 ] && [ "$explained" -eq "$nuncov" ]; then
  echo "PROVEN: delta($ndelta) = in-PR($((ndelta-nuncov))) + bak($bak) + project-intel($intel) + transcripts($trans) + upstream-29433($upstream), 0 orphans."
else
  echo "FAIL: $orphan unexplained orphans:"
  grep -vE '\.bak|^\.project-intel/|^transcripts/|subdirectory_hints' "$TMP/uncovered.txt"
fi
echo
echo "--- full uncovered (DISCARD+upstream) list, audit ---"
cat "$TMP/uncovered.txt"
rm -rf "$TMP"
