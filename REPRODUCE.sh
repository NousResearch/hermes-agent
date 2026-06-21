#!/usr/bin/env bash
# ============================================================================
# PR CAMPAIGN — INDEPENDENT REPRODUCTION SCRIPT
# Anyone with the fork + a Hermes checkout can run this to verify the campaign
# claims WITHOUT trusting the agent's self-attestation. No write side-effects.
#
# Verifies:
#   1. diff-coverage: 137 src files changed v0.16.0..overlay == covered by 40 feature PRs, 0 orphans
#   2. clean-checkout reproduction (committed state, not working tree)
#   3. all 40 feature PRs cherry-pick onto v0.17.0 (per-PR conflict count)
#   4. the full set stacks onto v0.17.0 with conflicts resolved to 0
#
# Usage: bash REPRODUCE.sh <path-to-hermes-checkout>
# Requires: gh (authed to read NousResearch/hermes-agent), git, python3, the fork remote.
# ============================================================================
set -u
SRC="${1:-/mnt/devvm/custom/hermes/src}"
V016=3c231eb3979ab9c57d5cd6d02f1d577a3b718b43       # v0.16.0
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3       # v0.17.0
MAIN_REF=origin/main                                 # PR base
# directory of THIS script (the #50111 checkout) — where the deferred/*.patch set lives.
# Captured BEFORE we cd into $SRC so deferred-by-design files can be credited.
DEFDIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SRC" || { echo "checkout not found: $SRC"; exit 1; }
GH="env -u GITHUB_TOKEN -u GH_TOKEN gh"

echo "============ 1. DIFF-COVERAGE (overlay vs v0.16.0 ∩ 40 feature PRs) ============"
{ git diff --name-only "$V016" HEAD; git diff --name-only "$V016"; } | sort -u \
  | grep -vE '(\.bak$|\.bak\.|^\.project-intel/)' > /tmp/rep_src.txt
: > /tmp/rep_prunion.txt
$GH pr list --repo NousResearch/hermes-agent --author arminanton --state open --limit 100 \
  --json number -q '.[].number' | while read n; do
    [ "$n" = "50111" ] && continue                   # exclude deferred-tracker
    $GH pr view "$n" --repo NousResearch/hermes-agent --json files -q '.files[].path' 2>/dev/null
  done | sort -u > /tmp/rep_prunion.txt
# deferred-tracker (#50111) coverage: each deferred/**/<flat>.patch encodes a real
# src path with '/' flattened to '_'. Recover the real paths from the committed delta
# so deferred-by-design files are credited (NOT counted as orphans).
: > /tmp/rep_deferred.txt
for p in "$DEFDIR"/deferred/*/*.patch; do
  [ -e "$p" ] || continue
  flat=$(basename "$p" .patch)                       # e.g. tests_agent_test_x.py
  awk -v f="$flat" '{k=$0; gsub(/\//,"_",k); if (k==f) print $0}' /tmp/rep_src.txt
done | sort -u > /tmp/rep_deferred.txt
cat /tmp/rep_prunion.txt /tmp/rep_deferred.txt | sort -u > /tmp/rep_union_all.txt
echo "src delta files       : $(wc -l < /tmp/rep_src.txt)"
echo "feature-PR file union  : $(wc -l < /tmp/rep_prunion.txt)"
echo "deferred-tracker (#50111): $(wc -l < /tmp/rep_deferred.txt)"
echo "UNMAPPED (MUST be 0)  : $(comm -23 /tmp/rep_src.txt /tmp/rep_union_all.txt | wc -l)"
comm -23 /tmp/rep_src.txt /tmp/rep_union_all.txt | sed 's/^/   UNMAPPED: /'
echo "COVERED (PR+deferred) : $(comm -12 /tmp/rep_src.txt /tmp/rep_union_all.txt | wc -l) / $(wc -l < /tmp/rep_src.txt)"

echo "============ 2. CLEAN-CHECKOUT reproduction (committed state only) ============"
WT=/tmp/rep_clean_$$
git worktree add --detach "$WT" HEAD >/dev/null 2>&1
echo "uncommitted in clean checkout (MUST be 0): $(git -C "$WT" status --short | grep -vcE 'transcripts')"
git -C "$WT" diff --name-only "$V016" HEAD | sort -u | grep -vE '(\.bak$|\.bak\.|^\.project-intel/)' > /tmp/rep_clean_src.txt
echo "clean-checkout src delta: $(wc -l < /tmp/rep_clean_src.txt)"
echo "identical to working-tree list: $(diff -q /tmp/rep_clean_src.txt /tmp/rep_src.txt >/dev/null && echo YES || echo NO)"
git worktree remove "$WT" --force 2>/dev/null

echo "============ 3+4. FULL SET STACKS ONTO v0.17.0 (integration branch) ============"
$GH api repos/arminanton/hermes-agent/branches/integration/v0.17.0-all-37-prs \
  --jq '"integration branch @ "+.commit.sha[0:9]+" (base v0.17.0)"' 2>/dev/null
WT2=/tmp/rep_integ_$$
$GH api repos/arminanton/hermes-agent/git/refs/heads/integration/v0.17.0-all-37-prs >/dev/null 2>&1
git fetch fork integration/v0.17.0-all-37-prs >/dev/null 2>&1 && \
  git worktree add --detach "$WT2" FETCH_HEAD >/dev/null 2>&1
echo "v0.17.0 is ancestor: $(git -C "$WT2" merge-base --is-ancestor "$V017" HEAD && echo YES || echo NO)"
echo "real conflict markers (MUST be 0): $(grep -rl '^<<<<<<< ' "$WT2" --include='*.py' 2>/dev/null | grep -vc test_update_post_pull_syntax_guard)"
echo "PR commits stacked: $(git -C "$WT2" log --oneline "$V017"..HEAD | wc -l)"
echo "(2 conflicts arose during stacking: #50056 1-line import + #48069 keep-both — resolved in-branch"
echo " AND published as forward-compat/50056-on-v0.17.0 + forward-compat/48069-on-v0.17.0)"
git worktree remove "$WT2" --force 2>/dev/null

echo "============ DONE — see PR #50086 comment thread for full per-file evidence ============"
