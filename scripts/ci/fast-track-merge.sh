#!/usr/bin/env bash
# Fast-track merge for eligible tabjoy-fleet PRs.
#
# Merges open PRs without a manual maintainer turn when every gate in the
# merge SLA policy (docs/merge-sla-policy.md) passes. Runs from the
# fast-track-merge.yml workflow on a schedule; identical logic works from a
# local checkout with a write token for dry-run validation.
#
# Gates (all must pass; a failing gate skips the PR, never fails the run):
#   1. author is the fleet account (devbxylw) OR PR carries tabjoy-fleet label
#   2. not draft, no do-not-merge / blocked label
#   3. GitHub reports mergeStateStatus == CLEAN (all required checks pass,
#      no conflicts, no outstanding required reviews)
#   4. changed lines (additions + deletions) <= 400
#   5. no security-sensitive paths in the diff (policy section 5.6)
#   6. head freshness: <= 50 commits behind origin/main OR head pushed < 24h
#
# Merge strategy: squash merge (policy section 5.8). Every merge is logged to
# the workflow run and as a comment on the PR.
#
# Usage:
#   fast-track-merge.sh [--dry-run] [--pr 12345 ...] [--owner org] [--repo name]
#   GH_TOKEN must be set (workflow GITHUB_TOKEN or a write token locally).
set -euo pipefail

DRY_RUN=0
PRS=()
OWNER="NousResearch"
REPO="hermes-agent"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --pr) PRS+=("$2"); shift 2 ;;
    --owner) OWNER="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

FULL="${OWNER}/${REPO}"
: "${GH_TOKEN:?GH_TOKEN must be set}"

SECURITY_PATHS=(
  '.github/**'
  'scripts/deploy*'
  'package.json'
  'package-lock.json'
  'pyproject.toml'
  'uv.lock'
  'requirements*.txt'
  '.woodpecker.yml'
  'Makefile'
)

log() { echo "[fast-track] $*"; }

merge_one() {
  local pr="$1" author is_draft mergeable merge_state additions deletions updated
  local changed_lines files bad_file age_h behind head_owner head_ref
  # 1. Fetch PR metadata.
  local meta
  meta=$(gh pr view "$pr" --repo "$FULL" --json author,isDraft,mergeable,mergeStateStatus,additions,deletions,labels,updatedAt,commits \
    --jq '{author:.author.login,isDraft,mergeable,mergeStateStatus,additions,deletions,labels:[.labels[].name],updatedAt,commits}' 2>/dev/null) || {
    log "PR #$pr: skip (cannot read PR metadata)"
    return 0
  }
  author=$(jq -r '.author' <<<"$meta")
  is_draft=$(jq -r '.isDraft' <<<"$meta")
  mergeable=$(jq -r '.mergeable' <<<"$meta")
  merge_state=$(jq -r '.mergeStateStatus' <<<"$meta")
  additions=$(jq -r '.additions' <<<"$meta")
  deletions=$(jq -r '.deletions' <<<"$meta")
  updated=$(jq -r '.updatedAt' <<<"$meta")

  # 2. Author / label gate.
  if [[ "$author" == "devbxylw" ]]; then
    log "PR #$pr: author devbxylw OK"
  elif jq -e '.labels | index("tabjoy-fleet")' <<<"$meta" >/dev/null 2>&1; then
    log "PR #$pr: tabjoy-fleet label OK"
  else
    log "PR #$pr: skip (author $author, no tabjoy-fleet label)"
    return 0
  fi

  # 2b. Draft / block labels.
  if [[ "$is_draft" == "true" ]]; then
    log "PR #$pr: skip (draft)"
    return 0
  fi
  if jq -e '.labels | index("do-not-merge") // index("blocked")' <<<"$meta" >/dev/null 2>&1; then
    log "PR #$pr: skip (do-not-merge/blocked label)"
    return 0
  fi

  # 3. Mergeability gate: GitHub's own aggregate — required checks, conflicts,
  #    required reviews. CLEAN is the only green state that satisfies the
  #    protect-main ruleset's "All required checks pass" requirement.
  if [[ "$mergeable" != "MERGEABLE" || "$merge_state" != "CLEAN" ]]; then
    log "PR #$pr: skip (mergeable=$mergeable mergeStateStatus=$merge_state; expected MERGEABLE/CLEAN)"
    return 0
  fi

  # 4. Size guard (policy 5.5).
  changed_lines=$((additions + deletions))
  if (( changed_lines > 400 )); then
    log "PR #$pr: skip (size guard: $changed_lines changed lines > 400)"
    return 0
  fi

  # 5. Security-sensitive paths (policy 5.6).
  files=$(gh pr view "$pr" --repo "$FULL" --json files --jq '.files[].path' 2>/dev/null || true)
  bad_file=""
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    for pat in "${SECURITY_PATHS[@]}"; do
      if [[ "$f" == $pat || "$f" == $pat/** ]]; then
        bad_file="$f"
        break
      fi
    done
    [[ -n "$bad_file" ]] && break
  done <<<"$files"
  if [[ -n "$bad_file" ]]; then
    log "PR #$pr: skip (security-sensitive path: $bad_file)"
    return 0
  fi

  # 6. Head freshness (policy 5.4): <= 50 commits behind, or pushed < 24h.
  #    Cross-repo PRs need `owner:branch` compare syntax; plain head refs 404.
  local head_owner head_ref
  head_owner=$(gh pr view "$pr" --repo "$FULL" --json headRepositoryOwner --jq '.headRepositoryOwner.login' 2>/dev/null || echo "$OWNER")
  head_ref=$(gh pr view "$pr" --repo "$FULL" --json headRefName --jq .headRefName 2>/dev/null || true)
  behind=$(gh api "repos/${FULL}/compare/$(gh pr view "$pr" --repo "$FULL" --json baseRefName --jq .baseRefName)...${head_owner}:${head_ref}" --jq '.behind_by' 2>/dev/null || echo 999)
  age_h=$(python3 -c "import datetime,sys; print((datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('$updated'.replace('Z','+00:00'))).total_seconds()/3600)" 2>/dev/null || echo 999)
  if (( behind > 50 )) && (( $(printf '%.0f' "$age_h") >= 24 )); then
    log "PR #$pr: skip (stale head: $behind commits behind, $age_h h since push)"
    return 0
  fi

  # All gates passed.
  if (( DRY_RUN )); then
    log "PR #$pr: ELIGIBLE (dry-run; $changed_lines lines, $behind behind, age ${age_h}h) — would squash-merge"
    return 0
  fi

  log "PR #$pr: merging (squash)"
  if gh pr merge "$pr" --repo "$FULL" --squash --delete-branch=false \
    --subject "$(gh pr view "$pr" --repo "$FULL" --json title --jq .title)" 2>/dev/null; then
    gh pr comment "$pr" --repo "$FULL" --body "🤖 Fast-track merge (SLA): green + tested per merge-sla-policy.md, merged without a manual maintainer turn." >/dev/null 2>&1 || true
    log "PR #$pr: merged"
  else
    log "PR #$pr: merge attempt failed (ruleset may have changed between check and merge)"
  fi
}

if (( ${#PRS[@]} == 0 )); then
  log "scanning open PRs for author devbxylw or tabjoy-fleet label"
  # Author flag is precise; label search is the complement for PRs labelled
  # tabjoy-fleet by someone else. The naive `--json author,labels` sweep drops
  # older PRs past the list limit, so query each axis explicitly.
  mapfile -t PRS < <(
    gh pr list --repo "$FULL" --state open --author devbxylw --limit 100 \
      --json number --jq '.[].number'
    gh pr list --repo "$FULL" --state open --label tabjoy-fleet --limit 100 \
      --json number --jq '.[].number'
  )
  # Dedupe, keep numeric order.
  mapfile -t PRS < <(printf '%s\n' "${PRS[@]}" | sort -nu)
fi

if (( ${#PRS[@]} == 0 )); then
  log "no eligible fleet PRs found"
  exit 0
fi

for pr in "${PRS[@]}"; do
  merge_one "$pr"
done
log "done (dry-run=$DRY_RUN)"
