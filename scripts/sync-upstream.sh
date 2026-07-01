#!/usr/bin/env bash
# Check upstream/main against main for conflicts, then either merge automatically
# or hand off to a PR for manual resolution.
#
# - Clean merge: open a PR against main and merge it immediately. Local main is
#   left as-is (now one commit behind origin/main) — main itself is never used
#   for the merge attempt directly, so this can never leave it dirty. Follow up
#   with `hermes update` (e.g. `sync-upstream.sh && hermes update --yes`), which
#   will see new commits on origin/main and run its full pull+rebuild+restart
#   pipeline.
# - Conflicting merge: open a PR from upstream/main into main so you can review
#   and resolve the conflicts yourself (via the PR's "Resolve conflicts" UI, or
#   by checking out the branch locally, merging main into it, fixing conflicts,
#   and pushing). The PR URL is printed and the script exits non-zero; main is
#   left untouched.
#
# Remotes expected:
#   origin   -> your fork   (https://github.com/alonre/hermes-agent.git)
#   upstream -> NousResearch/hermes-agent.git
#
# Requires: gh CLI authenticated with `repo` scope, plus `workflow` if
# .github/workflows/ files change.
set -euo pipefail

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree is not clean; commit or stash your changes first" >&2
  exit 1
fi

git fetch upstream
git fetch origin

if ! git merge-base --is-ancestor main origin/main; then
  echo "error: local main has commits not on origin/main; push or reset local main first" >&2
  exit 1
fi

if git merge-base --is-ancestor upstream/main origin/main; then
  echo "Already up to date with upstream/main."
  exit 0
fi

# gh resolves an unset default repo to the fork's *parent* (upstream), so PRs must
# name the fork explicitly or `gh pr create` opens against upstream and fails.
# Derive the fork's owner/repo from the origin remote (handles https and ssh URLs).
FORK_REPO="$(git remote get-url origin)"
FORK_REPO="${FORK_REPO#*github.com}"
FORK_REPO="${FORK_REPO#[:/]}"
FORK_REPO="${FORK_REPO%.git}"

# Fork invariants: local deltas a clean upstream merge could SILENTLY undo.
# Each entry is a symbol that must NOT reappear in the listed files; if it does,
# the merge re-landed an upstream change we deliberately reverted, and
# auto-merging would regress the live fleet. A conflict gets caught by git; a
# clean re-introduction on adjacent lines does not — this is the net for that.
# Add an entry whenever you revert an upstream commit a future sync could re-land.
#
#   forbidden_symbol | files | why
# get_default_hermes_root | cron/jobs.py cron/suggestions.py cron/scheduler.py |
#   cron storage must stay PER-PROFILE (revert of a5c09fd17 / #32091). This fleet
#   runs one `--profile <name>` gateway per agent, each ticking its own
#   ~/.hermes/profiles/<name>/cron; root-anchoring collapses all agents onto one
#   shared store and orphans their profile-local jobs (e.g. property-ops'
#   revital-email-sweep). Cron must resolve from get_hermes_home(). Drop this
#   entry only once upstream lands per-job profile-execution scoping (#48649).
FORK_INVARIANTS=(
  "get_default_hermes_root|cron/jobs.py cron/suggestions.py cron/scheduler.py|cron storage must stay per-profile (revert of a5c09fd17/#32091)"
)

# Returns non-zero (and reports) if any invariant is violated in the working tree.
check_fork_invariants() {
  local violated=0 entry symbol files why
  for entry in "${FORK_INVARIANTS[@]}"; do
    IFS='|' read -r symbol files why <<<"$entry"
    # Match the symbol, but drop hits on comment-only lines: these files carry
    # explanatory comments that name the forbidden symbol (e.g. cron/jobs.py's
    # "...NOT get_default_hermes_root()"), and a raw match would false-trip on
    # that prose on every clean sync. The `grep -v` filters `file:lineno:` rows
    # whose code starts with `#`; only real code re-introductions survive.
    # shellcheck disable=SC2086 — $files is an intentional space-separated list.
    if grep -nF "$symbol" $files 2>/dev/null | grep -vE '^[^:]+:[0-9]+:[[:space:]]*#'; then
      echo "  ^ FORK INVARIANT VIOLATED: '$symbol' reappeared — $why" >&2
      violated=1
    fi
  done
  return "$violated"
}

ORIGINAL_BRANCH="$(git branch --show-current)"
SYNC_BRANCH="sync-upstream-$(date +%Y%m%d-%H%M%S)"

git checkout -b "$SYNC_BRANCH" origin/main

if git merge --no-edit upstream/main; then
  # The merge was conflict-free, but a clean merge can still re-land an upstream
  # change we deliberately reverted (re-introduced on lines git didn't flag as a
  # conflict). Gate the auto-merge on the fork invariants; on violation, hand off
  # to a PR for manual re-revert instead of silently regressing the live fleet.
  if ! check_fork_invariants; then
    echo "error: clean merge re-introduced a reverted fork delta; NOT auto-merging." >&2
    git push -u origin "$SYNC_BRANCH"
    PR_URL="$(gh pr create \
      --repo "$FORK_REPO" \
      --base main \
      --head "$SYNC_BRANCH" \
      --title "Sync upstream/main - REVERTED DELTA REINTRODUCED, needs manual re-revert ($(date +%Y-%m-%d))" \
      --body "Automated sync merged cleanly but tripped a fork invariant (see check_fork_invariants in scripts/sync-upstream.sh): an upstream change this fork deliberately reverted has come back. **Do NOT merge as-is** — it would regress the live fleet. Check out $SYNC_BRANCH, re-apply the revert, push, then merge this PR yourself.")"
    git checkout "$ORIGINAL_BRANCH"
    git branch -D "$SYNC_BRANCH"
    echo "error: opened PR for manual re-revert: $PR_URL" >&2
    exit 1
  fi

  # Push and open the PR before deleting the local branch, so a gh failure
  # leaves the branch recoverable locally rather than stranded only on origin.
  git push -u origin "$SYNC_BRANCH"

  PR_URL="$(gh pr create \
    --repo "$FORK_REPO" \
    --base main \
    --head "$SYNC_BRANCH" \
    --title "Sync upstream/main ($(date +%Y-%m-%d))" \
    --body "Automated sync from NousResearch/hermes-agent main. Merged automatically after a conflict-free local check.")"

  git checkout "$ORIGINAL_BRANCH"
  git branch -D "$SYNC_BRANCH"

  echo "Opened PR: $PR_URL"
  # main is branch-protected: it requires the "All required checks pass" CI
  # aggregate before anything can merge. Enable auto-merge instead of an
  # immediate merge (which would fail while checks are still pending) — the PR
  # lands by itself once CI is green, and stays open (fail-closed) if CI is red,
  # so a regression never reaches the live fleet unattended.
  gh pr merge "$PR_URL" --repo "$FORK_REPO" --merge --auto --delete-branch

  echo "Auto-merge enabled; PR will merge once CI ('All required checks pass') is green."
  echo "The merge is asynchronous — run 'hermes update' after it lands (e.g. next nightly cycle) to pull, rebuild, and restart."
else
  git merge --abort
  git checkout "$ORIGINAL_BRANCH"
  git branch -D "$SYNC_BRANCH"

  # Push and open the PR before deleting the local branch, so a gh failure
  # leaves the branch recoverable locally rather than stranded only on origin.
  git branch "$SYNC_BRANCH" upstream/main
  git push -u origin "$SYNC_BRANCH"

  PR_URL="$(gh pr create \
    --repo "$FORK_REPO" \
    --base main \
    --head "$SYNC_BRANCH" \
    --title "Sync upstream/main - CONFLICTS, needs manual resolution ($(date +%Y-%m-%d))" \
    --body "Automated sync from NousResearch/hermes-agent main hit merge conflicts with main. Resolve them in this PR: checkout $SYNC_BRANCH, merge main into it, fix the conflicts, push, then merge the PR yourself.")"

  git branch -D "$SYNC_BRANCH"

  echo "error: upstream/main has conflicts with main; opened PR for manual resolution: $PR_URL" >&2
  exit 1
fi
