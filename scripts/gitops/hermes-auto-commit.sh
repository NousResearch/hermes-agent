#!/usr/bin/env bash
# KENSEI GitOps — debounced auto-commit watcher for ~/.hermes
# Cron-driven: 5-min debounce. Commits curated paths only.
# Idempotent: no-op when there is nothing to commit.
#
# Deployed by: scripts/gitops/install.sh
# Cron entry:  */5 * * * *  /home/kensei/.hermes/scripts/gitops/hermes-auto-commit.sh

set -euo pipefail

# P13 isolation: GITOPS_DRY_RUN=1 short-circuits before any git add/commit
# so the watcher can be exercised without mutating the repo.
if [[ "${GITOPS_DRY_RUN:-0}" == "1" ]]; then
    echo "dry-run: would commit config changes"
    exit 0
fi

HERMES_HOME="${HERMES_HOME:-/home/kensei/.hermes}"
STATE_DIR="$HERMES_HOME/scripts/gitops/state"
STATE_FILE="$STATE_DIR/last-watch"
LOCK_FILE="$STATE_DIR/.lock"
LOG_FILE="$STATE_DIR/watch.log"

# Curated paths to auto-track. Keep in sync with the .gitignore template.
# `.gitignore` is included so rule changes are tracked; `profiles` is a git
# submodule, so add updates the gitlink pointer (no-op if pointer unchanged).
TRACK_PATHS=(
  ".gitignore"
  "config.yaml"
  "profiles"
  "cron"
  "governance"
  "scripts"
  "skills"
  "runbooks"
  "feature-artifacts"
)

mkdir -p "$STATE_DIR"

# Exit cleanly if the repo is not initialised yet.
if [[ ! -d "$HERMES_HOME/.git" ]]; then
  echo "[$(date -u +%FT%TZ)] .git not present; nothing to do" >> "$LOG_FILE"
  exit 0
fi

# Lock against overlapping runs (cron + manual).
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date -u +%FT%TZ)] another watcher run is in progress" >> "$LOG_FILE"
  exit 0
fi

cd "$HERMES_HOME"

# Stage only curated paths. Never `git add .` — 10G of state would drown git.
# `--quiet` keeps the cron output free of noise.
git add -- "${TRACK_PATHS[@]}" 2>/dev/null || true

# Unstage cron/jobs.json — it mutates on every scheduler tick
# (last_run_at, next_run_at, repeat.completed, updated_at) and would
# cause a commit every 5 minutes. Job definitions are tracked via the
# curated allowlist; runtime metadata is not.
git reset -- cron/jobs.json 2>/dev/null || true

# If nothing is staged, exit cleanly.
if git diff --cached --quiet; then
  echo "[$(date -u +%FT%TZ)] no curated-path changes" >> "$LOG_FILE"
  exit 0
fi

# Detect what changed for a useful commit message.
CHANGED=$(git diff --cached --name-only | head -10 | tr '\n' ' ')
COUNT=$(git diff --cached --name-only | wc -l | tr -d ' ')

# Commit with author + structured message. Author is the running user.
COMMIT_MSG="auto: ${COUNT} curated-path change(s)

${CHANGED}
- triggered by: hermes-auto-commit.sh (5min debounce)
- hermes_home: ${HERMES_HOME}"

# Use the local KENSEI identity; committer/author come from git config.
git commit --quiet -m "$COMMIT_MSG"

date -u +%FT%TZ > "$STATE_FILE"
echo "[$(date -u +%FT%TZ)] committed ${COUNT} file(s): ${CHANGED}" >> "$LOG_FILE"
