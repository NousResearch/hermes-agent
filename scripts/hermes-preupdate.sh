#!/usr/bin/env bash
# hermes-preupdate: prepare for a `hermes update` run.
#
# What this does:
#   1. Snapshots the local Hermes source state (branch, commit, applied
#      patches) to ~/.hermes/state-snapshots/<timestamp>-pre-update/.
#   2. Snapshots the saved local patches to ~/.hermes/state-snapshots/...
#      /patches/ so they're not lost if the operator blows away
#      ~/.hermes/patches/ by accident.
#   3. Verifies the post-merge hook is installed and executable.
#   4. Verifies the saved patch's target file still exists in the
#      current source (i.e. the fix is still applied).
#   5. Verifies a recent `hermes update --backup` exists (within 14
#      days) — if not, suggests running one before proceeding.
#
# This script is intentionally idempotent and read-only. It does NOT
# touch /usr/local/lib/hermes-agent, does NOT run git, does NOT modify
# any Hermes config. It only writes to ~/.hermes/state-snapshots/.
#
# Usage:
#   hermes-preupdate
#   hermes-preupdate --check      # exit 1 if any check fails
#   hermes-preupdate --quiet      # no progress output, errors only

set -euo pipefail

QUIET=0
CHECK_MODE=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_MODE=1 ;;
        --quiet|-q) QUIET=1 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
REPO_ROOT="${HERMES_REPO_ROOT:-/usr/local/lib/hermes-agent}"
SNAPSHOT_BASE="$HERMES_HOME/state-snapshots"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
SNAPSHOT_DIR="$SNAPSHOT_BASE/$STAMP-pre-update"
LOG="$HERMES_HOME/logs/preupdate.log"

mkdir -p "$(dirname "$LOG")"

say() {
    if [[ $QUIET -eq 0 ]]; then
        printf '%s\n' "$*"
    fi
}

err() {
    printf '✗ %s\n' "$*" >&2
    printf '[%s] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"
}

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"
}

EXIT=0

log "hermes-preupdate started (check=$CHECK_MODE quiet=$QUIET)"

# Check 1: HERMES_HOME exists
if [[ ! -d "$HERMES_HOME" ]]; then
    err "$HERMES_HOME does not exist"
    exit 1
fi

# Check 2: source checkout is present
if [[ ! -d "$REPO_ROOT/.git" ]]; then
    err "$REPO_ROOT is not a git checkout — this script only handles the git-based install"
    exit 1
fi

say "◆ Creating pre-update snapshot at $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR/patches"
log "snapshot dir: $SNAPSHOT_DIR"

# Capture git state
{
    echo "# git state captured $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "branch: $(git -C "$REPO_ROOT" branch --show-current 2>&1)"
    echo "commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>&1)"
    echo "remote: $(git -C "$REPO_ROOT" remote get-url origin 2>&1)"
    echo "status: "
    git -C "$REPO_ROOT" status --short 2>&1 | sed 's/^/  /'
    echo "applied-patches: $(git -C "$REPO_ROOT" log --oneline -5 2>&1)"
} > "$SNAPSHOT_DIR/git-state.txt"
say "  git state → $SNAPSHOT_DIR/git-state.txt"

# Copy local patches
if [[ -d "$HERMES_HOME/patches" ]]; then
    cp -a "$HERMES_HOME/patches/." "$SNAPSHOT_DIR/patches/"
    say "  patches   → $SNAPSHOT_DIR/patches/ ($(ls "$SNAPSHOT_DIR/patches/" | wc -l) file(s))"
else
    say "  no patches/ directory to snapshot"
fi

# Check 3: post-merge hook is installed
HOOK_PATH="$REPO_ROOT/.git/hooks/post-merge"
if [[ -x "$HOOK_PATH" ]]; then
    say "  ✓ post-merge hook installed at $HOOK_PATH"
else
    err "post-merge hook is missing or not executable at $HOOK_PATH"
    err "  re-install with: cp ~/.hermes/bin/hermes-post-merge-hook.sh $HOOK_PATH && chmod +x $HOOK_PATH"
    EXIT=1
fi

# Check 4: saved patch's canary is present in current source
if [[ -d "$HERMES_HOME/patches" ]]; then
    for patch in "$HERMES_HOME/patches/"*.patch; do
        [[ -f "$patch" ]] || continue
        # Try to detect the target file from the patch header
        target=$(grep -m1 '^diff --git' "$patch" | awk '{print $NF}' | sed 's|^[ab]/||')
        marker=$(grep -m1 '+++ ' "$patch" | head -1)
        # The canary is the function name we added
        if grep -q '_build_minimax_oauth_aux_client' "$patch" 2>/dev/null; then
            if grep -qF '_build_minimax_oauth_aux_client' "$REPO_ROOT/$target" 2>/dev/null; then
                say "  ✓ patch $(basename "$patch") is currently applied (canary in $target)"
            else
                err "patch $(basename "$patch") is saved but the fix is MISSING from $target"
                err "  the post-merge hook will re-apply it, but the current state is broken"
                EXIT=1
            fi
        fi
    done
fi

# Check 5: recent backup exists (within 14 days)
if [[ -d "$HERMES_HOME/backups" ]]; then
    LATEST_BACKUP=$(find "$HERMES_HOME/backups" -name 'pre-update-*.zip' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
    if [[ -n "$LATEST_BACKUP" ]]; then
        BACKUP_AGE_DAYS=$(( ( $(date +%s) - $(stat -c %Y "$LATEST_BACKUP") ) / 86400 ))
        if [[ $BACKUP_AGE_DAYS -le 14 ]]; then
            say "  ✓ latest backup is $BACKUP_AGE_DAYS day(s) old: $(basename "$LATEST_BACKUP")"
        else
            err "latest backup is $BACKUP_AGE_DAYS day(s) old — consider running 'hermes update --backup' first"
            EXIT=1
        fi
    else
        say "  ⚠ no pre-update backup found in $HERMES_HOME/backups/"
        say "    consider running: hermes update --backup  (or)  hermes backup"
        if [[ $CHECK_MODE -eq 1 ]]; then
            EXIT=1
        fi
    fi
else
    say "  ⚠ no backups directory yet — first update will create one"
fi

say ""
say "Snapshot saved. To run the update safely:"
say "  hermes update --backup"
say ""
say "To restore from this snapshot if something goes wrong:"
say "  cp -a $SNAPSHOT_DIR/patches/. $HERMES_HOME/patches/"
say "  cd $REPO_ROOT && git checkout \$(awk '/commit:/ {print \$2}' $SNAPSHOT_DIR/git-state.txt)"

log "hermes-preupdate finished (exit=$EXIT)"

if [[ $CHECK_MODE -eq 1 ]]; then
    exit $EXIT
fi
