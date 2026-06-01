#!/usr/bin/env bash
# post-merge hook: re-apply the saved Hermes local patches if upstream
# `git pull` reverted them.
#
# Context: this server runs a custom fix for the minimax-oauth auxiliary
# routing bug (PR #36779, branch fix/minimax-oauth-auxiliary-routing). If
# a future `hermes update` (which calls `git pull --ff-only` on
# /usr/local/lib/hermes-agent) lands before upstream merges the fix, the
# local source will silently revert to the broken state.
#
# This hook checks after every merge whether the fix is still present in
# agent/auxiliary_client.py. If not, it re-applies the saved patch from
# ~/.hermes/patches/ and warns the operator. The patch is a no-op if
# upstream ever merges the fix — git apply --check will report
# "patch does not apply" and the hook exits 0.
#
# Triggered by: any successful `git pull` / `git merge` inside the
# /usr/local/lib/hermes-agent checkout. This includes `hermes update` on
# the git-based install path.
#
# Why a hook and not a wrapper: hooks run on the actual git operation
# with no way for the user to bypass them via an alias or a one-off
# command, and they're installed exactly once per checkout.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PATCH_DIR="$HERMES_HOME/patches"
LOG_FILE="$HERMES_HOME/logs/post-merge-hook.log"

# Helper target file: if the helper function is missing, the patch was
# dropped. This is the canary — the function was added in the fix commit
# and is the smallest, most specific marker.
CANARY_FILE="agent/auxiliary_client.py"
CANARY_MARKER="_build_minimax_oauth_aux_client"

# Ensure the log directory exists. We log to ~/.hermes/logs/ so the
# hook can surface failures to the operator even when invoked from
# a non-interactive `hermes update` (where stdout may be silenced).
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_FILE"
}

log "post-merge hook fired in $REPO_ROOT"

# Only act on the actual code checkout. ~/.hermes itself is not a git
# repo so the toplevel resolution never points there.
if [[ "$REPO_ROOT" != "/usr/local/lib/hermes-agent"* ]]; then
    log "  not the Hermes checkout — skipping"
    exit 0
fi

# Check 1: is the fix still present in the current source?
if [[ -f "$REPO_ROOT/$CANARY_FILE" ]] && \
   grep -qF "$CANARY_MARKER" "$REPO_ROOT/$CANARY_FILE"; then
    log "  fix present in $CANARY_FILE — nothing to do"
    exit 0
fi

log "  ⚠ fix marker '$CANARY_MARKER' missing from $CANARY_FILE"
log "  upstream merge appears to have reverted the local patch"

# Check 2: do we have a saved patch to re-apply?
if [[ ! -d "$PATCH_DIR" ]]; then
    log "  ✗ no patch directory at $PATCH_DIR — cannot recover"
    log "  manual fix required; see PR #36779"
    exit 0
fi

PATCH_FILE=""
for candidate in "$PATCH_DIR"/*.patch; do
    if [[ -f "$candidate" ]]; then
        PATCH_FILE="$candidate"
        break
    fi
done

if [[ -z "$PATCH_FILE" ]]; then
    log "  ✗ no saved patches in $PATCH_DIR — cannot recover"
    exit 0
fi

# Check 3: does the patch even apply? If upstream already merged the
# fix, git apply --check will fail and we want to silently exit 0 —
# the fix is in main, our local patch is redundant.
log "  attempting to re-apply $PATCH_FILE"
if ! git apply --check "$PATCH_FILE" 2>/dev/null; then
    if grep -qF "$CANARY_MARKER" "$REPO_ROOT/$CANARY_FILE" 2>/dev/null; then
        log "  patch would not apply cleanly, but canary is present —"
        log "  assuming upstream merged the fix. No action needed."
    else
        log "  ✗ patch does not apply and canary is missing —"
        log "  upstream may have restructured the file. Manual fix required."
    fi
    exit 0
fi

# Re-apply using a recorded commit hash (preferred) or fall back to the
# raw .patch file.  The commit-hash path is more robust: it uses
# `git cherry-pick --3way` so context-line drift in the upstream
# source doesn't break recovery.
#
# manifest.txt format (one line per patch):
#   <full-sha>  <branch-name>  [<description>]
#
# Example:
#   25222e49068daa243a45850a43e92a6498e6abf5  fix/minimax-oauth-auxiliary-routing
MANIFEST="$PATCH_DIR/manifest.txt"

applied_something=0

if [[ -f "$MANIFEST" ]]; then
    while IFS= read -r line; do
        # Skip comments and blanks
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        # Parse "<sha>  <branch>  [desc...]"
        sha=$(awk '{print $1}' <<<"$line")
        branch=$(awk '{print $2}' <<<"$line")
        [[ -z "$sha" ]] && continue
        # Only act on entries that match our canary (cheap filter)
        if ! grep -q '_build_minimax_oauth_aux_client' <<<"$line" \
           && ! git -C "$REPO_ROOT" cat-file -e "$sha" 2>/dev/null; then
            continue
        fi
        # Confirm this commit introduces the canary function
        if ! git -C "$REPO_ROOT" show "$sha" 2>/dev/null | grep -q '_build_minimax_oauth_aux_client'; then
            continue
        fi
        # Only re-apply if the canary is currently missing
        if grep -qF "_build_minimax_oauth_aux_client" "$REPO_ROOT/$CANARY_FILE" 2>/dev/null; then
            log "  ($sha) canary present — skipping"
            continue
        fi
        log "  attempting cherry-pick --3way of $sha ($branch)"
        # Cherry-pick onto the current HEAD without committing.  --3way
        # falls back to 3-way merge when context lines don't match
        # exactly, which is the common case after a rebase or refactor.
        if git -C "$REPO_ROOT" cherry-pick --no-commit --3way "$sha" >>"$LOG_FILE" 2>&1; then
            log "  ✓ cherry-pick of $sha applied (uncommitted)"
            applied_something=1
        else
            log "  ✗ cherry-pick of $sha failed — aborting this attempt"
            git -C "$REPO_ROOT" cherry-pick --abort 2>/dev/null || true
        fi
    done < "$MANIFEST"
fi

# If the commit-hash path didn't apply anything, try the raw .patch
# files.  This is the legacy fallback for hooks installed before
# manifest.txt was introduced.
if [[ $applied_something -eq 0 ]]; then
    for candidate in "$PATCH_DIR"/*.patch; do
        [[ -f "$candidate" ]] || continue
        log "  attempting git apply on $candidate (fallback)"
        if git -C "$REPO_ROOT" apply --3way "$candidate" 2>>"$LOG_FILE"; then
            log "  ✓ git apply succeeded for $(basename "$candidate")"
            applied_something=1
            break
        else
            log "  ✗ git apply --3way failed for $(basename "$candidate")"
        fi
    done
fi

if [[ $applied_something -eq 0 ]]; then
    log "  ✗ no patch from $PATCH_DIR could be applied — manual fix required"
    log "  see PR #36779 for context"
    cat >&2 <<EOF

⚠ HERMES LOCAL PATCH RECOVERY FAILED
  The minimax-oauth auxiliary fix (PR #36779) was reverted by an
  upstream merge and could not be auto-re-applied.
  Saved patches: $PATCH_DIR
  Log: $LOG_FILE
  Action: cd $REPO_ROOT && git cherry-pick 25222e49068daa243a45850a43e92a6498e6abf5
EOF
    exit 0
fi

cat >&2 <<EOF

⚠ HERMES LOCAL PATCH AUTO-REAPPLIED
  Patch: $CANARY_MARKER (PR #36779)
  Reason: upstream merge reverted the local minimax-oauth auxiliary fix.
  Review with: cd $REPO_ROOT && git diff
  Log: $LOG_FILE
EOF
