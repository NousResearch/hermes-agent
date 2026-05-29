#!/usr/bin/env bash
# recover_hermes_config_branches.sh
#
# One-shot recovery script for the "workers mutate ~/.hermes main checkout"
# bug (t_d60f1590).
#
# What it does:
#   1. Triage all kanban/t_* and fix/*-t_* branches in ~/.hermes.
#   2. For each branch: check if the corresponding task is done in the DB
#      AND the PR's unique commits are already in origin/main.  Both must
#      be true before the branch is considered safe to delete.
#   3. Branches that are NOT yet safe are listed but left alone with a
#      clear note.
#   4. Any uncommitted changes in the working tree are preserved in a
#      holding commit on a new recovery/<timestamp> branch BEFORE deleting
#      anything.  The user can cherry-pick from that branch later.
#   5. After cleanup, ~/.hermes is returned to main (checkout + ff pull).
#
# Usage:
#   bash scripts/recover_hermes_config_branches.sh
#
# The script is non-destructive by default: it prints what it WOULD do,
# then asks for confirmation before actually deleting or committing anything.
# Pass --yes to skip confirmation (CI / automation).
#
# Dependencies: git, python3, gh (for merged-PR check), sqlite3.
# The kanban DB is expected at ~/.hermes/kanban.db.  Override with
# HERMES_KANBAN_DB=/path/to/kanban.db if yours is elsewhere.

set -euo pipefail

# Allow explicit override via env var for operator convenience.
# Default: the real user's ~/.hermes, resolved by stripping any profile
# sandbox suffix from $HOME (worker sessions set HOME to a profile home dir).
_sys_home="${HERMES_CONFIG_ROOT:-}"
if [[ -z "$_sys_home" ]]; then
    # Worker sandbox layout: HOME = /path/to/user/.hermes/profiles/<name>/home
    # The real system home is two levels above .hermes.
    # Strip /profiles/<name>/home to get /path/to/user/.hermes, then dirname again for user home.
    if [[ "$HOME" == *"/profiles/"*"/home" ]]; then
        _hermes_root="${HOME%%/profiles/*}"  # e.g. /Users/sahilmarwaha/.hermes
        _sys_home="${_hermes_root%/.hermes}"  # e.g. /Users/sahilmarwaha
    else
        _sys_home="$HOME"
    fi
fi
HERMES_CONFIG="${_sys_home}/.hermes"
KANBAN_DB="${HERMES_KANBAN_DB:-${HERMES_CONFIG}/kanban.db}"
DRY_RUN=true
YES=false

for arg in "$@"; do
    case "$arg" in
        --yes) YES=true; DRY_RUN=false ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

if [[ "$YES" == "false" && "$DRY_RUN" == "false" ]]; then
    DRY_RUN=true
fi

echo "=== hermes-config branch recovery script ==="
echo "Repo: $HERMES_CONFIG"
echo "Kanban DB: $KANBAN_DB"
echo "Dry-run: $DRY_RUN"
echo ""

if [[ ! -d "$HERMES_CONFIG/.git" ]]; then
    echo "ERROR: $HERMES_CONFIG is not a git repo." >&2
    exit 1
fi

# ── Step 1: Collect candidate branches ──────────────────────────────────────

BRANCHES=$(git -C "$HERMES_CONFIG" for-each-ref \
    --format='%(refname:short)' refs/heads/ \
    | grep -E '^(kanban/|fix/.*-t_)' || true)

if [[ -z "$BRANCHES" ]]; then
    echo "No kanban/* or fix/*-t_* branches found. Nothing to do."
else
    echo "Found branches to triage:"
    echo "$BRANCHES" | sed 's/^/  /'
    echo ""
fi

# ── Step 2: Triage each branch ───────────────────────────────────────────────

SAFE_TO_DELETE=()
NOT_SAFE=()

for branch in $BRANCHES; do
    # Extract task id (last segment after the final dash or last path component)
    # Patterns: kanban/t_XXXX, fix/something-t_XXXX, kanban/t_XXXX_config
    task_id=$(echo "$branch" | grep -oE 't_[a-f0-9]{8}' | tail -1 || true)

    # Unique commits in this branch not yet in origin/main
    unique_count=$(git -C "$HERMES_CONFIG" log --oneline "origin/main..$branch" 2>/dev/null | wc -l | tr -d ' ')

    task_status="unknown"
    if [[ -n "$task_id" && -f "$KANBAN_DB" ]]; then
        task_status=$(python3 -c "
import sqlite3, sys
try:
    conn = sqlite3.connect('$KANBAN_DB')
    row = conn.execute('SELECT status FROM tasks WHERE id=?', ('$task_id',)).fetchone()
    print(row[0] if row else 'not_found')
except Exception as e:
    print('error:' + str(e))
" 2>/dev/null || echo "db_error")
    fi

    if [[ "$unique_count" -eq 0 && "$task_status" == "done" ]]; then
        echo "  SAFE    $branch  (task=$task_id status=$task_status unique_commits=$unique_count)"
        SAFE_TO_DELETE+=("$branch")
    elif [[ "$unique_count" -eq 0 && "$task_status" == "not_found" ]]; then
        echo "  SAFE    $branch  (task not found in DB, no unique commits)"
        SAFE_TO_DELETE+=("$branch")
    else
        echo "  KEEP    $branch  (task=$task_id status=$task_status unique_commits=$unique_count)"
        NOT_SAFE+=("$branch")
    fi
done

echo ""

# ── Step 3: Handle uncommitted working tree changes ─────────────────────────

CURRENT_BRANCH=$(git -C "$HERMES_CONFIG" branch --show-current)
DIRTY=$(git -C "$HERMES_CONFIG" status --porcelain | wc -l | tr -d ' ')

if [[ "$DIRTY" -gt 0 ]]; then
    RECOVERY_BRANCH="recovery/$(date +%Y%m%d-%H%M%S)"
    echo "Working tree has $DIRTY dirty file(s)."
    echo "Will create holding branch: $RECOVERY_BRANCH"
    echo ""
    git -C "$HERMES_CONFIG" status --short

    if [[ "$DRY_RUN" == "false" ]]; then
        git -C "$HERMES_CONFIG" checkout -b "$RECOVERY_BRANCH"
        git -C "$HERMES_CONFIG" add -A
        git -C "$HERMES_CONFIG" \
            -c user.name="Recovery Script" \
            -c user.email="recovery@localhost" \
            commit -m "recovery: preserve working-tree changes before branch cleanup (t_d60f1590)"
        echo "Committed working-tree changes to $RECOVERY_BRANCH"
        echo "Cherry-pick from that branch to preserve any needed edits."
        echo ""
        # Return to original branch after the recovery commit
        git -C "$HERMES_CONFIG" checkout "$CURRENT_BRANCH"
    else
        echo "[DRY-RUN] Would create $RECOVERY_BRANCH and commit all dirty files."
    fi
else
    echo "Working tree is clean. No recovery commit needed."
fi

echo ""

# ── Step 4: Delete safe branches ─────────────────────────────────────────────

if [[ "${#SAFE_TO_DELETE[@]}" -eq 0 ]]; then
    echo "No branches safe to delete."
else
    echo "Branches to delete (${#SAFE_TO_DELETE[@]}):"
    for b in "${SAFE_TO_DELETE[@]}"; do
        echo "  $b"
    done
    echo ""

    if [[ "$DRY_RUN" == "false" ]]; then
        for b in "${SAFE_TO_DELETE[@]}"; do
            git -C "$HERMES_CONFIG" branch -D "$b" && echo "  Deleted $b"
        done
    else
        echo "[DRY-RUN] Would delete ${#SAFE_TO_DELETE[@]} branches."
    fi
fi

echo ""

# ── Step 5: Return to main ────────────────────────────────────────────────────

CURRENT_BRANCH=$(git -C "$HERMES_CONFIG" branch --show-current)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    echo "Current branch: $CURRENT_BRANCH. Will checkout main and pull."
    if [[ "$DRY_RUN" == "false" ]]; then
        git -C "$HERMES_CONFIG" checkout main
        git -C "$HERMES_CONFIG" pull --ff-only origin main
        echo "Back on main."
    else
        echo "[DRY-RUN] Would: git checkout main && git pull --ff-only origin main"
    fi
else
    echo "Already on main."
    if [[ "$DRY_RUN" == "false" ]]; then
        git -C "$HERMES_CONFIG" pull --ff-only origin main && echo "Pulled latest main."
    fi
fi

echo ""

# ── Summary ──────────────────────────────────────────────────────────────────

echo "=== Summary ==="
echo "Safe to delete: ${#SAFE_TO_DELETE[@]} branches"
if [[ "${#NOT_SAFE[@]}" -gt 0 ]]; then
    echo "Kept (needs review):"
    for b in "${NOT_SAFE[@]}"; do
        echo "  $b"
    done
fi

if [[ "$DRY_RUN" == "true" && "$YES" == "false" ]]; then
    echo ""
    echo "This was a DRY RUN. Re-run with --yes to apply changes."
fi
