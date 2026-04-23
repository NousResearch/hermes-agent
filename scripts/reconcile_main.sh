#!/usr/bin/env bash
# Reconcile local main vs origin/main divergence.
#
# Spec: ~/wiki/sop/worktree-integration-workflow.md §5
#
# This script is INTERACTIVE. It will not make destructive changes without
# Brian's explicit confirmation. Run it when local main is diverged (the
# dashboard's 🔀 Git divergence signal is 🟡 or 🔴).
#
# Usage:
#   ./scripts/reconcile_main.sh        # interactive
#   ./scripts/reconcile_main.sh --plan # print recommended plan only
#   ./scripts/reconcile_main.sh --help

set -e

REPO="${REPO:-$HOME/dev/hermes-agent}"
cd "$REPO"

if [ "${1:-}" = "--help" ]; then
    head -20 "$0" | tail -n +2 | sed 's/^# \?//'
    exit 0
fi

echo "=== Fetching origin ==="
git fetch origin --prune

AHEAD=$(git rev-list --count origin/main..main)
BEHIND=$(git rev-list --count main..origin/main)
echo "main: ahead $AHEAD · behind $BEHIND"

if [ "$BEHIND" -eq 0 ] && [ "$AHEAD" -eq 0 ]; then
    echo "✅ main is in sync with origin/main — nothing to do"
    exit 0
fi

echo ""
echo "=== Your unique commits (ahead=$AHEAD) ==="
git log origin/main..main --oneline

echo ""
echo "=== Cherry-pick duplicate detection ==="
LOCAL_SUBJ=$(git log origin/main..main --format="%s")
ORIGIN_SUBJ=$(git log main..origin/main --format="%s")
DUPS=$(comm -12 \
    <(printf '%s\n' "$LOCAL_SUBJ" | sort) \
    <(printf '%s\n' "$ORIGIN_SUBJ" | sort))
if [ -n "$DUPS" ]; then
    DUP_COUNT=$(echo "$DUPS" | wc -l | tr -d ' ')
    echo "⚠️  $DUP_COUNT duplicate subjects found (likely cherry-picks):"
    echo "$DUPS" | sed 's/^/  - /'
else
    echo "✓ no cherry-pick duplicates detected"
fi

if [ "${1:-}" = "--plan" ]; then
    echo ""
    echo "=== Recommended plan ==="
    if [ -n "$DUPS" ]; then
        echo "Plan C (wipe local main, rely on feature branches):"
    else
        echo "Plan B (merge origin/main into main):"
    fi
    echo "  See ~/wiki/sop/worktree-integration-workflow.md §5 for the full steps."
    exit 0
fi

echo ""
echo "=== Choose reconciliation plan ==="
echo "A) Rebase-drop duplicates, keep 6 unique commits, force-push main (RISKY)"
echo "B) Merge origin/main into main (KEEPS HISTORY UGLY BUT SAFE)"
echo "C) Reset local main to origin/main, work only on feature branches (CLEAN)"
echo "X) Exit, do nothing"
echo ""
read -r -p "Plan [A/B/C/X]: " PLAN

case "$PLAN" in
    A|a)
        echo ""
        echo "Plan A requires interactive rebase. Opening editor…"
        git checkout -b reconcile-$(date +%Y-%m-%d) main
        git rebase -i origin/main
        echo ""
        echo "Rebase complete. Review with:"
        echo "  git log origin/main..HEAD --oneline"
        echo ""
        echo "To finalize (force-push required because main is rewritten):"
        echo "  git checkout main && git reset --hard reconcile-$(date +%Y-%m-%d)"
        echo "  # VERIFY no other worktree has main checked out first"
        echo "  git push --force-with-lease origin main"
        ;;
    B|b)
        echo ""
        echo "Plan B — merging origin/main into main…"
        git checkout main
        git merge origin/main
        echo "Merge done. Test, then push when ready:"
        echo "  pytest tests/"
        echo "  git push origin main"
        ;;
    C|c)
        echo ""
        read -r -p "Plan C will RESET local main — your 9 commits will be DISCARDED from main (but still reachable via claude/dashboard-v2-phase1-complete branch). Confirm? [yes]: " CONFIRM
        if [ "$CONFIRM" = "yes" ]; then
            # Verify that dashboard branch exists and has the 9 commits
            DASH_BRANCH_HEAD=$(git rev-parse origin/claude/dashboard-v2-phase1-complete 2>/dev/null || echo "")
            if [ -z "$DASH_BRANCH_HEAD" ]; then
                echo "❌ claude/dashboard-v2-phase1-complete not found on origin — aborting to avoid data loss"
                exit 1
            fi
            echo "Safety check: local main HEAD is reachable from claude/dashboard-v2-phase1-complete"
            MAIN_HEAD=$(git rev-parse main)
            if git merge-base --is-ancestor "$MAIN_HEAD" "$DASH_BRANCH_HEAD"; then
                echo "  ✓ confirmed"
            else
                echo "  ❌ NOT reachable — local main has commits not on the backup branch"
                echo "  Run: git push origin main:claude/pre-reconcile-backup-$(date +%Y%m%d)"
                exit 1
            fi
            git checkout main
            git reset --hard origin/main
            echo "✓ local main now = origin/main"
            echo "Your 9 commits are preserved on claude/dashboard-v2-phase1-complete"
        else
            echo "aborted"
        fi
        ;;
    X|x|"")
        echo "aborted"
        ;;
    *)
        echo "unknown plan: $PLAN"
        exit 1
        ;;
esac
