#!/usr/bin/env bash
# hermes-rebuild-release.sh
# Updates release/ironin by merging origin/main while preserving ALL feature code.
#
# Usage: ./scripts/hermes-rebuild-release.sh [--push]
#
# APPROACH:
#   1. Merge origin/main INTO release/ironin (not the other way around!)
#   2. Resolve conflicts by keeping our feature code (-Xours from release/ironin)
#   3. Verify key features are still present
#   4. Optionally push to fork
#
# WHY NOT rebuild from scratch:
#   Features like subagent-panel and stash-panel don't have clean ironin/* branches.
#   Rebuilding from origin/main + clean branches would lose them.
#   Merging origin/main INTO release/ironin preserves everything.

set -euo pipefail
cd "$(dirname "$0")/.."

PUSH=false
if [[ "${1:-}" == "--push" ]]; then
    PUSH=true
fi

echo "=== Fetching latest ==="
git fetch origin 2>/dev/null || echo "  (origin fetch skipped)"
git fetch fork 2>/dev/null || echo "  (fork fetch skipped)"

# Check release/ironin exists
if ! git rev-parse --verify release/ironin >/dev/null 2>&1; then
    echo "❌ release/ironin does not exist."
    echo "  Create it first from a known-good state, then re-run."
    exit 1
fi

current=$(git branch --show-current 2>/dev/null)
if [[ "$current" != "release/ironin" ]]; then
    git checkout release/ironin 2>/dev/null
fi

echo ""
echo "=== Merging origin/main into release/ironin ==="

ahead=$(git rev-list --count origin/main..release/ironin 2>/dev/null || echo "?")
behind=$(git rev-list --count release/ironin..origin/main 2>/dev/null || echo "?")
echo "  release/ironin: $ahead commits ahead, $behind behind origin/main"

if [[ "$behind" -eq 0 ]] || [[ "$behind" == "?" ]]; then
    echo "  Already up to date with origin/main"
else
    # Merge origin/main into release/ironin, keeping our feature code on conflicts
    if git merge origin/main --no-edit -Xours -q 2>&1; then
        echo "  Merge complete (feature code preserved)"
    else
        echo ""
        echo "⚠️  Merge had conflicts. Auto-resolved by keeping our feature code."
        echo "  Review the changes before pushing."
    fi
fi

echo ""
echo "=== Feature verification ==="
MISSING=0

check_feature() {
    local pattern="$1"
    local desc="$2"
    local file="${3:-cli.py}"
    local count
    count=$(git show release/ironin:"$file" 2>/dev/null | grep -c "$pattern" || true)
    if [[ "$count" -gt 0 ]]; then
        echo "  ✓ $desc ($count refs)"
    else
        echo "  ❌ $desc — MISSING from $file"
        MISSING=$((MISSING + 1))
    fi
}

check_feature "_render_resume_panel" "resume panel renderer"
check_feature "_resume_panel_open" "resume panel state"
check_feature "resume_panel_widget=resume_panel_widget" "resume panel in layout"
check_feature "not self._resume_panel_open" "resume blocks enter key"
check_feature "'any'" "resume blocks typed input"
check_feature "self._session_db.get_session(target)" "resume uses local session_db"
check_feature "_subagent_panel_open" "subagent/ctrlx panel"
check_feature "_stash_panel_open" "stash panel"
check_feature "max_stream_retries" "stream retries config"
check_feature "max_api_retries" "API retries config"
check_feature "_followup_queue" "dual/followup queue"
check_feature "terminal_title" "terminal title"
check_feature "_paste_collapse_threshold" "paste collapse"

echo ""
if [[ $MISSING -gt 0 ]]; then
    echo "❌ $MISSING feature(s) MISSING — DO NOT PUSH!"
    echo "  Something went wrong. Check the merge output."
    exit 1
fi

echo "  All features verified ✓"
echo "  release/ironin: $(git log --oneline origin/main..release/ironin | wc -l | tr -d ' ') commits ahead of origin/main"

if [[ "$PUSH" == "true" ]]; then
    echo ""
    echo "=== Pushing to fork ==="
    git push fork release/ironin --force-with-lease
    echo "  Pushed release/ironin"
fi
