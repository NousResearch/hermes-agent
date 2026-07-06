#!/usr/bin/env bash
# Upstream sync helper for the KarinAI Hermes-agent fork.
#
# Automates the mechanical part of docs/karinai-patches.md → "Upstream sync
# checklist": fetch upstream tags, create a sync worktree/branch off main,
# merge the chosen upstream release tag, and (once conflicts are resolved)
# run the KarinAI gates. It STOPS at "open the PR" — merging stays user-gated.
#
# Usage:
#   karinai/scripts/sync_upstream.sh start [<tag>]   # default: latest upstream tag
#   karinai/scripts/sync_upstream.sh gates           # run inside the sync worktree
#
# Typical cycle:
#   1. start  → worktree at ../worktrees/agent-sync-upstream-<tag>, merge attempted
#   2. resolve any conflicts (docs/karinai-patches.md maps intentional divergence;
#      a conflict hunk that isn't part of a documented KarinAI patch → upstream wins)
#   3. git commit, then: sync_upstream.sh gates
#   4. push + open PR "sync: upstream <tag>" → CI → user-gated merge
#   5. after merge: stage deploy + Tier-2 smokes, then update karinai-patches.md
#      if any KarinAI patch changed shape during resolution.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM_REMOTE="upstream"
WORKTREES_DIR="$(dirname "$REPO_ROOT")/worktrees"

die() { echo "ERROR: $*" >&2; exit 1; }

cmd_start() {
    cd "$REPO_ROOT"
    git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1 \
        || die "no '$UPSTREAM_REMOTE' remote; expected https://github.com/NousResearch/hermes-agent.git"

    echo "==> Fetching $UPSTREAM_REMOTE (tags)..."
    git fetch "$UPSTREAM_REMOTE" --tags

    local tag="${1:-}"
    if [ -z "$tag" ]; then
        # Latest upstream release tag by version sort (vYYYY.M.D...).
        tag=$(git tag --list 'v*' --sort=-v:refname | head -1)
        [ -n "$tag" ] || die "no v* tags found after fetch"
    fi
    git rev-parse -q --verify "$tag^{commit}" >/dev/null || die "tag '$tag' not found"

    local branch="sync/upstream-$tag"
    local wt="$WORKTREES_DIR/agent-sync-upstream-$tag"
    [ -e "$wt" ] && die "worktree $wt already exists"

    echo "==> Behind upstream: $(git rev-list --count origin/main.."$tag" 2>/dev/null || echo '?') commits (tag $tag)"
    echo "==> Creating worktree $wt on $branch (off origin/main)..."
    git fetch origin --quiet
    git worktree add -b "$branch" "$wt" origin/main

    cd "$wt"
    echo "==> Merging $tag..."
    if git merge --no-edit "$tag"; then
        echo "==> Clean merge. Next: cd $wt && $0 gates"
    else
        echo ""
        echo "==> CONFLICTS — resolve using docs/karinai-patches.md as the map:"
        git diff --name-only --diff-filter=U | sed 's/^/     /'
        echo "    Then: git add -A && git commit --no-edit && $0 gates"
    fi
}

cmd_gates() {
    # Run from inside a sync worktree (or the main checkout).
    local root
    root="$(git rev-parse --show-toplevel)"
    cd "$root"
    git diff --name-only --diff-filter=U | grep -q . && die "unresolved conflicts remain"

    if [ ! -x .venv/bin/python ]; then
        echo "==> Creating venv (uv sync --extra dev --extra messaging)..."
        uv sync --extra dev --extra messaging
    fi

    echo "==> Gate 1/4: KarinAI test suite"
    .venv/bin/python -m pytest tests/karinai/ -q

    echo "==> Gate 2/4: branding audit"
    .venv/bin/python karinai/scripts/check_branding.py

    echo "==> Gate 3/4: patched-area tests (per-file, CI-style isolation)"
    local patched_tests=(
        tests/gateway/test_api_server_attachments.py
        tests/tools/test_register_artifact_tool.py
        tests/tools/test_karinai_app_tools.py
        tests/tools/test_registry.py
        tests/hermes_cli/test_auth_codex_provider.py
        tests/hermes_cli/test_auth_codex_self_heal.py
        tests/hermes_cli/test_runtime_provider_resolution.py
        tests/test_toolsets.py
        tests/agent/test_system_prompt.py
        tests/tools/test_approval.py
    )
    local f
    for f in "${patched_tests[@]}"; do
        [ -f "$f" ] || { echo "     (skip missing $f)"; continue; }
        .venv/bin/python -m pytest "$f" -q || die "gate failed: $f"
    done

    echo "==> Gate 4/4: import smoke of patched modules"
    .venv/bin/python -c "import gateway.platforms.api_server, gateway.session_context, karinai.runtime.start_managed, karinai.runtime.image_gateway_provider, tools.register_artifact_tool, tools.image_generation_tool, tools.approval, tools.registry, toolsets; print('imports OK')"

    echo ""
    echo "==> All gates green. Next:"
    echo "    git push -u origin \$(git branch --show-current)"
    echo "    gh pr create --base main --title 'sync: upstream <tag>' (merge is user-gated)"
}

case "${1:-}" in
    start) shift; cmd_start "$@" ;;
    gates) shift; cmd_gates "$@" ;;
    *) sed -n '2,20p' "$0"; exit 1 ;;
esac
