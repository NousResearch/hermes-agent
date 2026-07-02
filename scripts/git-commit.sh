#!/usr/bin/env bash
# ==========================================================
# git-commit.sh -- Platform repository policy gateway
# ==========================================================
#
# Every commit in this repository flows through this script.
# Direct `git commit` is unsupported.
#
# This gateway enforces Platform repository policy -- rules that
# apply to every commit regardless of which tool (Hermes, CLI,
# IDE, CI) created it. None of these rules depend on Hermes.
#
#   scripts/git-commit.sh "commit message"
#   scripts/git-commit.sh -m "commit message"
#   scripts/git-commit.sh -am "commit message"
#
# All arguments pass through to `git commit` unchanged.
#
# -- Policy Pipeline --------------------------------------
#
#   Phase 0 -- Pre-commit policy checks
#     Rules that run BEFORE the commit is created.
#     Must complete in under 30 seconds (cumulative).
#     Any failure blocks the commit.
#
#     -> conventional commit format
#     -> binary/large-file guard
#     -> syntax check (shellcheck, yamllint, ruff)
#     -> formatting (auto-fix or reject)
#     -> affected unit tests
#     -> architecture guardrails
#
#   Phase 1 -- Commit
#     -> set deterministic Author/Committer identities
#     -> execute `git commit`
#
#   Phase 2 -- Post-commit actions
#     Rules that run AFTER the commit is created.
#     Failures are reported but do not roll back.
#
#     -> identity verification
#     -> optional auto-push
#     -> metrics / release note generation
#
# -- The 30-Second Rule -----------------------------------
#
# Phase 0 is a gate, not a test suite. Full benchmarks,
# integration tests, and multi-minute verification belong in
# CI or dedicated commands (`hermes verify`, `hermes benchmark`).
# If pre-commit collectively exceeds 30 seconds, developers
# will bypass it. Keep the gate fast; put everything else in CI.
#
# -- Adding Policy ----------------------------------------
#
# Drop an executable script into scripts/hooks/pre-commit.d/
# or scripts/hooks/post-commit.d/. They run in alphabetical
# order. Any pre-commit hook that exits non-zero blocks the
# commit. Post-commit hook failures are warnings only.
#
# These hooks enforce Platform repository policy.
# They do not depend on Hermes. They apply to any project
# that adopts this gateway.
#
# ==========================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/scripts/hooks"

# ---------------------------------------------------------
# Phase 0 -- Pre-commit policy checks
# ---------------------------------------------------------
run_hooks() {
    local hook_dir="$1"
    local phase_name="$2"

    if [ -d "$hook_dir" ]; then
        for hook in "$hook_dir"/*; do
            if [ -x "$hook" ]; then
                echo "  [$phase_name] $(basename "$hook")"
                "$hook" || {
                    echo >&2 "  [$phase_name] FAILED -- commit blocked"
                    exit 1
                }
            fi
        done
    fi
}

run_hooks "$HOOKS_DIR/pre-commit.d" "pre-commit"

# ---------------------------------------------------------
# Phase 1 -- Commit with deterministic identity
# ---------------------------------------------------------

# Author identity -- hardcoded, never environment-dependent.
# Committer identity -- from repo config (johnalencar-agent).
export GIT_AUTHOR_NAME="John P. Alencar"
export GIT_AUTHOR_EMAIL="johnalencar@hotmail.com"

git commit "$@"

# ---------------------------------------------------------
# Phase 2 -- Post-commit verification
# ---------------------------------------------------------

EXPECTED="Author: John P. Alencar <johnalencar@hotmail.com> | Committer: johnalencar-agent <johnalencar.agent@gmail.com>"
ACTUAL=$(git log --format='Author: %an <%ae> | Committer: %cn <%ce>' -1)

if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo >&2 "  IDENTITY VIOLATION -- expected: $EXPECTED"
    echo >&2 "                          actual:   $ACTUAL"
    exit 2
fi

echo "  identity: $ACTUAL"

run_hooks "$HOOKS_DIR/post-commit.d" "post-commit"
