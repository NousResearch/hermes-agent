#!/usr/bin/env bash
# Run CI step scripts outside GitHub Actions, as their workflow steps run them.
#
#   scripts/ci/local_step.sh 'scripts/ci/check_common_ancestor.py' 'scripts/ci/scan_supply_chain.py patterns'
#
# Each argument is one step: a Python script and its arguments, split on whitespace and run
# with the PM toolchain Python first on PATH (the bare `python`/`python3` a step gets before
# setup-pm). The steps publish review statuses and reports through the runner's files
# (scripts/ci/review_status.py), so each step gets its own throwaway RUNNER_TEMP,
# GITHUB_OUTPUT and GITHUB_STEP_SUMMARY, and the pull-request commits the supply-chain scan
# diffs are the merge base with origin/main and HEAD. A step fails when it exits non-zero or
# reports `found=true`, which its workflow turns into a failing step of its own. Every step
# runs; the exit status is non-zero when any failed.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/_activation.sh
. "$repo/scripts/_activation.sh"
python="$(hermes_bootstrap_python "$repo")" || {
    echo "no bootstrap Python found; run setup-hermes.sh" >&2
    exit 1
}
bin="${python%[/\\]*}"
if command -v cygpath >/dev/null 2>&1; then bin="$(cygpath -u "$bin")"; fi
# verify_uv_lock.py runs `python -m pm.build_env`, as setup-pm's toolchain Python.
PATH="$bin:$PATH"
# A step on the runner starts with no Python environment of its own.
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
cd "$repo"
git rev-parse -q --verify origin/main >/dev/null || {
    echo "no origin/main; run git fetch origin main" >&2
    exit 1
}
# No merge base: check_common_ancestor.py fails and says why; the scans then diff nothing.
_HERMES_PR_BASE_SHA="$(git merge-base origin/main HEAD || :)"
_HERMES_PR_HEAD_SHA="$(git rev-parse HEAD)"
export _HERMES_PR_BASE_SHA _HERMES_PR_HEAD_SHA

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
failed=()
for step in "$@"; do
    dir="$scratch/step"
    rm -rf "$dir"
    mkdir -p "$dir/runner"
    export RUNNER_TEMP="$dir/runner" GITHUB_OUTPUT="$dir/output" GITHUB_STEP_SUMMARY="$dir/summary"
    : > "$GITHUB_OUTPUT"
    echo "== $step"
    status=0
    # shellcheck disable=SC2086  # a step is a script and its whitespace-separated arguments
    "$python" $step || status=$?
    if [ -s "$GITHUB_STEP_SUMMARY" ]; then cat "$GITHUB_STEP_SUMMARY"; fi
    if grep -qx 'found=true' "$GITHUB_OUTPUT"; then
        for report in "$RUNNER_TEMP"/*; do
            if [ -f "$report" ]; then cat "$report"; fi
        done
        status=1
    fi
    [ "$status" = 0 ] || failed+=("$step")
done
if [ "${#failed[@]}" -gt 0 ]; then
    printf 'failed: %s\n' "${failed[@]}" >&2
    exit 1
fi
