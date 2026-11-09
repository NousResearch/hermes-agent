#!/usr/bin/env bash
# Fail when the committed icons differ from what scripts/generate-icons.mjs renders.
#
# On a GitHub Actions runner the whole checkout must be clean afterwards, so the step also
# catches any file an earlier step of the job left behind. Anywhere else the tree is first
# snapshotted into a scratch index and only changes against that snapshot count, so
# uncommitted work is not reported as icon drift and the real index is left alone. A failure
# leaves the regenerated icons in the tree.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Build identity recolours the desktop icons; the committed ones carry none.
export HERMES_PAYLOAD_TAG='' HERMES_BUILD_COMMIT=''
if [ "${GITHUB_ACTIONS:-}" = true ]; then
    changed() { git status --porcelain --untracked-files=all; }
else
    index="$(mktemp)"
    trap 'rm -f "$index"' EXIT
    cp "$(git rev-parse --git-path index)" "$index"
    export GIT_INDEX_FILE="$index"
    git add -A
    changed() { git ls-files --others --exclude-standard; }
fi
node scripts/generate-icons.mjs
node scripts/generate-icons.mjs --check
if ! git diff --exit-code --stat || [ -n "$(changed)" ]; then
    changed
    echo "::error::Committed icons are stale. Run 'node scripts/generate-icons.mjs' and commit the result."
    exit 1
fi
