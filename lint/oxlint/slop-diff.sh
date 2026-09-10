#!/usr/bin/env bash
# Run the anti-slop ratchet on the files this branch touches. Findings at or
# below lint/oxlint/slop-baseline.json AS COMMITTED AT THE MERGE BASE are
# allowed (the PR's own copy is never consulted, so it cannot raise its own
# allowance); only net-new findings are reported. Diff base: merge-base with origin/main (override with SLOP_BASE,
# e.g. SLOP_BASE=origin/develop).
set -euo pipefail

exec node "$(git rev-parse --show-toplevel)/lint/oxlint/slop-ratchet.mjs" diff
