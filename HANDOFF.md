# Hermes Execution Router API — Handoff

## Current objective

Complete accepted T001 only: establish the local source baseline with preserved project and upstream ancestry, without changing Hermes runtime behavior.

## Resume sequence

1. Commit the accepted planning/project-control snapshot with neutral existing repository identity.
2. Fetch complete ancestry of exact upstream commit `110736c0bc9fd249f1ce7f7ca5d353040f640be6` without retaining a remote.
3. Verify non-shallow state, complete objects and exact fetched identity.
4. Merge unrelated histories without auto-commit.
5. Require exactly three add/add conflicts: `.gitignore`, `AGENTS.md`, `README.md`.
6. Keep upstream README, combine both AGENTS rule sets, and union ignore rules.
7. Verify no upstream runtime/source/test/package delta outside the accepted planning/conflict allowlist.
8. Create the topology merge commit and perform exact clean/ancestry/fsck/no-remote readback.

## Stop conditions

Stop for reconciliation on incomplete ancestry, any additional conflict, any source-semantic delta, ambiguous path, secret, retained remote/tag/submodule/nested repository or failed verification.

## Not authorized

T002 or later work, API source implementation, push, publication, installation, profile/gateway/runtime changes, consumer work, pilot and LIVE.
