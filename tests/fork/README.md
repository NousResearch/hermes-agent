# `tests/fork/` — fork-invariant registry

This directory is the **presence net** for fork-required features: behavior and
modules this fork adds on top of NousResearch/hermes-agent that an upstream sync
must not silently drop.

## The problem it solves

Syncing (`scripts/sync-upstream.sh` + the nightly `hermes update`) merges
`upstream/main`. `git` only flags **overlapping-line** conflicts. It does *not*
flag:

- an upstream refactor that **deletes** a module the fork depends on,
- an upstream change that **replaces** the fork's variant of a feature on
  non-overlapping lines,
- a clean re-introduction of an upstream default the fork deliberately reverted.

Any of these can land in a "clean" merge and regress the live fleet. The
motivating incident: a sync swapped the fork's deterministic `web_extract` for
upstream's, with no conflict and no guard firing.

## The two nets (defense in depth)

1. **Absence net** — `scripts/sync-upstream.sh::check_fork_invariants`. A
   grep run at *sync time*: a reverted upstream symbol must NOT reappear. Fails
   the auto-merge fast, before the PR is even green.
2. **Presence net** — this suite. Each fork feature MUST still exist after the
   merge. Because `testpaths = ["tests"]`, it runs in the CI aggregate that
   gates auto-merge (`.github/workflows/ci.yml::all-checks-pass`). A sync PR
   that removes a fork feature goes RED and cannot auto-merge — it hands off for
   manual re-apply instead of regressing the fleet.

## Maintaining it

- **Adding** a fork feature a future sync could silently undo → add an entry to
  `test_fork_features_present.py` (a module to `FORK_MODULES`, or a structural
  assertion).
- **Retiring** a fork delta on purpose (upstream now covers it) → delete its
  entry in the **same change** that removes the delta. Every entry carries a
  `drop_when` note so a future syncer can tell a load-bearing guard from an
  obsolete one.

A red test here means: *an upstream merge removed something the fleet depends
on — re-apply it before merging*, not *the fork is broken*.

See also the roadmap memory `project-upstream-sync-hardening`.
