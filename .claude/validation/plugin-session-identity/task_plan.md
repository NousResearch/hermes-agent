# Plugin Session Identity Fix

## Goal
Implement and locally validate the confirmed plugin hook session identity concurrency correction without altering callback-global timeout suppression or fail-closed pre-tool behavior.

## Phases
- [x] Phase 1: Read requirements, RCA, instructions, and repository state
- [x] Phase 2: Map current gate behavior and call sites
- [x] Phase 3: Add deterministic regression tests and preserve literal RED evidence
- [x] Phase 4: Apply the smallest production correction and prove GREEN
- [x] Phase 5: Run focused, neighboring, repeated concurrency, lint, format, syntax, and diff checks
- [x] Phase 6: Complete additional independent verification rounds required by user policy
- [x] Phase 7: Write report, review diff, commit coherent changes, and verify clean state

## Next Step
Controller publication after independent review; no further local implementation action.

## Decisions Made
| Decision | Rationale |
|---|---|
| Use existing PluginManager public hook dispatch in tests | Exercises real gate behavior rather than implementation helpers |
| Synchronize callbacks with threading.Event | Removes scheduler/sleep dependence from overlap and dedupe assertions |
| Keep production delta to identity fallback unless a call-site test proves otherwise | Brief requires smallest correction |

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| Canonical runner found no pytest-capable venv | 1 | Locate an existing dev venv and pass it through `HERMES_PYTHON`; do not install into live runtime |
