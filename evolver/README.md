# Harness Evolver — Phase 0: gate calibration

Offline tooling for evolving Hermes' own agent scaffold. This package is
**not** part of the shipped product (`pyproject.toml` deliberately omits it
from the wheel): it runs offline, never in the hot path, and proposes
changes that a human merges.

Full plan: issues #110671–#110676 (this repo).

## Trigger and cost

Nothing here runs in the hot path or on any schedule. The evolver is
invoked manually, offline:

- `collect.py` — reads JSON files, milliseconds.
- `validity_gate` — one scratch worktree + `git apply --check` + an AST
  parse per touched file. Seconds.
- `activation_gate` — two runs of one regression test file via
  `scripts/run_tests.sh`. Minutes, bounded by the test itself.
- `credit_gate` — 10k bootstrap resamples over ≤ dozens of pairs.
  Sub-second, deterministic (seeded).
- `calibrate.py` — ~15 single-file test runs across scratch worktrees
  (each capped at 300 s). Slow on purpose: it is the kill criterion.

A patch that never reaches a human merge gate costs nothing but compute;
a patch that does reach it costs a review.

## Phase 0 scope (this PR)

1. `archive.py` — pathology-keyed failure archive. Records are
   `(task, trace, failure_class)` with OpenTelemetry-flavored spans,
   append-only JSONL, versioned schema (`evolver.SCHEMA_VERSION`).
2. `collect.py` — read-only ingestion of the factory's
   `fix_results/*.json` records into the archive. Reads only; the factory
   is never modified.
3. `gates.py` — the three credit gates as testable code:
   - **validity**: non-empty diff, `git apply --check` on the base tree,
     touched `.py` files still parse (checked in a scratch worktree).
   - **activation**: the regression test is red on base and green on
     patched. Both directions required.
   - **credit**: paired bootstrap CI on the mean paired delta; credit iff
     the lower bound is above zero. Fails closed on < 5 pairs (a thin
     measurement is a guess, not evidence).
4. `calibrate.py` — the kill criterion, measured. Runs validity +
   activation over 5 recent human-authored merged fixes (known-good) and
   synthetic bad variants (empty diff, syntax-breaking diff, behaviorally
   no-op diff). Exit 0 iff every gate expectation holds.
5. `SEALED_BATTERY.md` — what the sealed battery is and how it rotates.
   No sealed tasks live in this repo, ever.

## Running calibration

```bash
python -m evolver.calibrate \
  --repo ~/workspace/hermes/hermes-agent \
  --venv ~/workspace/hermes/hermes-agent/.venv \
  --workdir /tmp/evolver-calib \
  --report /tmp/evolver-calib/report.json
```

Each case builds scratch worktrees at the base/merge commits and runs the
regression test via `scripts/run_tests.sh` (base gets the merge's test
overlaid, the standard red-on-base proof). Expect ~15 single-file test
runs; it takes a while — that is the point. If calibration fails, the
methodology is broken: do not proceed to any evolver.

## What comes next (not this PR)

- `proposer.py` — strong model → diff-scoped patch proposals (Phase 1)
- `champions.py` — per-pathology champion archive (Phase 1)
- `adapter.py` — signed evolvability manifest enforcement (Phase 1)
- `rollout.py` — signed bundles, shadow canary, one-command rollback (Phase 2)
