---
name: hillclimb
description: "Optimize one metric via frozen harness and decision log."
version: 1.0.0
author: "Emmanuel Ketcha (@ketchalegend) + Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [optimization, metrics, experiments, decision-log, benchmarking, iteration]
    category: software-development
    related_skills: [systematic-debugging, spike, test-driven-development, requesting-code-review, subagent-driven-development]
---

# Harness Lifecycle Reference

State machine for the measurement harness. The harness defines **what** is measured and **how** it is measured. Changing the harness invalidates all prior measurements.

## States
- **DRAFT** - You are still deciding what to measure or how to measure it. No baseline recorded.
- **MEASURED** - You have run the harness multiple times and observed a stable median. The harness prints a single number reliably.
- **BASELINED** - `sample_metric.py baseline` has been run. The median is recorded in `.hillclimb/baseline.json` with `frozen: true` and a `harness_id`. The harness is now immutable for the duration of the loop.
- **INVALIDATED** - The harness was edited after BASELINED. All prior rows in `decision.tsv` are now invalid. The skill will reject further `compare` calls with exit code 3 (harness changed). A legitimate INVALIDATION must be recorded as a decision row before re-baselining.
- **RE-BASELINED** - After a recorded INVALIDATION, you may run `baseline` again. This creates a new frozen baseline with a new `harness_id`, which opens a new epoch of comparable numbers. The existing log is NOT reset: earlier rows keep their old `harness_id` and are reported by `verify` as `stale-harness` from then on, which is exactly what should happen - they were measured a different way.

## Transition Rules
- **DRAFT -> MEASURED**: Run harness repeatedly (`--samples N`). Verify the output is a single number and repeated runs land within an acceptable spread. No file is created yet.
- **MEASURED -> BASELINED**: Run `sample_metric.py baseline`. This records the median, the sample count and the extraction spec, computes `harness_id` (the first 12 hex characters of the SHA-256 of the whitespace-normalized harness command), and writes `.hillclimb/baseline.json`. Recording the extraction spec is what lets `compare` read the number the same way the baseline did instead of guessing.
- **BASELINED -> INVALIDATED**: Any edit to the harness command (even whitespace changes) makes `compare` exit 3. The only legitimate path is to record the harness edit as a decision row (`tests: none`, `before: na`, `after: na`) before re-baselining.
- **INVALIDATED -> RE-BASELINED**: After recording the harness edit, run `baseline` again. This creates a new frozen baseline.

## Why `harness_id` Matters
The `harness_id` is the first 12 hex characters of SHA256 of the whitespace-normalized harness command. It serves as a fingerprint:
- Changing the harness changes the ID -> `compare` refuses (exit 3).
- Silent re-baselining without recording the harness edit is impossible because the ID would differ.
- It prevents cheating by editing the harness to improve the metric without re-recording the baseline.

## Legitimate Re-Baseline Examples
1. **New measurement target** - You decide to measure CI duration instead of test suite time. Record: "Changed measurement target from test suite time to CI duration" (`tests: none`). Then re-baseline.
2. **Harness bug fix** - The harness script had a bug that caused occasional failures. Fix the bug, record the fix, then re-baseline.
3. **Environment change** - CI runner upgraded Python version; you re-baseline to capture the new baseline under the new environment.

## Cheating Re-Baseline Examples (and why they're caught)
1. **Editing harness because change didn't show a win** - You tweak the command to make the metric look better. The `harness_id` changes, so `compare` exits 3. You can't silently re-baseline.
2. **Loosening the metric** - You change from "wall-clock seconds" to "CPU seconds" to show improvement. The harness command changes -> ID mismatch -> exit 3.
3. **Changing the sample count after seeing a bad result** - You raise `--samples` after a run disappointed you, hoping the median improves. Be aware this one is NOT caught by the id: `harness_id` fingerprints the harness command only, so `--samples` is outside it. `baseline.json` records the sample count the baseline was taken with - hold it fixed for the duration of the loop, and treat a change of sample count the same as a change of harness (record it, then re-baseline).
4. **Swapping the command silently** - You replace the real command with a dummy that always returns 0. The harness command changes -> ID mismatch -> exit 3.

## Mechanical Guard
`sample_metric.py compare` computes the current harness command's `harness_id` and requires it to equal the frozen baseline's. On a mismatch it prints `harness-id mismatch; baseline invalidated` and exits 3. Silent re-baselining is therefore impossible: the mismatch has to be resolved by taking a new baseline deliberately, which is recorded.

## Usage Workflow
1. Start in DRAFT. Build harness, verify it prints a single number reliably.
2. Move to MEASURED by running it with sufficient samples to establish a stable spread.
3. Move to BASELINED with `sample_metric.py baseline`. Now the harness is frozen.
4. Make one change to the codebase (not the harness).
5. Run `sample_metric.py compare`. If it exits 0, you have an improvement; record it.
6. If you need to change the harness itself, first record that as a decision row (`tests: none`), then re-baseline.

The lifecycle ensures that every number compared is from the same, unchanged measurement method.