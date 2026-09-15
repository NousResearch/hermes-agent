# Sealed battery — design (Phase 0)

The credit gate is only as honest as the battery it measures on. This
document defines the sealed battery; **no sealed tasks are stored in this
repo** (a battery the evolver can read is a battery the evolver can game —
see the sealed-battery poisoning incident in the SOTA sweep).

## Composition

- The open `evals/toolperf_abeval` trap tasks (evolver-visible), PLUS
- a held-out slice of recent real GitHub issues: genuine bug reports with
  reproduction steps, frozen per quarter, curated by a human, evolver-blind.

Never the agent's own evals. Trusting-trust rots the bar.

## Access: scoring API, not task content

The evolver never observes sealed tasks. Credit runs go through a scoring
API: `(harness_version, task_id)` in, `(pass/fail, cost)` out. The task
bodies live outside the repo, served by infrastructure the evolver cannot
read. A patch that improves the open battery but not the sealed battery is
overfit, not improvement.

## Rotation policy

- Frozen per quarter; rotated quarterly by a human curator.
- A new slice is drawn from recent issues each quarter; the old slice is
  retired to the open set (it is no longer a secret once replaced).
- Any evidence of battery gaming (lift on open, flat on sealed, twice in a
  row) triggers an out-of-cycle rotation and an incident note.

## Pre-registration

Before each credit run, the battery version, the paired metric, and the
decision threshold (`ci_lo > 0` at `alpha = 0.05`) are recorded. Changing
the battery or the threshold after seeing the numbers invalidates the run.
