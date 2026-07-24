# Preserve pre-advanced cron slots across late completion

## Problem

The in-process scheduler advances a recurring job before dispatch for
at-most-once crash safety. `mark_job_run()` then recomputed `next_run_at` from
the completion timestamp. For a `* * * * *` job claimed at `:54` and completed
after the next `:00` boundary, the recomputation skipped the already reserved
minute. A one-minute watchdog therefore ran every two minutes.

## Change

Pre-dispatch paths record the exact recurring slot they reserve.
`mark_job_run()` retains that slot only when it still matches the job's current
`next_run_at`, then clears the marker. Direct completion calls and changed
schedules continue to compute from completion time.

## Safety

The change preserves existing at-most-once crash protection. It does not add
or invoke any order-submission path.

## Verification

- Focused regression reproduces a `:54` dispatch that completes at `:01` and
  asserts the `:00` slot is retained.
- Focused test passes after the fix.
