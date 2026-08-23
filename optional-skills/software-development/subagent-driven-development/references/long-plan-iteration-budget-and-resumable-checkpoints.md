# Long-Plan Iteration Budgets and Resumable Checkpoints

Use this pattern for implementation programs with many phases, multiple repositories/worktrees, receipt capture, and independent review gates.

## Problem

A controller can exhaust its tool-call iteration budget while still doing legitimate prerequisite work: loading skills, reading a long plan, auditing dirty trees, inspecting receipts, dispatching writers/reviewers, and verifying source state. If no durable phase checkpoint exists, the next session must reconstruct too much context and may accidentally overstate closure or duplicate active agent work.

## Required controller pattern

1. **Estimate orchestration cost before dispatch.** Count expected controller calls for plan reads, baseline inspection, worktree setup, each writer batch, controller verification, review, integration, and final gates. Treat this as a finite budget, not an open-ended session.
2. **Create the continuation ledger first.** Before semantic edits, establish a durable run root outside Git containing:
   - source binding and fixed bases;
   - dirty-path ownership and rollback snapshots;
   - worktree locks and serialized controller paths;
   - phase status (`pending`, `in_progress`, `blocked`, `verified`);
   - exact outstanding agents/processes;
   - receipt index and known baseline failures;
   - next safe action.
3. **Checkpoint after every gate, not only at the end.** A phase is `verified` only after the controller has checked the artifact and any required independent review has returned. “Writer dispatched,” “tests recorded,” and “review pending” are distinct states.
4. **Keep independent reviewer dependencies explicit.** If a phase says code cannot start before an architecture verdict, do not overlap semantic implementation with that pending verdict. Read-only readiness audits may run in parallel; writer work may not cross the gate.
5. **Reserve the final 15–20% of controller iterations for closure.** Use it to inspect late agent results, verify Git state, update the ledger, and produce an evidence-backed handoff. Do not spend the reserve on optional broad reads.
6. **When the limit approaches, stop at a safe checkpoint.** Do not start another writer. Record what is complete, what is merely dispatched, what is blocked, and which side effects were not performed.

## Efficient reading and delegation

- Read a long plan once in large, non-overlapping chunks and extract all task boundaries immediately.
- Batch independent file/receipt reads.
- Delegate one writer per non-overlapping phase or owner scope; run read-only audits in parallel only when they do not authorize premature implementation.
- Avoid separate tool calls for every receipt when a deterministic script can summarize status, digests, and source-change flags into one bounded artifact.
- Never poll background subagents that return automatically. Continue only with non-overlapping controller work.

## Truthful phase vocabulary

- **Established:** prerequisite artifact exists and has basic structural verification.
- **Recorded:** command or test receipt exists, regardless of pass/fail.
- **Reviewed:** independent reviewer returned a verdict.
- **Verified:** controller reproduced the required gate against the current source.
- **Integrated:** reviewed changes were imported and post-integration gates passed.
- **Activated:** built/installed/fresh-process/current-live proof exists where applicable.

Do not collapse these into “done.” A failing baseline can still satisfy a baseline-recording task, but it cannot satisfy a later correctness or release gate.

## Handoff minimum

A forced-stop handoff must name:

- exact run root and worktrees;
- fixed commits and dirty-state boundaries;
- verified artifacts and their digests when available;
- baseline pass/fail/blocked results;
- agents still outstanding;
- phases not started;
- commits, pushes, installs, restarts, publication, and deployment actually performed—or explicitly not performed;
- the next gate that must be resolved before semantic work resumes.
