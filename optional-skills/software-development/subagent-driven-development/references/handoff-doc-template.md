# Handoff Doc Template — `HOSTILE_AUDIT_FINDINGS_<project>_<date>.md`

When a multi-batch workstream lands partially — typically because parallel
agents conflicted on shared files, the React/TS agent produced unfixable
type errors, or session time ran out — write ONE handoff doc at the end
of the session. This is the user's "did it actually ship" answer.

The doc is the canonical answer. Commit it. Reference it from the final
session message.

## When to write this doc

- 3+ batches were dispatched (e.g. Batch A/B/C, or initial spec → agent
  revisions → controller fixes)
- Some items shipped, some deferred
- User asked for the work to be done in a finite time window and
  reality was less than that window

If everything in the original plan shipped cleanly, the
`hostile-auditor-style final summary` in the last commit message is
enough. No separate doc needed.

## Template

```markdown
# HOSTILE_AUDIT_FINDINGS_<PROJECT>_<YYYYMMDD>

Branch: `<branch-name>`
Base: `<commit-hash> <base-commit-subject>`
Commits: `<h1>` (<workstream-A>), `<h2>` (<workstream-B>), ...
Tests: **<one-line summary>**

## What this pass did

Group committed work by workstream. For each item, cite file:line.

### Batch A (commit <h>) — <one-line summary>
The original items this batch closed, in priority order.

| ID | File:line | Fix |
|---|---|---|
| <id> | `path/to/file.rs:NN-MM` | <one-line> |

(Repeat table per batch.)

### Batch B (commit <h>) — <one-line summary>
...

## What was NOT done (and why)

Be brutally honest. Three buckets:

### Deferred — concrete reason
The item was attempted, failed for a specific reason, captured as
follow-up. Include the failure mode (e.g. "agent's selector refactor
was 41 TS errors") so the next session can scope around it.

### Deferred — capability gap
An item the audit listed that the agent/controller couldn't do safely
in the time available. E.g. "embedder unification requires rewriting
the FastEmbed adapter as async; deferred because A1 makes the dangerous
path unreachable."

### Deferred — session time
Items the audit listed that are real work but the session ran out
before landing them. Be explicit: don't hide behind "future work."

## Receipts

Verification command + actual output, one per line. Run all of these
fresh right before writing the doc; do not paste stale results.

```
cargo fmt --all -- --check
(no output = clean)

cargo test --features <feature> --lib --no-fail-fast
test result: ok. 170 passed; 0 failed; 1 ignored

cargo test commands::chat
test result: ok. 3 passed

cargo test providers::tests
test result: ok. 14 passed

npm run build
✓ built in 11.17s

npm test
"status": "pass", 12 checks

python3 validation/validate_<gate>.py .
PASS: ...
```

## Risk assessment of the deferred work

One paragraph per major deferred bucket:

> The deferred React-perf items (B6/D6/B5) are a UX quality issue (jank
> on long responses, no virtualization) but do not affect correctness.
> A user can have a 100% working chat with 200+ messages; the chat just
> doesn't scroll as smoothly as it could. The other deferred items
> (C2, C4, C5-FIX, C6-CONFIG, E2, F4) are reliability/quality items
> that affect edge cases (stuck jobs, model switch, very slow providers)
> but not the common path.

## Hostile-auditor handoff

Last section. One paragraph:

> The Rust performance half is fully done and tested. The UX polish half
> is half-done (cmd palette + onboarding, but not virtualization, not
> selectors). The reliability half is half-done (eager warmup, error
> boundary, fatal dialog, but not job-level cancel, not embedder
> unification). All 5 AGENTS.md mandatory gates pass; receipt state is
> clean. To resume: `git checkout <branch>` and pick up from the
> deferred list above.
```

## Anti-patterns to avoid

- **"Shipped everything"** when you shipped 25/44 items. The user
  always finds the gap. Explicit deferral is the only honest answer.
- **Pasting verification output you didn't actually run.** Re-run
  every command in the receipts section right before writing the doc.
  "I ran these 30 minutes ago" is not a receipt.
- **Hiding deferrals under "future work" or "next session" without
  naming the failure mode.** "Fix React performance" is unactionable.
  "Re-apply useShallow selectors as a single-file task that matches
  the StatusBar.tsx:21-34 pattern" is actionable.
- **Omitting the risk assessment.** Without it, the user can't
  prioritize the deferred work for their own schedule.
- **Writing the doc in retrospect.** Write it before final verification,
  then update the receipts with actual output. This prevents the
  "I forgot to run X" failure mode.

## Why this matters

A hostile-auditor handoff is the difference between a session that
"fixed everything" and a session that fixed 25 things honestly and
left 19 documented for next time. The user's mental model after a
complex multi-agent session is built from this doc, not from the
agent summary at the end. If the doc is missing or wrong, the user
loses 30 minutes of next-session orientation.

Worse: if the doc says "fixed everything" and the user takes it at
face value, they ship a release where the unfixed items are
production-visible. The hostile-auditor pattern catches this before
the doc is written.
