---
name: multi-agent-pr-convergence
description: "Converge multi-agent code work into a verified PR."
version: 0.1.0
author: Herin Yudha Pratama (hrnbld), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Multi-Agent, Pull-Request, Orchestration, Evidence]
    related_skills: [merge-reconciler, hermes-agent]
---

# Multi-Agent PR Convergence

Turn parallel agent work into one minimal, current-upstream pull request. This
skill borrows the useful shape of persistent agent runtimes—stable identities,
event-driven state, and durable terminals—without treating pane status or an
agent's self-report as completion evidence.

## When to Use

- A feature or PR uses multiple writers, reviewers, or QA agents.
- Work repeatedly rebases, overlaps upstream, changes scope, or stalls between
  candidate and review.
- Agent status says `done` or `idle`, but Git, tests, or PR state disagree.
- Do not use for a single low-risk edit with one writer and one test command.
- Use `merge-reconciler` when two admitted branches already have real merge
  conflicts; this skill prevents and converges the campaign around them.

## Prerequisites

- The canonical repository and target remote are known.
- Git authentication can fetch the target base; push authority is optional and
  must be stated separately from merge authority.
- The project test commands and independent checker roles are known.
- A durable runtime such as Herdr is optional. If used, read
  `references/herdr-lessons.md` before designing the control loop.

## Quick Reference

| State | Meaning | Required next action |
|---|---|---|
| `DISCOVERY` | Canonical repo/upstream behavior not proved | Fetch and classify requirements |
| `WRITING` | Exactly one writer owns one worktree | Implement only frozen scope |
| `FROZEN` | Clean candidate SHA exists | Stop writer; dispatch exact-SHA gates |
| `CHANGES_REQUIRED` | A checker proved a blocker | Issue one deduplicated repair packet |
| `READY_FOR_PR` | Independent gates agree on one SHA | Re-fetch base, simulate merge, open/update PR |
| `BLOCKED` | Authority, environment, or evidence is missing | Name one blocker and next owner |

Never use `idle`, `done`, process exit, or a pane label as a release state.

## Procedure

### 1. Prove the target before planning

1. Fetch the canonical remote with `terminal` and record the full base SHA.
2. Inspect current upstream for every requested behavior. Classify each as
   `already_upstream`, `partial`, `missing`, or `architecturally_blocked`.
3. Remove already-upstream scope before any writer starts. File true upstream
   regressions separately instead of rebuilding the same feature.
4. Record the canonical repo, base SHA, classifications, and excluded scope.
5. Repeat the upstream classification before **every candidate freeze**, not
   only before the PR. A long implementation can become redundant mid-flight.

Done when every requested behavior has evidence from current upstream and no
writer is implementing a duplicate.

### 2. Freeze authority and the task packet

1. Name exactly one writer for the repository. All reviewers and QA agents are
   read-only on the candidate worktree.
2. Record allowed paths, excluded paths, required tests, push authority, merge
   authority, and destructive-action authority as separate fields.
3. Assign an immutable `task_id`, worktree, branch, and report path. A useful
   packet contains:

```json
{
  "task_id": "feature-unique-id",
  "repo": "owner/repository",
  "base_sha": "full-remote-sha",
  "writer": "one-agent-id",
  "allowed_paths": ["feature/**", "tests/**"],
  "required_checks": ["focused", "full", "lint", "typecheck"],
  "push_authorized": true,
  "merge_authorized": false
}
```

4. Quarantine unknown-writer artifacts. Preserve them for provenance, but never
   include internal registers or forensic packets in the product PR.
5. Treat the packet as a writer lease: record writer identity, allowed paths,
   lease epoch, and last heartbeat. Before each mutation, compare worktree state
   with the last writer-owned snapshot. A foreign or unexplained delta revokes
   the lease and returns the task to `BLOCKED / authority-unproven`.
6. Freeze owner-approved scope as a versioned list. Scope changes require a new
   explicit owner decision and packet version; checker comments alone cannot
   silently expand or shrink the product request.

Done when one writer, one worktree, one base, and one authority packet are
unambiguous.

### 3. Dispatch once and observe by events

1. Prefer a tracked background process, durable task queue, or persistent agent
   runtime. Avoid long synchronous status calls that lock the session while the
   worker is still active.
2. Submit the task once. Require a transition to `working`; typed text alone is
   not dispatch proof.
3. React to state-change events or completion notifications. Do not busy-poll.
4. Every five minutes, reconcile only material state: live process, worktree
   dirtiness, latest SHA, missing evidence, owner, and time since the last
   material event.
5. If stale, send one bounded recovery instruction to the exact owner. Move
   heavy tests to a provisioned runner rather than weakening gates or deleting
   user data.

Done when the worker is active with acknowledged scope or the task is explicitly
`BLOCKED` with one named next owner.

### 4. Freeze one candidate

1. Require a clean worktree and full candidate SHA.
2. Compare the candidate against the recorded current base and account for every
   changed path.
3. Remove unrelated formatting, generated drift, internal evidence, and stale
   implementations of features that landed upstream.
4. Run author-side focused tests, but label them author evidence.
5. Stop writer mutation after publishing the SHA.

Done when one clean, minimal SHA is ready for independent review.

### 5. Run executable independent gates

Run independent lanes in parallel on the exact candidate SHA:

- **Safety/quality:** lint, typecheck, focused tests, full suite, import-graph
  regressions, and security checks.
- **Behavior/UX:** mounted or live behavior for the changed interaction.
- **Architecture/upstream:** generic boundary, current-upstream overlap, and
  minimal diff.

For stateful controllers, test the real lifecycle: create → snapshot → stage →
multi-consumer → finalize → retry/overlap. Include executable coverage for
identity rename/re-key, teardown/disband, same-name recreation, multi-responder
fan-out, attachment-only submission, and overlapping sends whenever those
transitions exist. Source regex tests may supplement but never replace mounted
executable tests. For every repaired bug class, prove a mutation or revert makes
the regression test fail.

Done when all required lanes independently report `PROVEN` on the same SHA, or
one deduplicated blocker packet is returned.

### 6. Repair in one bounded round

1. The coordinator deduplicates all checker findings into one repair packet.
2. Re-authorize the same writer for only those blockers.
3. Freeze a replacement SHA and carry evidence forward only when its baseline
   and impacted blobs are unchanged.
4. Dispatch checkers once on the replacement. Do not invent a new gate after the
   agreed contract is satisfied; newly discovered real bugs are evidence, not
   scope churn.

Done when the replacement is either `READY_FOR_PR` or honestly `BLOCKED`.

### 7. Re-read remote truth and open the PR

1. Fetch the remote again immediately before push or PR update.
2. Re-run the upstream-overlap classification and remove newly landed duplicate
   work.
3. Confirm merge-base freshness and simulate the merge. A green stale branch is
   not merge-ready.
4. Verify the PR head equals the independently reviewed SHA.
5. Open or update the PR with the minimal behavior summary and exact test
   evidence. Keep merge and deploy on their own owner gates.

Done when the PR URL, head SHA, CI state, review state, and merge authority are
reported from live GitHub data.

## Pitfalls

- **Multiple writers:** fastest locally, slowest at fan-in. Keep one writer per
  repository and parallelize read-only gates.
- **Synchronous inbox waits:** they hide liveness and create session locks. Use
  background work plus completion notifications.
- **Self-report as proof:** `281/281 pass` from the writer is useful, not
  independent closure.
- **Stale upstream:** re-check before design and before PR, not only at branch
  creation.
- **Regex-only lifecycle tests:** they miss controller identity, cleanup order,
  multi-consumer, and overlapping-send bugs.
- **Gate churn:** freeze the acceptance contract; add only proved blockers.
- **Environment waiver:** move to a capable runner instead of skipping lint,
  typecheck, or integration tests.
- **Pane status confusion:** persistent runtimes make execution fast, but
  `idle`, `done`, and terminal survival remain signals rather than Git/test/PR
  evidence.

## Verification

- Current canonical base SHA and requirement classification are recorded.
- One writer and one clean candidate SHA exist.
- Every changed path is in scope; no internal evidence enters the product diff.
- Independent executable gates agree on the exact SHA.
- Remote base and PR head are re-read immediately before the PR action.
- Final output includes PR URL, head SHA, CI status, residual risk, and explicit
  merge/deploy authority.
