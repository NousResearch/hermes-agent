---
name: hermes-code-evolution
description: "Use for frozen, review-gated autonomous code improvement."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, windows]
metadata:
  hermes:
    tags: [code-evolution, kanban, worktree, verification, review]
    related_skills: [test-driven-development, requesting-code-review]
---

# Hermes Code Evolution

Run one evidence-backed code improvement inside a frozen contract. This skill
never grants permission to edit a live checkout, rewrite Git history, merge,
push, deploy, or approve your own work.

## Launch Mode (Normal Conversation)

When `$HERMES_KANBAN_TASK` is not set, this skill launches a campaign; it does
not edit source code in the current conversation.

Campaign execution is limited to Linux child-subreaper containment and Windows
Job Objects. On macOS or another POSIX host, use `--dry-run` only; launch must
fail closed before task creation.

1. Confirm the request contains one concrete objective, falsifiable evidence
   (a reproduced failure, failing test, or measured regression), and a
   measurable success metric. Do not turn vague cleanup preferences into
   evidence.
2. Resolve the Git repository, exact allowed files/directories, and immutable
   no-shell quality-gate commands from the user's request and the live repo.
3. Inspect available Projects with `hermes project list` and profiles with
   `hermes profile list`. The selected Project's primary repository must equal
   `--repo`; the implementer and reviewer must both exist and be different.
   Retrieve what you can before asking the user for a missing decision.
4. Ensure the source repository is clean. Preserve any user changes; do not
   stash, reset, commit, or discard them to make the launcher accept the repo.
5. Run `hermes kanban improve ... --dry-run --json` first and inspect the exact
   base identity, path set, gates, budgets, and reviewer it prints.
6. If the frozen plan matches the authorized request, run the same command
   without `--dry-run`. Report the durable task id and leave execution to the
   Kanban dispatcher.

Example shape (replace every placeholder with verified values):

```bash
hermes kanban improve "<objective>" \
  --evidence "<reproduced evidence>" \
  --success-metric "<measurable pass condition>" \
  --repo "/absolute/repo" \
  --project "<existing-project-id-or-slug>" \
  --assignee "<implementer-profile>" \
  --reviewer "<different-reviewer-profile>" \
  --allow "<relative/path>" \
  --gate "<executable arg1 arg2>" \
  --dry-run --json
```

Do not use shell operators inside `--gate`; wrap compound checks in a reviewed
script and freeze that script invocation instead. Do not launch from a dirty
repo or invent evidence just to keep an autonomous loop busy.

## Required Context

This skill is valid only when all of these are present on the current Kanban
task:

- `workspace_kind=worktree`;
- a first-class Project link whose primary path owns that worktree;
- `code-evolution-contract.json` attachment;
- `code-evolution-verifier.py` attachment;
- an exact base commit and tree in the contract;
- an explicit measurable success metric;
- at least one allowed path and one frozen quality gate;
- different implementation and reviewer profiles.

If any item is absent, altered, unreadable, or ambiguous, call `kanban_block`
with the exact problem and stop. Do not reconstruct or replace missing evidence.

## Orient Before Acting

1. Call `kanban_show()` and read the full body, comments, attachments, prior
   attempts, and current status.
2. Work only inside `$HERMES_KANBAN_WORKSPACE`.
3. Locate the two attachment paths in the worker context or with
   `kanban_attachments()`.
4. Read the contract. Treat its SHA-256, base identity, path allow-list,
   commands, timeouts, budgets, reviewer, and forbidden actions as immutable.
5. Never edit or copy over either attachment. The verifier must execute from
   its absolute attachment path, outside the candidate worktree.

## Implementation Lane

Use this lane when you are the contract's `assignee` and the task is not a
review run.

### 1. Fail-closed preflight

Run the attached evaluator before any edit:

```bash
python "/absolute/path/code-evolution-verifier.py" \
  --contract "/absolute/path/code-evolution-contract.json" \
  --expected-contract-sha256 "<sha256 from the immutable task body>" \
  --repo "$HERMES_KANBAN_WORKSPACE" \
  --expected-workspace "$HERMES_KANBAN_WORKSPACE" \
  --expected-branch "$HERMES_KANBAN_BRANCH" \
  --preflight
```

Proceed only when the process exits zero and its JSON says `"passed": true`.
A repository mismatch, base drift, dirty starting worktree, contract mismatch,
or verifier mismatch is a hard block.

### 2. Reproduce the evidence

Reproduce the contract's stated bug, failure, or benchmark before changing
production code. A preregistration, issue description, suspicious source line,
or model judgment is not reproduction. If the evidence does not reproduce on
the frozen base, block the task rather than inventing a different objective.

### 3. Use a bounded TDD loop

1. Add or strengthen one regression test inside the allowed path set.
2. Run it and observe the expected failure.
3. Trace the real call path and implement the smallest fix for the bug class.
4. Re-run the focused test.
5. Check sibling call paths for the same flaw without expanding the contract.
6. Stop when the frozen objective is met; do not perform drive-by refactors.

Do not commit. Keeping `HEAD` on the frozen base lets the external verifier
prove that the evaluator and provenance did not move.

### 4. Run the frozen gates

Run the attached evaluator after the candidate change:

```bash
python "/absolute/path/code-evolution-verifier.py" \
  --contract "/absolute/path/code-evolution-contract.json" \
  --expected-contract-sha256 "<sha256 from the immutable task body>" \
  --repo "$HERMES_KANBAN_WORKSPACE" \
  --expected-workspace "$HERMES_KANBAN_WORKSPACE" \
  --expected-branch "$HERMES_KANBAN_BRANCH" \
  --run-gates
```

A non-zero exit means the campaign has not passed. Fix only candidate code
inside the allowed paths. Never weaken, delete, replace, skip, or reinterpret a
gate. Never run a hand-edited substitute command as proof. The verifier binds
the candidate bytes before and after the frozen gates and rejects net-created or
net-modified entries still present in the final snapshot. Gates must not mutate
the candidate at all: under the trusted-local-user boundary, a transient
mutate-then-restore sequence is prohibited but is not observable from two
snapshots without an OS sandbox or filesystem tracing. The verifier also rejects
candidate symlinks, junctions, hardlinks, and special files; use ordinary files
only.

### 5. Stop in independent review

Self-review the exact diff, then call `kanban_request_review` with the reviewer
named in the frozen contract. Include structured metadata containing:

- contract SHA-256;
- exact base commit;
- changed files;
- regression test observed RED then GREEN;
- the external verifier command;
- verifier `passed` state and frozen gate results.

Do **not** call `kanban_complete` from the implementation lane. Do not commit,
push, merge, deploy, restart Hermes, or modify a running installation.
The supported Kanban tool, CLI, and database transition paths verify the frozen
contract and reject both a different reviewer and implementation-lane
completion. The shared database transition primitives require active-run
ownership, rerun the frozen verifier before accepting the handoff, and store its
machine-generated JSON report, with standard secret redaction, as a task
attachment. Treat a rejection as a hard stop; do not bypass it through raw
database mutation or attachment edits.

## Reviewer Lane

Use this lane only when you are the contract's named `reviewer` and the task is
in first-class review.

1. Call `kanban_show()` and independently read the frozen attachments.
2. Confirm you are not the implementation profile.
3. Inspect `git diff` against the exact base; do not rely on the implementer's
   summary.
4. Re-run the attached verifier with `--run-gates`. The reviewer run, not the
   implementer's reported run, is the acceptance evidence.
5. Confirm the original failure is represented by a meaningful regression
   test, the fix addresses its root cause, all changed paths are allowed, and
   no evaluator, contract, credential, history, or live-runtime state changed.
6. If any contract, provenance, scope, test, or quality condition fails, call
   `kanban_request_changes` with concrete required changes.
7. Only when the independent rerun passes and the diff is correct, call
   `kanban_complete` with the contract SHA, base identity, changed files, and
   verifier results in metadata.

Approval marks the task done; it still does not authorize merge, push, deploy,
or restart. The completion tool performs one final frozen-verifier rerun and
stores its redacted JSON report before it can mark the task done. Those later
integration actions remain separate and human-controlled.

## Absolute Stop Conditions

Block or request changes immediately when any of these occurs:

- contract/verifier hash mismatch;
- wrong repository, commit, tree, worktree, assignee, or reviewer;
- dirty preflight;
- stated evidence cannot be reproduced;
- edit outside the allow-list;
- gate command or timeout changed;
- gate failure or timeout;
- request to commit, push, merge, deploy, restart, expose secrets, or rewrite
  history inside this campaign;
- runtime or turn budget exhausted.

Failure is retained as evidence. Never erase it with reset, checkout, stash, or
attachment deletion.
