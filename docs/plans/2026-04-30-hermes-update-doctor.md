# Hermes Update Doctor — Implementation Plan

> For Hermes: use this plan to implement the feature incrementally with small, auditable patches.

**Goal:** turn the current safe-update dry-run into an agentic update doctor that can diagnose replay conflicts, classify them, attempt safe sandbox repairs, run focused tests, and only stop for real risk.

**Architecture:** keep the existing `hermes-safe-update` safety wrapper and `hermes update` runtime unchanged at first. Add a new orchestration layer that performs a realistic sandbox replay in a temporary worktree, classifies any conflict, applies only low-risk repairs by policy, runs focused tests, and emits a structured report. The first implementation should be diagnosis-first, with auto-repair limited to trivial/test-only cases and explicit opt-in for any destructive or merge-like action.

**Tech Stack:** Bash wrapper(s), existing Hermes CLI Python code, git worktrees, patch-id/cherry-pick classification, pytest, and the current update policy doc.

---

## 1) Problem Statement

`hermes-safe-update --dry-run` currently does the right thing by stopping when replay looks risky, but it leaves too much manual work to the operator. The desired end state is a tool that can:

- detect update/replay conflicts early,
- classify the conflicting commit and hunk,
- try safe resolution in a sandbox,
- validate with focused tests,
- and create a branch/PR if the repair is safe enough.

It must never silently touch `origin`, never force-push, and never mutate the active `main` checkout during diagnosis.

---

## 2) Desired User Flow

Target flow for the normal `hermes-update-doctor --run` command (with `--analyze` and `--repair` kept as technical/debug modes):

1. Preflight checks
   - confirm current branch and remotes
   - confirm clean or explainable status
   - confirm target tracking branch

2. Create rescue ref
   - always create a rescue ref before any replay or destructive action
   - print the rescue ref in the final report

3. Create sandbox worktree
   - use a temporary worktree starting from the exact base that the replay would use
   - never do diagnosis in the active checkout

4. Simulate update/replay realistically
   - fetch the target base if needed
   - replay the exact candidate commits in order
   - stop on the first conflict

5. Classify the conflict
   - patch-id duplicate
   - already covered in `fork/main`
   - obsolete commit
   - test expectation only
   - simple hunk/context conflict
   - runtime-sensitive conflict
   - critical/sensitive area

6. Resolve by policy
   - skip duplicate/covered/obsolete commits
   - apply trivial hunk fixes when confined and unambiguous
   - update tests when the failure is test-only and behavior is already correct
   - stop immediately for sensitive areas or ambiguous runtime changes

7. Run focused tests
   - infer tests from changed paths
   - run the smallest meaningful test set first
   - escalate only if needed

8. Apply safely
   - only apply changes to a branch/worktree that is explicitly designated for repair
   - never rewrite the user’s active `main`

9. Create branch/PR
   - if repair succeeds and the policy allows publication, create a branch and PR on the fork
   - do not auto-merge in the first version

10. Final report
   - root cause
   - classification
   - actions taken
   - tests run
   - branch/PR URL if created
   - any remaining risks or manual follow-up

---

## 3) Conflict Classification Model

The doctor should classify each conflicting commit/hunk into one of these buckets:

### A. Duplicate by patch-id
Signals:
- identical or equivalent patch-id already seen
- same effect already present locally or on fork/main

Action:
- skip from replay queue
- record as upstream-equivalent

### B. Already covered in fork/main
Signals:
- commit patch-id not identical, but the effect is present in fork/main
- diff against fork/main shows no meaningful net change

Action:
- skip or mark as already integrated
- do not attempt cherry-pick

### C. Obsolete
Signals:
- commit references a code path or test that has been superseded
- newer code path renders the old change unnecessary

Action:
- skip, but record rationale in report

### D. Test desynchronized
Signals:
- runtime behavior already matches the intended outcome
- only the test expectation is stale
- patch changes are limited to tests or assertions

Action:
- update the test, rerun focused tests, then continue

### E. Simple hunk conflict
Signals:
- one small, isolated file hunk
- conflict is context-only and can be resolved mechanically
- no sensitive area touched

Action:
- attempt minimal patch in sandbox
- rerun focused tests

### F. Runtime-sensitive
Signals:
- auth, memory, DB/WAL, gateway, update/replay internals, or package-lock mutation
- code path affects actual runtime behavior

Action:
- stop unless explicitly approved for deeper repair

### G. Critical area
Signals:
- security/auth/provider secrets
- storage integrity
- replay machinery itself
- destructive filesystem or git history actions

Action:
- stop immediately
- require human review

---

## 4) Resolution Policies

### Skip commit when
- patch-id matches an already-seen or already-applied effect
- commit is proven covered by `fork/main`
- commit is obsolete and no longer needed

### Apply a patch manually when
- conflict is a simple, isolated hunk
- context is unambiguous
- file is not in a sensitive area
- sandbox repair passes focused tests

### Correct a test when
- runtime behavior is already correct
- the failure is due to stale expectation
- changed files are test-only or assertion-only

### Create a PR when
- repair changes user-visible code or tests beyond a tiny local adjustment
- the branch is clean and tests pass
- the policy allows publication to the fork

### Stop when
- a conflict touches a sensitive area
- classification is uncertain
- tests fail in a way that may indicate runtime regression
- the repair would require touching `origin`, force-push, or rewriting shared history

---

## 5) Sensitive Areas

Treat the following as high-risk and default to stop:

- auth/providers and credential flows
- memory, state DB, WAL, and persistence
- security and redaction boundaries
- gateway processes and long-lived service control
- update/replay machinery itself
- package-lock / dependency refresh in the active checkout
- any docs-only change that appears to mask runtime drift

These areas can be auto-classified, but not auto-repaired in v1 unless a commit is obviously test-only and the runtime behavior is already validated.

---

## 6) Test Selection Strategy

Map changed files to focused tests before running a broad suite.

Suggested initial mapping:

- `tests/hermes_cli/test_cmd_update.py`
  - `tests/hermes_cli/test_cmd_update.py`
  - `tests/hermes_cli/test_update_final_report.py`
  - `tests/hermes_cli/test_update_commit_replay.py`
  - `tests/hermes_cli/test_update_local_commit_guard.py`

- `hermes_cli/main.py`
  - update-focused tests under `tests/hermes_cli/`
  - any directly impacted area tests, selected by path or symbol name

- `tests/gateway/test_update_command.py`
  - gateway update command tests

Heuristic rules:
- start with the smallest test file directly tied to the changed file
- if a runtime file changes, run the adjacent feature tests first
- if only a test file changes, run that file plus the closest sibling update tests
- if package-lock or web build changes, add the web UI tests or targeted build checks

---

## 7) Safety Rules

Non-negotiable:

- never touch `origin`
- never force-push
- never reset or rebase shared history automatically
- always create a rescue ref before any risky operation
- never delete a remote branch without explicit authorization
- preserve logs and the final report
- keep the active `main` checkout untouched during sandbox diagnosis

Additional safety posture:
- use a temporary worktree for replay simulation
- do not apply repairs directly to the active checkout until the sandbox version passes tests
- prefer narrow, reversible changes
- if the classifier cannot decide, stop and report rather than guess

---

## 8) Output Contract

The doctor’s final report should include:

- current branch and tracking branch
- remotes and which ones were consulted
- rescue ref
- base head / target ref / replay base
- replay result and first conflict commit if any
- classification of the conflict
- actions taken in sandbox
- tests executed and results
- branch/PR created, if any
- explicit stop reason if the flow halted
- `origin_ahead_count`
- `skipped_safe_commits`
- `replay_continued_after_skip`
- `integration_status`
- `material_changes_detected`
- `--run` is the human-facing command; `--analyze` and `--repair` remain technical/debug modes
- `--pr` adds publication to the fork when a validated material change exists
- `--auto-merge-low-risk` only merges low-risk PRs after clean GitHub metadata checks
- dry-run/replay is internal to `--run`, not a separate manual step
- confirmation that `origin` was not touched

---

## 9) Existing Components to Reuse

Reuse the current building blocks instead of rewriting them:

- `hermes-safe-update` wrapper
- `~/.hermes/update-policy.md`
- `hermes_cli/main.py` update/replay functions
- current update-focused pytest coverage
- git worktree/replay simulation already used by the dry-run wrapper
- `--run` should consume that internal dry-run and decide the safe path automatically

The new work should sit on top of these primitives, not replace them in one shot.

---

## 10) Implementation Phases

### Phase 1 — Specification and scaffolding
- write this plan
- define the command/mode name and user-facing contract
- identify the diagnostic data model and report fields
- no runtime behavior changes yet

### Phase 2 — Sandbox diagnosis only
- add the new doctor entrypoint or `--auto-repair` mode shell wrapper
- create rescue ref + temporary worktree
- simulate replay
- classify conflict types
- emit structured report
- stop before any repair

### Phase 3 — Low-risk auto-repair
- handle duplicate-by-patch-id as skip-safe
- handle already-covered commits as skip-safe
- when a safe skip occurs in the sandbox, abort the cherry-pick, record it in `skipped_safe_commits`, and continue replaying later commits
- if safe skips happen while `origin/main` is still ahead, report `integration_status=needs-integration` instead of `completed`
- keep test-desync and simple hunk repair scaffolding sandbox-only for a later patch
- rerun focused tests after any actual repair in sandbox

### Phase 4 — Branch/PR publication
- create a fork branch only when a validated material change exists
- open a PR against fork/main
- keep `origin` untouched
- allow `--pr` to stay a no-op when the run produced no material change

### Phase 5 — Optional low-risk auto-merge gate
- only under explicit opt-in via `--auto-merge-low-risk`
- only when `--pr` is enabled
- only for low-risk scopes: docs-only, tests-only, metadata/report-only, skip-safe/no-op, patch-id duplicate, already-covered-in-fork-main
- require GitHub metadata to be `MERGEABLE` and `CLEAN`
- merge normally, never squash
- refresh local main after merge and run final validation

---

## 11) First Patch Minimum Recommended

The first code patch should be intentionally small:

1. Add a new orchestration command or mode that wraps the existing safe-update flow.
2. Keep `hermes-safe-update` behavior intact by default.
3. Introduce a diagnosis-only `--run` or `hermes update-doctor` path that:
   - creates a rescue ref,
   - creates a temp worktree,
   - simulates replay,
   - classifies the first conflict,
   - prints a structured report,
   - and exits without making changes.
4. Add a tiny classifier with only these initial buckets:
   - patch-id duplicate
   - already covered in fork/main
   - conflict in one hunk
   - stale test expectation
5. Add tests for the classifier, run mode, and the dry-run report.

This gives us a safe foundation before any repair automation is allowed.

---

## 12) Files Likely to Change Later

Potential follow-up files:

- `/root/.local/bin/hermes-safe-update`
- `/root/.hermes/update-policy.md`
- `/opt/hermes-agent/hermes_cli/main.py`
- new helper modules under `hermes_cli/` for classification/reporting
- `tests/hermes_cli/test_update*.py`
- `tests/hermes_cli/test_update_doctor*.py`
- this plan file itself

Do not change runtime code until the diagnosis-only mode and tests are specified clearly.

---

## 13) Success Criteria

The feature is useful when:

- a conflicting replay no longer stops at “manual diagnosis needed” for trivial cases
- the doctor can correctly skip duplicates and already-covered commits
- stale test expectations are identified and corrected
- simple hunk conflicts can be repaired in sandbox
- the user gets a clear report and a PR when the repair is safe
- `origin` remains untouched throughout
