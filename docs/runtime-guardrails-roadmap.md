# Hermes Runtime Guardrails Roadmap

This roadmap tracks the remaining runtime-enforcement work needed to move Hermes from policy/docs guidance to hard pre-tool and turn-level guardrails.

## Current baseline

The first hard runtime gate is now committed locally on `feat/runtime-guardrails`:

- Commit: `a2577d1de feat: add runtime self-modification guard`
- Hook point: `model_tools.handle_function_call()`
- Purpose: block unrequested writes to protected Hermes control/config paths before tool execution.
- Current coverage:
  - protected `.hermes` skill/profile/docs/hooks/scripts/.github/config paths;
  - terminal/shell write attempts to protected paths;
  - `skill_manage` or equivalent self-improvement writes;
  - Notion-healthcheck-style attempts to mutate Notion skill files;
  - propagation of current user request into nested tool calls.

This is the first runtime-enforced guard. The remaining gates below are still pending.

## Phase 0 — Stabilise current runtime guard

Goal: prove the committed self-modification gate works in the actual live Hermes CLI.

Tasks:

```bash
python3 -m pytest tests/test_self_modification_guard.py -q
```

Run a live Hermes smoke test:

```text
Check my Notion connection/config only. This is a read-only smoke test. Do not edit files, skills, docs, config, scripts, profiles, hooks, or any `.hermes` tracked files. Report only whether Notion config/API reachability is present.
```

After the smoke test, check:

```bash
cd /home/renchey/.hermes/hermes-agent
git status --short --branch

cd /home/renchey/.hermes
git status --short --branch
```

Exit criteria:

- `hermes-agent` branch stays clean except intentional runtime-guardrail work.
- parent `.hermes` config repo has no surprise skill/config/doc edits.
- no self-improvement patch is silently applied.

## Phase 1 — Finish self-modification lifecycle gate

Goal: extend the current self-modification write gate from simple write-blocking into the full protected-edit lifecycle.

Required runtime state:

- explicit user instruction;
- named or clearly scoped target files;
- pre-edit `git status --short` captured;
- protected write performed;
- post-edit validation captured;
- final diff/status summary required.

Rules:

A protected write may proceed only when:

1. the latest real user request explicitly asks for Hermes self-improvement, Hermes config editing, skill editing, or runtime guardrail work;
2. the target files are named or clearly scoped;
3. `git status --short` was checked before editing.

After a protected write, Hermes must report:

- files changed;
- validation/tests run;
- secret scan result where relevant;
- final diff summary;
- final git status;
- whether commit/push happened.

Tests to add:

- explicit self-improvement request without target files is blocked;
- explicit request with named/scoped target files passes;
- protected write without pre-edit git status is blocked;
- after protected write, final response must include validation/diff/status summary;
- shell write variants are blocked when unrequested:
  - `>`;
  - `>>`;
  - `tee`;
  - `sed -i`;
  - Python `open(..., "w")`;
  - `cp` or `mv` into protected paths.

Exit criteria:

- current self-modification guard is lifecycle-complete.
- ordinary tasks cannot silently mutate protected Hermes files.
- explicit protected edits are auditable.

## Phase 2 — Guardrail evidence store

Goal: create shared runtime evidence state so later guards do not invent separate ad-hoc state.

Suggested evidence object fields:

- latest user request;
- pre-edit git status;
- PR number;
- PR repository;
- live PR head branch;
- live PR head SHA;
- local HEAD SHA;
- CI run SHA;
- CI checkout SHA;
- failing blob SHA;
- failing file;
- failing line;
- review thread/comment ID;
- whether fetched PR patch was checked.

Exit criteria:

- evidence can be stored in turn/session context.
- nested tool calls can access the same evidence.
- later gates can consume the evidence without duplicating context plumbing.

## Phase 3 — PR-head invariant gate

Goal: Hermes cannot diagnose, patch, push, merge, rerun CI, or resolve PR comments unless the local checkout matches the live PR head.

Required invariant:

```text
live_pr_head_sha == git rev-parse HEAD
```

Applies before:

- `git push`;
- `gh pr edit`;
- `gh pr merge`;
- workflow reruns or CI reruns;
- review-thread resolution;
- patching files while working a PR;
- diagnosing CI failure as PR-related.

Blocked response should include:

- PR number;
- live PR head branch;
- live PR head SHA;
- local branch;
- local HEAD SHA;
- clear instruction to repair checkout/ref alignment first.

Tests:

- matching live PR head allows PR mutation;
- mismatched local HEAD blocks PR mutation;
- mismatched local HEAD blocks CI diagnosis;
- mismatched local HEAD blocks review-thread resolution;
- stale local branch alias is blocked.

Exit criteria:

- the PR #1010/#1020 stale-local-branch failure class cannot recur.

## Phase 4 — Review-thread resolution gate

Goal: Hermes cannot mark a review thread resolved just because local files look fixed.

Required evidence:

- specific thread/comment ID;
- live PR head verified;
- fix committed to live PR head;
- fetched PR patch confirms reviewer issue is gone.

Tests:

- no thread ID blocks resolution;
- local-only fix blocks resolution;
- fix not pushed to PR head blocks resolution;
- fetched patch still contains issue blocks resolution;
- live PR head patch proving issue gone allows resolution.

Exit criteria:

- review threads cannot be resolved without remote PR-head evidence.

## Phase 5 — Stale-CI evidence gate

Goal: Hermes cannot claim "CI is stale" unless it can prove that claim.

Required evidence:

- live PR head SHA;
- CI checkout SHA or run/head SHA;
- failing file/blob/line evidence;
- proven SHA mismatch.

Allowed classifications without full evidence:

- unknown;
- local checkout mismatch;
- CI failure unclassified;
- needs run metadata;
- base failure suspected but unproven;
- PR-introduced failure suspected but unproven.

Disallowed without evidence:

- "CI is stale";
- "dirty CI";
- "old failure";
- "just rerun it";
- empty CI retrigger commit.

Tests:

- stale-CI wording blocked without CI checkout SHA;
- stale-CI wording blocked without live PR head SHA;
- stale-CI wording blocked without failing blob/line evidence;
- stale-CI classification allowed only when mismatch is proven;
- empty CI retrigger commit blocked unless stale evidence exists.

Exit criteria:

- stale-CI claims are evidence-gated, not guessed.

## Phase 6 — No-progress loop detector

Goal: stop Hermes from burning time/tokens repeating the same commands without new facts.

Track fingerprints:

- tool name;
- cwd;
- command/action;
- important arguments;
- result status;
- short result hash.

Rules:

- After 2 repeated no-new-fact cycles: force self-audit.
- After 3 repeated no-new-fact cycles: block further repeats unless the user explicitly instructs continuation.

Required self-audit format:

- current objective;
- latest real user request;
- last new fact;
- current invariant status;
- next single action;
- stop/continue decision.

Tests:

- repeated identical `gh` checks trigger self-audit;
- repeated `grep`/`status` loops trigger self-audit;
- third repeat blocks execution;
- genuinely new evidence resets the loop counter.

Exit criteria:

- repeated no-progress loops stop automatically.

## Phase 7 — Latest-user-task after compaction

Goal: compaction summaries cannot silently resurrect old tasks.

Runtime rule:

```text
latest real user message > compaction summary > old todo/task state
```

Likely implementation location:

- session/context loader;
- message assembly layer;
- task-state restoration layer.

Tests:

- stale compaction says "continue PR work";
- later user says "check Notion only";
- active task resolves to Notion only;
- GitHub/PR tools are not invoked.

Exit criteria:

- old summaries cannot override newer user instructions.

## Phase 8 — Notion healthcheck route

Goal: Notion checks become first-class and read-only.

Allowed:

- check token presence without printing value;
- GET `/v1/users/me`;
- optionally read database/data source metadata.

Blocked:

- editing skills;
- editing Notion docs;
- updating PR tracker;
- inspecting GitHub;
- mutating config.

Tests:

- token presence does not print token;
- `/users/me` reachable reports workspace/bot safely;
- missing token reports missing config clearly;
- Notion healthcheck cannot write Notion skill files;
- Notion healthcheck cannot drift into GitHub PR work.

Exit criteria:

- "check Notion connection" has a safe, narrow runtime path.

## Branch strategy

Recommended branch sequence:

- `feat/runtime-guardrails` — current self-modification write gate;
- `feat/runtime-guardrails-lifecycle`;
- `feat/runtime-pr-head-guard`;
- `feat/runtime-review-thread-guard`;
- `feat/runtime-ci-evidence-guard`;
- `feat/runtime-loop-detector`;
- `feat/runtime-compaction-task-authority`;
- `feat/runtime-notion-healthcheck`.

Rules:

- Keep each phase as a small branch/commit.
- Do not combine unrelated gates.
- Do not inspect or apply the pre-existing local stash on these branches.
- Push only to `fork`, not upstream `origin`, unless explicitly preparing an upstream PR.

## Production-readiness ladder

### Level 1 — Controlled use

Self-modification guard committed and smoke-tested.

Allowed:

- low-risk read-only checks;
- simple local diagnostics;
- no PR/CI/review automation.

### Level 2 — Protected self-editing

Lifecycle gate complete.

Allowed:

- controlled Hermes config/skill updates with audit trail.

### Level 3 — PR-safe

PR-head invariant and review-thread gates complete.

Allowed:

- PR diagnosis and review-thread work with evidence checks.

### Level 4 — CI-safe

Stale-CI evidence gate complete.

Allowed:

- CI diagnosis and rerun recommendations with evidence.

### Level 5 — Production automation

Loop detector, compaction task authority, and Notion route complete.

Allowed:

- production work queues with reduced supervision.
