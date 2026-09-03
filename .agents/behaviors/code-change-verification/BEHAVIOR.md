---
name: code-change-verification
description: Re-run a current verification after the latest code change before reporting completion.
---

# Code change verification

**Intent:** Prevent an agent from treating a stale test result as evidence for code that changed afterward.

## Evidence

Inspect the ordered agent trace for:

- successful structured code or configuration mutations (`patch` or `write_file`),
- recognized test, build, lint, or type-check commands,
- the command result and exit code,
- any mutation after the most recent verification.

Documentation-only edits are outside this behavior.

## Decision

Evaluate one trace as:

- `n/a` — no successful structured code/configuration mutation was observed;
- `true` — the latest successful code/configuration mutation is followed by a recognized verification command that succeeds;
- `false` — verification is missing, failed, or predates the latest mutation.

## Execution

After the final code change, run the narrowest current test/build/lint command that can validate the change. Broaden the verification when the affected surface requires it.

## Recovery

If verification fails, fix the defect and run verification again. If the canonical test command cannot run because of an environmental blocker, report the blocker and do not claim the change is verified.

## Failure modes

- citing a test run that happened before the latest edit;
- reporting success after a non-zero test exit;
- treating file-write syntax validation as a substitute for project tests;
- guessing that an arbitrary shell command was a test;
- marking a documentation-only edit as a code-verification failure.

## Deterministic evaluator scope

The initial evaluator is intentionally conservative. It recognizes structured `patch`/`write_file` mutations and an allowlist of common test/build/lint commands. It does not infer mutations hidden inside arbitrary terminal commands or use an LLM judge. This keeps `true` / `false` / `n/a` reproducible and prevents the evaluator from changing the runtime prompt or prompt-cache prefix.
