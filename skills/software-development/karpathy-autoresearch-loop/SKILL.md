---
name: karpathy-autoresearch-loop
description: "Run Karpathy-style autoresearch loops in Hermes: bounded code surface, fixed evaluator, scalar metric, git keep/revert, and long unattended experiment iteration. Use when user mentions autoresearch, Karpathy auto research, overnight repo optimization, benchmark hill-climbing, or autonomous research loops."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [autoresearch, karpathy, experiments, optimization, agents]
    related_skills: [karpathy-autonomy-persona, research-proof-kernels, run-scoped-causality-gate]
---

# Karpathy-Style AutoResearch Loop

## Purpose

Run a bounded autonomous experiment loop that can improve code against a fixed verifier without turning into theater.

This is not general "make the repo better" autonomy. It requires a tight contract:

```text
human strategy file -> one bounded code mutation -> fixed eval -> scalar metric -> keep/revert -> ledger -> repeat
```

## Required repo contract

Before running unattended, require:

- `program.md` or equivalent strategy file written/owned by the human;
- one or few explicitly allowed editable files;
- fixed evaluator script the agent cannot edit;
- machine-readable metric output;
- clear lower/greater-is-better rule;
- timeout/runtime budget;
- git branch for the run;
- untracked `results.tsv` or equivalent ledger;
- rollback rule using git reset/revert.

If any item is missing, build the harness first. Do not start a long loop.

## Standard loop

1. Check branch and git status.
2. Establish baseline metric if missing.
3. Read `program.md`, recent results, and allowed files.
4. Make exactly one experimental change.
5. Commit the change.
6. Run the fixed evaluator with timeout and redirected logs.
7. Parse metric output.
8. If metric improves under acceptance rule, keep commit.
9. If metric regresses/crashes, reset to prior commit.
10. Append attempt to `results.tsv`.
11. Repeat until experiment/time budget ends.

## Keep/revert rule

Keep only when the primary metric improves and guardrail metrics do not regress beyond allowed thresholds. Simpler equal-performance changes may be kept only if the human strategy permits simplification wins.

## Anti-theater constraints

- Do not edit evaluator, metric parser, or data split.
- Do not accept self-reported status as metric.
- Do not count report generation as improvement.
- Do not hide crashes; log and classify them.
- Do not run for hours until one-run and small-N smoke tests pass.

## Output report

Report:

```text
baseline metric:
best metric:
experiments attempted:
kept commits:
discarded/crashed attempts:
best commit:
main improvement classes:
failure modes:
next human strategy update:
```

## Hermes execution notes

For long runs, use `terminal(background=true, notify_on_complete=true)` or a self-contained cron job with `workdir`, `terminal/file` toolsets, and this skill. Prefer manual/background first; cron only after the harness proves safe.
