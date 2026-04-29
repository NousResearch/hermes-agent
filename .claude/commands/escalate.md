---
name: escalate
description: MUST BE USED when normal workflow has tried 2-3 fixes on a critical-path bug with no clear hypothesis and waiting overnight is acceptable. Spawns N>=3 sub-agents from N>=3 different model families to attack in parallel. Head agent (escalate_head role) selects only fixes that pass the bar (own test passes, full test suite passes, defensible reasoning, minimal-given-bug diff). $100 default cost cap. Operator approves at morning gate. Never auto-fires.
---

# /escalate

## Trigger criteria

All required:

1. Workflow has tried 2-3 fixes already
2. Code path is critical
3. No clear hypothesis among normal sub-agents
4. Operator wants to wait until morning
5. Cost cap is acceptable

If any fails, do not escalate. Use Chen plus audit-fix-build instead.

## Invocation

`/escalate "<problem statement>" [--budget <USD>] [--family-pool <list>]`

Defaults: `--budget 100`, family-pool from `config/models.yml`.

## Pattern

Round 1: head agent bundles context, failing test, error trace, last five commits, relevant code regions, and problem statement. It spawns N>=3 sub-agents in parallel, target N=4, spanning at least three model families. Each sub-agent independently diagnoses, patches, writes/runs a proving test, runs the existing suite, and reports PASS/FAIL with evidence.

Head agent selection bar:

1. HARD: fix passes its own test
2. HARD: existing tests still pass
3. HARD: reasoning is defensible against architecture
4. HEURISTIC: diff is minimal given the bug

Winner found: apply fix, run full test suite, report success, stop. No winner: bundle attempts and failure reasons into the next round.

## Stopping conditions

1. Winning fix found and validated
2. Convergence: three rounds with the same root-cause keyword
3. Cost cap reached
4. Operator interrupts

## Output

Write round artifacts to `audits/escalations/<timestamp>/`:

- `round-N-context.md`
- `round-N-subagent-<n>.md`
- `round-N-selection.md`
- `models-used.yml`

Morning summary writes to `OPERATOR-INBOX/<date>-escalate-<run-id>.md` as a ratification request.

## Approval gate

Never auto-commit. Operator reads, approves, and commits.

## Anti-patterns

- Not for every bug
- Not a replacement for normal workflow
- Not auto-firing
- Not a substitute for reading the morning summary
- Not coupled to any specific provider
- Head agent never participates as sub-agent in the same round
