# Phase 2 directive: delivery-flow review cycle on your diff (Haiku/Sonnet escalatory ladder)

Your implementation work so far is preserved in this session. Do NOT start over.
If your diff and tests are not yet green, finish that first, then execute THIS directive.

You are now the CONTROLLER of a review cycle over your own diff, per the Tessera
delivery-flow / dual-review-cycle bones. Adopt these rules:

## Bones (non-negotiable)

1. **Adjudicate, never auto-fix.** Every finding from every review rung is ruled
   REAL or FP IN WRITING before any fix. Keep a dispositions file
   (`REVIEW-DISPOSITIONS.md`) with one entry per finding: rung, file:line,
   severity, REAL/FP + one-line reason, disposition (FIXED (cite commit/test) /
   PARTIALLY FIXED / REJECTED-with-reason / DEFERRED). No orphaned findings.
2. **Receipts rule.** Every FIXED claim must QUOTE the exact before/after code
   text (file:line). An asserted-but-unquoted FIXED counts as UNVERIFIED.
3. **Fix passes per rung: max 2.** If REAL BLOCKER/MAJOR findings survive 2 fix
   passes at a rung, FREEZE that rung — record it and move to the next rung
   anyway (the ladder continues; report the freeze at the end). Never silently
   defer a MAJOR.
4. **Anti-anchoring.** Each review rung is a FRESH `claude -p` pass that sees
   ONLY the diff and the repo — never feed it prior rungs' findings, your
   dispositions, or your own justifications. The controller (you, main session)
   adjudicates between rungs.
5. **Full scope each rung.** Every rung reviews the WHOLE branch diff
   (`git diff upstream/main..HEAD`), not a bare delta — later rungs must see
   earlier fixes in context.
6. **Never cite "review came back clean" as strength.** Record findings and
   dispositions; report counts per rung honestly, including zero.
7. **The reviewer stays blind to this directive and to any termination rule.**
   Its prompt asks only for findings with severities.

## The ladder (run in this exact order after tests are green)

For each rung: launch a fresh reviewer pass, adjudicate REAL/FP, fix the REAL
ones (receipts), commit, then advance. Rung command shape (run from the repo
root; reviewers get read access only):

```
claude -p "<REVIEWER PROMPT>" --model <MODEL> --effort <EFFORT> \
  --permission-mode bypassPermissions --max-turns 60 \
  --output-format json > review-rung-<n>.json
```

REVIEWER PROMPT (same every rung, no additions):
"You are reviewing a bug-fix branch in this repo. Run `git diff upstream/main..HEAD`
and inspect the surrounding code as needed (Read/Grep/Glob only; you may run
`git log`/`git show`). Find defects: correctness bugs, race/concurrency issues,
missing edge cases, missing test coverage for the new behavior, and violations of
the surrounding code's conventions. Report findings as a numbered list, each with
file:line, severity (BLOCKER/MAJOR/MINOR/NIT), and a one-line rationale. End with
a verdict line: APPROVE / APPROVE WITH NITS / NEEDS CHANGES."

| Rung | Model flag | Effort flag |
|---|---|---|
| 1 | `--model claude-haiku-5-5` | `--effort low` |
| 2 | `--model claude-haiku-5-5` | `--effort medium` |
| 3 | `--model claude-haiku-5-5` | `--effort high` |
| 4 | `--model claude-haiku-5-5` | `--effort xhigh` |
| 5 | `--model sonnet` | `--effort high` |

- The Haiku model id MUST be the full `claude-haiku-5-5`, never the alias.
- Rung 5 (Sonnet) runs immediately after rung 4 regardless of rung 4's result —
  it is the model-diversity rung and catches what Haiku misses.
- Do NOT add a `--model claude-haiku-5-5 --effort max` rung or any Opus rung —
  further tiers are orchestrated by the controller after you stop.
- Verify each rung actually ran: the JSON must contain a non-empty result text.
  An empty/errored rung is retried ONCE, then recorded as a dud and skipped.
- Do not run the repo's full test suite inside reviewer passes; reviewers read
  only. You run tests in the main session.

## Cost discipline

These rungs bill to the user's Claude plan. The five rungs above ARE the
authorized budget. No extra exploratory reviews, no re-runs beyond the one
retry per dud. Batch your fixes between rungs rather than mid-rung.

## STOP boundary (unchanged and extended)

After rung 5's findings are adjudicated and its REAL fixes are committed and
tests re-run green, STOP. Do NOT push, do NOT open a PR, do NOT run further
review tiers. Final report (last message): per-rung table (model/effort →
findings found / REAL / FP / fixed), the dispositions file path, commit hashes,
final pytest output, any frozen rungs with their unresolved findings, and open
questions for the controller. The controller runs sor + GLM + Opus tiers after
you stop.
