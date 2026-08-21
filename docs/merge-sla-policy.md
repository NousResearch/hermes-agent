# Merge SLA and Fast-Track Merge Policy

Status:   proposal for maintainer approval
Version:  1.0
Date:     2026-08-19
Applies:  pull requests opened by the tabjoy fleet (author `devbxylw`, or PRs
          carrying the `tabjoy-fleet` label) against NousResearch/hermes-agent.

## 1. Purpose

Land implemented, tested fixes from the fleet into upstream fast enough that
the pipeline does not lose days per infra fix. Baseline (2026-08-18/19): 5 PRs
sat open 8–24h with green local test evidence, zero reviews, zero merged — the
fix loop was complete but parked on an unstaffed human merge queue.

## 2. SLA target

**SLA-1: Green tested PRs merge in under 24 hours from PR-open.**

- Clock starts at PR `created_at`; ends when the PR head is on `origin/main`.
- Draft PRs do not clock; the clock starts when marked ready for review.
- Close-and-reopen starts a new clock; force-push/rebase/new commits do not.
- Reverted-then-reopened PRs clock from the new open time.

## 3. What counts as green/tested

Eligible only if ALL hold:

1. Required CI status checks pass. A PR with an empty checks rollup is not
   green from CI alone.
2. Test evidence attached: PR body, PR comment, or the originating card carries
   an explicit report (command, environment/commit, pass/fail/skip counts).
   Evidence must reference the PR head SHA. "Tests pass" with no counts or
   command is NOT sufficient.
3. Zero failures. Skips allowed only when the reason is stated.
4. GitHub reports no merge conflict (`mergeable` clean). `UNKNOWN` is not
   clean; re-evaluate once GitHub resolves it.
5. No unresolved review threads. A formal maintainer review is NOT required
   for eligibility; absence of review does not block the clock.

**Not green regardless of evidence:** CI failing/pending/cancelled on head;
tests green on a non-head commit; evidence from a different codebase;
security-sensitive paths (section 5.6) — those always require human review.

## 4. Escalation when the SLA is missed

| Tier | Trigger | Action |
|------|---------|--------|
| T0 | eligible, < 24h | normal queue |
| T1 | eligible, >= 24h | alert on ops dashboard; comment on PR + originating card with PR URL, head SHA, age, evidence link |
| T2 | eligible, >= 48h | blocker card with baseline evidence shape; notify root orchestrator |
| T3 | eligible, >= 72h | human escalation to maintainers via the repo's preferred channel; max one per PR per 24h |

Escalation pauses while a PR is ineligible, but the clock keeps running.

## 5. Fast-track merge (this automation)

The workflow (`.github/workflows/fast-track-merge.yml`) + script
(`scripts/ci/fast-track-merge.sh`) MAY merge a PR without a manual maintainer
turn when ALL hold:

1. **Eligible** per section 3.
2. **Author/label**: author `devbxylw` OR `tabjoy-fleet` label.
3. **Not draft**; no `do-not-merge` / `blocked` label.
4. **Head freshness**: <= 50 commits behind `origin/main`, OR head pushed
   within the last 24h. Otherwise it skips (the script does not rebase).
5. **Size guard**: additions + deletions <= 400. Larger PRs do NOT auto-merge;
   they keep the SLA clock and require human review.
6. **No security-sensitive paths**: `.github/**`, `scripts/deploy*`,
   dependency manifests (`package.json`/lockfiles, `pyproject.toml`,
   `uv.lock`, `requirements*.txt`), CI config (`.woodpecker.yml`,
   `Makefile`). These always require human review.
7. **Clean at merge time**: GitHub `mergeStateStatus == CLEAN` immediately
   before merge (this enforces the `protect-main` "All required checks pass"
   requirement; a ruleset change between check and merge aborts the merge).
8. **Merge strategy**: squash merge with the PR title as the commit subject.

The bot's merge action is logged (account, PR, head SHA, check snapshot, time)
in the workflow run and as a comment on the PR, for the two-week verification.

## 6. Edge cases

- **No CI on fork PRs (current baseline state).** Upstream runs no completed
  status checks on fleet PRs while fork-PR workflow runs await maintainer
  approval (`action_required`). CI absence never makes a PR green; the PR
  must carry attached test evidence (3.2). If maintainers approve the pending
  runs (one-time), required checks supersede local evidence.
- **mergeable=UNKNOWN.** Treat as not clean; re-evaluate after GitHub
  resolves. UNKNOWN past 24h triggers T1 with the UNKNOWN state named.
- **Stale head / origin/main advanced.** Not mergeable as-is; needs rebase;
  the clock continues.
- **Reopened PRs.** New clock. Reopen used to dodge the SLA is a violation.
- **Reverts.** Treated like any other PR.
- **Author-not-fleet PRs.** Out of scope; upstream's own review norms apply.

## 7. Non-goals

- This document does not grant the fleet write access to upstream; it defines
  turnaround expectations and automation criteria for maintainers to approve.
- Monitoring/alerting lives in the ops dashboard, not this workflow.

## 8. Verification

Measure PR-open-to-merge latency and blocked age over two weeks after this
policy + automation + monitoring are live. Pass: eligible PRs merge within
24h on average and in all but documented-exception cases.

## Exact thresholds

- SLA target: < 24h PR-open to merge (eligible PRs).
- Escalation tiers: 24h / 48h / 72h (T1/T2/T3).
- Skips allowed with stated reason; failures = 0.
- Size guard: <= 400 changed lines; larger requires human review.
- Head freshness: <= 50 commits behind, or pushed within 24h.
- Security-sensitive paths: never bot-merged; clock still runs and escalates.
