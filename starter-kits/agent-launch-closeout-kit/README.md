# Agent Launch Closeout Kit

## Purpose
Turn a proof-backed AI MVP into a publish-ready launch package without losing the final mile in scattered notes, hidden steps, or stale status reporting.

This week's proof surface is the existing closeout state of `starter-kits/agentic-cron-orchestration-kit/`.

## Narrow ship line
The kit is intentionally scoped to launch closeout only.

It must help an operator take a product that is already truth-backed and close the last mile with:
- one canonical publish runbook
- one canonical demo-capture runbook
- one launch execution log
- one asset-pack index tying proof, copy, and media together
- one explicit separation between product proof and launch execution

Out of scope this week:
- dashboards
- analytics/control planes
- broad marketing suites
- multi-platform scheduling systems
- product-proof expansion for the underlying MVP

## Week-one proof surface
Use the Agentic Cron Orchestration Kit as the mandatory before/after case.

Current live pain to solve:
- `starter-kits/agentic-cron-orchestration-kit/launch/launch-execution-log.md` shows `pending publish`
- `starter-kits/agentic-cron-orchestration-kit/launch/launch-execution-log.md` shows `pending capture`

The closeout kit is only credible if it can close that real gap cleanly.

## Tue–Fri execution checklist
- [ ] Audit the Agentic Cron Orchestration Kit proof surface and confirm the closeout state is still the live blocker.
- [ ] Extract the canonical publish path from `agentic-cron-orchestration-kit/launch/publish-runbook.md`.
- [ ] Extract the canonical demo-capture path from `agentic-cron-orchestration-kit/launch/demo-capture-runbook.md`.
- [ ] Assemble reusable closeout-kit files for publish, capture, logging, and asset indexing.
- [ ] Validate the reusable files against the current pending-publish / pending-capture state as the before state.
- [ ] Run `bash starter-kits/agent-launch-closeout-kit/scripts/publish-preflight.sh` and surface any publish-auth gap before attempting browser or `x-cli` publish.
- [ ] Verify the live publish session is actually signed into X (the `x-access.json` marker is not enough by itself).
- [ ] If the marker is stale or the browser still lands on login, run `scripts/browser-auth-recovery.sh --prepare` and use the generated recovery packet before retrying publish.
- [ ] Run one real closeout cycle against the proof surface and record URL, timestamp, attachment, and asset path.
- [ ] Tighten the kit to the proved path only and remove any broadened launch-system scope.
- [ ] Freeze the kit with docs, checklist alignment, and retrospective-ready packaging.

## Division outputs
### Packaging
Create the reusable kit surface:
- `launch-execution-log.md`
- `publish-runbook.md`
- `demo-capture-runbook.md`
- `asset-pack-index.md`

### QA
Prove the kit can take the proof-backed MVP from pending closeout to logged closeout:
- published thread or fully publish-ready thread path
- URL + timestamp recorded
- demo asset path recorded
- no hidden steps in the log

### Launch
Use the proved closeout path to finish the Agentic Cron Orchestration Kit launch honestly:
- narrow claim only
- correct attachment priority
- weekly notes updated with launch execution state

## Repo scaffold decision
The MVP will live at:
- `starter-kits/agent-launch-closeout-kit/`

The first durable artifact is this README. Next artifacts should be the four core launch-closeout files, copied or adapted only where the proof surface demonstrates they are actually needed.

## Ship gate
This MVP is not done when the notes look organized.

It is done when the kit closes one real launch gap and leaves behind a reusable package that another operator could follow without asking what happens next.
