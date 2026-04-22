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

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/publish-preflight.sh` | Check required files, auth vars, and browser-session marker |
| `scripts/browser-auth-recovery.sh` | Generate recovery packet (`--prepare`) or record verified sign-in (`--verified`) |
| `scripts/publish-unblock-helper.sh` | One-shot unblock helper: runs preflight, captures current state, emits a filled `--verified` command, and generates a timestamped handoff artifact plus `auth-artifacts/latest-publish-unblock-handoff.md` |
| `scripts/demo-capture-preflight.sh` | Verify the demo source files, rerun the underlying product preflight, and refresh `demo-artifacts/latest-demo-capture-readiness.md` before recording |
| `scripts/demo-capture.sh` | Freeze a timestamped capture-session packet (`--prepare`) and finalize `launch-execution-log.md` with recording/edit asset paths (`--finalize`) |
| `scripts/demo-capture-headless-finalize.sh` | Headless closeout path: resolve the latest session packet, require real recording/edit files, and call `demo-capture.sh --finalize` with no GUI dependency |
| `scripts/demo-capture-timed-record-wrapper.sh` | macOS timed recording path: optionally refreshes the latest packet, records the display with native `screencapture`, finalizes the launch log, and runs post-finalize verification in one command |
| `scripts/verify-demo-capture-finalize.sh` | Smoke-test the finalize flow against a temp copy of the launch log so format drift is caught before a real recording run |
| `scripts/demo-capture-post-finalize-verify.sh` | Verify the real launch log after capture: require `captured on ...`, require non-empty recording/edit files, and confirm closeout checkboxes are checked |
| `scripts/verify-demo-capture-post-finalize.sh` | End-to-end smoke test for the post-finalize verifier using a temp launch log plus non-empty temp assets |
| `scripts/demo-capture-launcher.sh` | macOS workspace primer: refreshes the latest capture packet, opens the exact walkthrough surfaces, activates QuickTime Player, and prints the finalize command |
| `scripts/verify-demo-capture-timed-record-wrapper.sh` | End-to-end smoke test for the timed recording wrapper: records a 1-second capture to temp assets, finalizes a temp launch log, and verifies closeout proof |
| `demo-trigger.md` | One-screen operator card for recording the walkthrough and closing the loop in the launch log |

Run the unblock helper any time browser auth is the known blocker:

```bash
bash starter-kits/agent-launch-closeout-kit/scripts/publish-unblock-helper.sh
```

Use `--execute` mode only when the browser is already signed in and you have a fresh signed-in screenshot saved on disk:

```bash
bash starter-kits/agent-launch-closeout-kit/scripts/publish-unblock-helper.sh --execute --screenshot-path /absolute/path/to/signed-in-proof.png
```

## Tue–Fri execution checklist
- [ ] Audit the Agentic Cron Orchestration Kit proof surface and confirm the closeout state is still the live blocker.
- [ ] Extract the canonical publish path from `agentic-cron-orchestration-kit/launch/publish-runbook.md`.
- [ ] Extract the canonical demo-capture path from `agentic-cron-orchestration-kit/launch/demo-capture-runbook.md`.
- [ ] Assemble reusable closeout-kit files for publish, capture, logging, and asset indexing.
- [ ] Validate the reusable files against the current pending-publish / pending-capture state as the before state.
- [ ] Run `bash starter-kits/agent-launch-closeout-kit/scripts/publish-preflight.sh` and surface any publish-auth gap before attempting browser or `x-cli` publish.
- [ ] Run `bash starter-kits/agent-launch-closeout-kit/scripts/publish-unblock-helper.sh` when browser auth is the blocker — it emits the exact `--verified` command and refreshes `auth-artifacts/latest-publish-unblock-handoff.md` so the next sign-in event converts directly into a publish unblock.
- [ ] Verify the live publish session is actually signed into X (the `x-access.json` marker is not enough by itself).
- [ ] If the marker is stale or the browser still lands on login, run `scripts/browser-auth-recovery.sh --prepare` and use the generated recovery packet before retrying publish.
- [ ] After a real signed-in browser proof event, run `scripts/browser-auth-recovery.sh --verified --surface-url ... --screenshot-path ...` so the launch log, audit, and `x-access.json` marker all refresh from the same evidence.
- [ ] Prime the walkthrough workspace with `bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-launcher.sh` so the latest packet, proof artifact, log, and demo outline are open before recording starts.
- [ ] On macOS, prefer `bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-timed-record-wrapper.sh` when the goal is one-command timed capture + finalize + verification from the latest packet.
- [ ] After the recording exists on disk, prefer `bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-headless-finalize.sh` to close the loop from the latest session packet without reopening the GUI launcher.
- [ ] Run `bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-post-finalize-verify.sh` after finalize so the real launch log proves `captured on ...` plus non-empty recording/edit assets before notes claim the walkthrough is done.
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
