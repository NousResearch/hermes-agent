# Demo Capture Runbook — Agent Launch Closeout Kit

## Goal
Capture one proof-backed walkthrough for a shipped MVP without drifting into broader, unproven marketing claims.

## Week-one proof surface
Use `starter-kits/agentic-cron-orchestration-kit/` as the example product.

Demonstrate only this proved line:
1. preflight succeeds
2. the explicit path-injection requirement is visible
3. the clean-room proof artifact is visible
4. the closeout system records launch execution state without hidden steps

Proof reference:
- `starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md`
- elapsed proof time: **1.74 minutes**

## Recording setup
- Terminal window large enough to show commands and outputs without scrolling
- Notes window open on the weekly factory note, current pipeline note, CEO note, and ship checklist
- Launch execution log open so the operator can show where publish/capture state gets recorded
- Do not improvise broader workflow claims during capture

## Canonical command path
Run from repo root:

```bash
bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture.sh --prepare
```

This reruns `scripts/demo-capture-preflight.sh`, refreshes `demo-artifacts/latest-demo-capture-readiness.md`, and freezes a timestamped `demo-artifacts/demo-capture-session-*.md` packet with the suggested raw/edit asset paths and the exact finalize command.

For the one-screen operator version of this path, use:
- `starter-kits/agent-launch-closeout-kit/demo-trigger.md`

For the macOS workspace-launch version of this path, use:

```bash
bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-launcher.sh
```

That command reruns `--prepare`, resolves the newest session packet, opens the readiness/proof/log surfaces in an editor, activates QuickTime Player, and prints the exact finalize command again so the walkthrough can start without hunting files.

After recording/editing, prefer the headless closeout path so the launch log can be finalized later without reopening QuickTime or the GUI workspace:

```bash
bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-headless-finalize.sh \
  --recording-path /absolute/path/to/raw-demo-capture.mov \
  --edited-asset-path /absolute/path/to/final-demo.mp4 \
  --duration 00:01:19
```

By default that helper resolves the latest `demo-capture-session-*.md` packet, reuses its suggested raw/edit paths when you omit them, and requires the recording/edit files to exist before it touches `launch-execution-log.md`.

Before a real recording run, verify the finalize path still matches the current log format:

```bash
bash starter-kits/agent-launch-closeout-kit/scripts/verify-demo-capture-finalize.sh
```

After the recording/edit is complete, close the loop with:

```bash
bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture.sh --finalize \
  --recording-path /absolute/path/to/raw-demo-capture.mov \
  --duration 00:01:19 \
  --edited-asset-path /absolute/path/to/final-demo.mp4
```

Use that lower-level command directly only when you intentionally need to override the headless helper's session-packet defaults.

Then verify the real launch log and asset files before updating the weekly notes:

```bash
bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-post-finalize-verify.sh
```

For a temp-file smoke test of that verifier, run:

```bash
bash starter-kits/agent-launch-closeout-kit/scripts/verify-demo-capture-post-finalize.sh
```

Then show the proof artifact:

```bash
open starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md
```

Then show the closeout logging surface:

```bash
open starter-kits/agent-launch-closeout-kit/launch-execution-log.md
```

## Shot list
### Shot 1 — Problem framing
- Show the launch execution log with `pending publish` and `pending capture`
- Caption: "Proof-backed products still die if the last mile lives only in your head."

### Before recording
- Prefer `scripts/demo-capture-launcher.sh` on macOS so the workspace is opened in the same order every time.
- Confirm the latest session packet's suggested raw/edit asset paths before pressing record.

### Shot 2 — Preflight
- Run `bash starter-kits/agentic-cron-orchestration-kit/scripts/preflight.sh`
- Hold on `Preflight OK`
- Caption: "Start from a verified product surface, not vibes."

### Shot 3 — Hidden setup contract
- Show that exact note paths and workspace path must be injected into prompt templates
- Caption: "No fake magic — the setup contract is explicit."

### Shot 4 — Proof artifact
- Show `qa/clean-room-proof-run-2026-04-17.md`
- Highlight `1.74 minutes`
- Caption: "The product claim is already proved."

### Shot 5 — Closeout system
- Show `launch-execution-log.md` plus the publish/demo runbooks
- Emphasize URL, timestamp, attachment choice, and asset path fields
- Caption: "This kit closes the launch mile without reopening scope."

### Shot 6 — Close
- Return to the closeout-kit README or asset-pack index
- Caption: "Separate product proof from launch execution, then finish the launch."

## Demo guardrails
- Do not claim the demo itself proves the product; it proves the closeout process.
- Do not widen the claim beyond the underlying shipped line.
- Do not hide the path-injection requirement.
- Keep the walkthrough under 90 seconds if used for social video.

## Done criteria
- `Preflight OK` visible
- path-injection requirement shown explicitly
- proof artifact with **1.74 minutes** shown explicitly
- launch execution log shown explicitly
- narration/captions match the proved claim only
