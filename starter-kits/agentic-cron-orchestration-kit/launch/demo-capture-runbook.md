# Demo Capture Runbook — Agentic Cron Orchestration Kit

## Goal
Capture a proof-backed walkthrough for the shipped **starter-workflow claim** without drifting into the unproven four-job-pack story.

## Proven line to demonstrate
From a fresh notes context, after injecting exact note paths and workspace path into the prompt template, an operator can:
1. run preflight successfully
2. schedule one recurring workflow
3. execute the evening-doc-sync loop
4. update durable notes/checklists from that workflow logic

Proof reference:
- `qa/clean-room-proof-run-2026-04-17.md`
- elapsed proof time: **1.74 minutes**

## Recording setup
- Terminal window large enough to show command + result without scrolling
- Notes window open on the weekly factory note, current pipeline note, CEO note, and ship checklist
- Timer visible or easy to overlay in post
- Do **not** improvise extra workflow claims during capture

## Canonical command path
Run from repo root:

```bash
bash starter-kits/agentic-cron-orchestration-kit/scripts/preflight.sh
```

Then show the prompt file the operator copies from:

```bash
open starter-kits/agentic-cron-orchestration-kit/prompts/evening-doc-sync.md
```

Then show the proof artifact to anchor the claim:

```bash
open starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md
```

## Shot list
### Shot 1 — Problem framing
- Show project notes that need recurring maintenance
- Voiceover / caption: "Most agents stop the second you stop prompting them."

### Shot 2 — Preflight
- Run `bash starter-kits/agentic-cron-orchestration-kit/scripts/preflight.sh`
- Hold on `Preflight OK`
- Caption: "Starter kit checks the exact files and schedule surface first."

### Shot 3 — Explicit setup contract
- Show the evening-doc-sync prompt template
- Highlight that exact note paths and workspace path must be injected
- Caption: "No fake magic — you wire the real note/workspace paths once."

### Shot 4 — Proof artifact
- Show `qa/clean-room-proof-run-2026-04-17.md`
- Highlight `1.74 minutes`
- Caption: "Fresh-context proof already recorded."

### Shot 5 — Outcome
- Show the target notes/checklist after the workflow run
- Emphasize built / blocker / next move fields updated from durable state
- Caption: "The system carries yesterday's truth into tomorrow's review."

### Shot 6 — Close
- Return to README or launch thread headline
- Caption: "Ship one recurring workflow first. Expand after proof."

## Demo guardrails
- Do not claim the full four-job pack is fully verified end-to-end
- Do not hide the path-injection requirement
- Do not frame unpublished launch tasks as product-proof blockers
- Keep the walkthrough under 90 seconds if used for social video

## Suggested narration
1. "This kit is for operators whose agents only work when babysat."
2. "The shipped line is narrow on purpose: one proof-backed recurring workflow."
3. "Preflight verifies the kit surface before you waste time."
4. "You inject your exact note and workspace paths, then run the loop against durable notes."
5. "We already recorded a fresh-context proof run in 1.74 minutes."
6. "That is enough to ship the starter workflow honestly today."

## Done criteria for demo capture
- `Preflight OK` visible in capture
- path-injection requirement shown explicitly
- proof artifact with 1.74-minute timing shown explicitly
- note/checklist outcome visible
- narration/captions match the shipped starter-workflow claim only
