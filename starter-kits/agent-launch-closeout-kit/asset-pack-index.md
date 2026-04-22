# Asset Pack Index — Agent Launch Closeout Kit

## Purpose
Tie the final-mile launch surface into one reusable package so a proof-backed MVP can move from "ready in theory" to an actually executed launch without hidden steps.

Use this index to keep four things aligned:
1. product-proof anchors
2. launch copy source
3. media / attachment source
4. execution logging

## Narrow ship line
This kit is for **launch closeout only**.

It does not prove the underlying product. It packages and executes the last mile for a product that already has an honest ship line.

## Separate these two truths
### Product proof
Evidence that the MVP claim is true.

Required fields:
- Product name: Agentic Cron Orchestration Kit
- Ship line: starter-workflow claim only
- Proof artifact path: `starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md`
- Proof command or run reference: `bash starter-kits/agentic-cron-orchestration-kit/scripts/preflight.sh`
- Key metric / evidence: fresh-context proof recorded at **1.74 minutes**
- Hidden setup contract to disclose: inject the exact note paths and workspace path into the prompt templates before claiming a fresh-context run

### Launch execution
Evidence that the launch actually happened or is fully publish-ready.

Required fields:
- Launch thread source path: `starter-kits/agentic-cron-orchestration-kit/launch/launch-thread.md`
- Publish runbook path: `starter-kits/agent-launch-closeout-kit/publish-runbook.md`
- Demo runbook path: `starter-kits/agent-launch-closeout-kit/demo-capture-runbook.md`
- Launch execution log path: `starter-kits/agent-launch-closeout-kit/launch-execution-log.md`
- Primary attachment: short walkthrough cut using `starter-kits/agentic-cron-orchestration-kit/launch/demo-captions.srt`
- Fallback attachment: proof-artifact still showing **1.74 minutes**
- Current closeout state: pending publish / pending capture

## Canonical asset set
### Copy
- README / product page: `starter-kits/agent-launch-closeout-kit/README.md`
- Ship note: `starter-kits/agentic-cron-orchestration-kit/launch/ship-note.md`
- Launch thread: `starter-kits/agentic-cron-orchestration-kit/launch/launch-thread.md`

### Proof
- QA / proof artifact: `starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md`
- Clean-room notes or verification log: `starter-kits/agentic-cron-orchestration-kit/launch/launch-execution-log.md`
- Screenshot or still source: proof-artifact still showing **1.74 minutes**

### Media
- Demo outline: `starter-kits/agentic-cron-orchestration-kit/launch/demo-outline.md`
- Demo capture runbook: `starter-kits/agent-launch-closeout-kit/demo-capture-runbook.md`
- Captions file: `starter-kits/agentic-cron-orchestration-kit/launch/demo-captions.srt`
- Edited asset path: record in `starter-kits/agent-launch-closeout-kit/launch-execution-log.md`

### Execution tracking
- Launch execution log: `starter-kits/agent-launch-closeout-kit/launch-execution-log.md`
- MVP ship checklist: `Projects/Hermes/Agent Launch Closeout Kit — Ship Checklist.md`
- Proof-surface ship checklist: `Projects/Hermes/Agentic Cron Orchestration Kit — Ship Checklist.md`
- Weekly pipeline note: `Projects/Hermes/MVP Pipeline — Week of 2026-04-20.md`
- CEO note: `Projects/Hermes/Agent Launch Closeout Kit — CEO Note.md`
- Factory note: `Projects/Hermes/Weekly MVP Factory.md`

## Attachment priority
1. short walkthrough cut
2. proof-artifact still showing the key metric
3. verification screenshot

Record the exact file or URL chosen in the launch execution log.

## Closeout gate
The asset pack is only complete when:
- the launch copy matches the proved ship line
- the attachment choice is explicit
- the publish path has no hidden steps
- the execution log has a URL or a named publish-ready fallback state
- the weekly notes and CEO notes can be updated from the same artifact set without reinterpretation

## Week-one proof surface
Use this real before/after case when filling the template:
- Product: `starter-kits/agentic-cron-orchestration-kit/`
- Source asset pack: `starter-kits/agentic-cron-orchestration-kit/launch/asset-pack.md`
- Source execution log: `starter-kits/agentic-cron-orchestration-kit/launch/launch-execution-log.md`
- Source publish runbook: `starter-kits/agentic-cron-orchestration-kit/launch/publish-runbook.md`

## First operator move
Before writing new launch copy, fill the proof fields and closeout-state fields above. If those are vague, the launch surface is not ready and the next block should tighten proof or execution logging instead of adding more assets.

## Auth validation
- Live browser auth audit: `starter-kits/agent-launch-closeout-kit/live-browser-auth-audit.md`
- Rule: `x-access.json` is a marker, not publish proof. Trust the actual publish session.
