# Publish Trigger Card — Agent Launch Closeout Kit

## Purpose
Collapse the real publish moment into one operator-facing card so the thread can be posted, logged, and closed out in one block as soon as X auth is truly restored.

Use this only after the live browser session is verified signed in.

## Preconditions
1. `bash starter-kits/agent-launch-closeout-kit/scripts/publish-preflight.sh`
2. If the browser still hits logged-out/login surfaces, stop and run:
   - `bash starter-kits/agent-launch-closeout-kit/scripts/browser-auth-recovery.sh --prepare`
3. When sign-in is restored, refresh the proof surface first:
   - `bash starter-kits/agent-launch-closeout-kit/scripts/browser-auth-recovery.sh --verified --surface-url https://x.com/compose/post --screenshot-path /absolute/path/to/signed-in-screenshot.png`

## Canonical thread payload
Source of truth: `starter-kits/agentic-cron-orchestration-kit/launch/launch-thread.md`

1. Most "autonomous" agent setups are fake autonomy. They only move when you remember to prompt them again.
2. I packaged the recurring operator loop we use inside Hermes into a starter kit: the Agentic Cron Orchestration Kit.
3. Proof-backed outcome so far: from a fresh notes context, one recurring evening-doc-sync workflow was scheduled and run in 1.74 minutes once the exact note/workspace paths were injected.
4. It ships one opinionated weekly operating system:
   - Monday kickoff
   - Daily CEO review
   - Evening doc sync
   - Friday ship review
5. The kit includes:
   - cron job prompts
   - project-note templates
   - ship checklist template
   - a local preflight script
6. This is intentionally not a dashboard or control plane. It is the fastest path to keeping one project moving without babysitting your agent.
7. The real setup contract is now clear: you still have to inject the exact note paths and workspace path into each prompt before the loop is runnable from a fresh context.
8. If you run Hermes/Codex/Claude/OpenCode-style agents and want them to keep advancing while you sleep, this is the starter system.
9. The MVP is shipping against the proved starter-workflow claim now. Next expansion after ship: broaden proof to the full four-job operating pack and record the walkthrough demo.

## Honest claim lock
- Ship line: starter-workflow claim only
- Proof anchor: `starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md`
- Key metric to preserve: **1.74 minutes**
- Hidden setup contract to disclose: exact note paths and workspace path must be injected before a fresh-context run
- Do not widen this into a fully proven four-job operating pack

## Attachment priority
1. Short walkthrough cut using `starter-kits/agentic-cron-orchestration-kit/launch/demo-captions.srt`
2. Proof-artifact still showing **1.74 minutes**
3. Screenshot of `Preflight OK`

## Publish move
1. Open `https://x.com/compose/post` in the verified signed-in browser session.
2. Paste the canonical thread payload above with no broader-claim edits.
3. Attach the highest-priority available asset from the list above.
4. Post immediately.
5. Record the result in `starter-kits/agent-launch-closeout-kit/launch-execution-log.md` before leaving the block.

## Record immediately after post
- URL:
- Timestamp:
- Attachment used:
- Notes:

## Closeout sync targets
Update these from the recorded execution result:
- `starter-kits/agent-launch-closeout-kit/launch-execution-log.md`
- `Projects/Hermes/Weekly MVP Factory.md`
- `Projects/Hermes/MVP Pipeline — Week of 2026-04-20.md`
- `Projects/Hermes/Agent Launch Closeout Kit — CEO Note.md`
- `Projects/Hermes/Agent Launch Closeout Kit — Ship Checklist.md`

## Verification
After using this card, launch closeout is not done until the execution log no longer says pending publish and the publish fields are filled.

## Demo capture — auth-independent parallel move
If X auth is the live blocker, demo capture can proceed without waiting.

Precondition check:
- Run `bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-preflight.sh`
- Confirm `starter-kits/agent-launch-closeout-kit/demo-artifacts/latest-demo-capture-readiness.md` shows `Status: ready`
- Confirm the **1.74 minutes** proof metric and the path-injection requirement are both present

Then record immediately using:
- `starter-kits/agent-launch-closeout-kit/demo-capture-runbook.md`

Record the output asset path in `starter-kits/agent-launch-closeout-kit/launch-execution-log.md` under `Demo walkthrough` → `Recording path`.

Do not wait for publish to unblock demo capture. The proof surface is already verified and preflight already passes. Record the walkthrough while auth is being restored so the attachment choice is ready the moment publish unblocks.