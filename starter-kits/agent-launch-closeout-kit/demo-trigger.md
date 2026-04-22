# Demo Trigger Card — Agent Launch Closeout Kit

## Purpose
Collapse walkthrough capture into one operator-facing card so the demo can be recorded, finalized, and logged in one block without rebuilding context from multiple notes.

Use this while X publish auth is blocked or whenever the walkthrough needs to be captured as the primary attachment.

## Preconditions
1. Run the readiness pass:
   - `bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture.sh --prepare`
   - or on macOS for a primed workspace: `bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-launcher.sh`
   - or on macOS for one-command timed capture + finalize + verification: `bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-timed-record-wrapper.sh`
2. Confirm `starter-kits/agent-launch-closeout-kit/demo-artifacts/latest-demo-capture-readiness.md` shows `Status: ready`.
3. Open the latest session packet in `starter-kits/agent-launch-closeout-kit/demo-artifacts/demo-capture-session-*.md` and use its suggested raw/edit asset paths.

## Claim lock
- This walkthrough proves the **closeout process**, not broader product proof.
- Keep the underlying product claim narrow: one proof-backed starter workflow completed in **1.74 minutes** after exact note/workspace path injection.
- Do not widen the story into a fully proven four-job operating pack.

## Surfaces to show
- `starter-kits/agent-launch-closeout-kit/demo-artifacts/latest-demo-capture-readiness.md`
- `starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md`
- `starter-kits/agent-launch-closeout-kit/launch-execution-log.md`
- `starter-kits/agentic-cron-orchestration-kit/launch/demo-outline.md`

## Six-shot capture path
1. **Problem framing** — show the launch execution log still carrying pending closeout state
   - Caption: "Proof-backed products still die if the last mile lives only in your head."
2. **Preflight** — show `Preflight OK`
   - Caption: "Start from a verified product surface, not vibes."
3. **Explicit setup contract** — show the exact path-injection requirement
   - Caption: "No fake magic — the setup contract is explicit."
4. **Proof anchor** — show the clean-room proof artifact and highlight **1.74 minutes**
   - Caption: "The product claim is already proved."
5. **Closeout system** — show the log plus runbook fields for URL, timestamp, attachment, and asset path
   - Caption: "This kit closes the launch mile without reopening scope."
6. **Close** — return to the README or asset-pack index
   - Caption: "Separate product proof from launch execution, then finish the launch."

## Suggested narration
1. "This kit is for operators who get to proof and still fail to finish the launch."
2. "The claim stays narrow on purpose: one proof-backed starter workflow, not a fake fully autonomous system."
3. "Preflight verifies the real product surface before capture starts."
4. "The hidden setup contract is explicit: inject the exact note and workspace paths first."
5. "The proof artifact already records a fresh-context result in 1.74 minutes."
6. "This closeout kit makes the final launch steps durable instead of trapped in memory."

## Finalize immediately after recording
Use the exact command from the latest `demo-capture-session-*.md` packet, then verify the log no longer says `pending capture`.

Canonical form:

```bash
bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture.sh --finalize \
  --recording-path /absolute/path/to/raw.mov \
  --duration 00:01:19 \
  --edited-asset-path /absolute/path/to/final.mp4
```

## Record immediately after finalize
- Recording path
- Duration
- Edited asset path
- Posted URL if publish already happened
- Notes on any visible setup assumption or edit constraint

## Verification
Capture is not closed out until `starter-kits/agent-launch-closeout-kit/launch-execution-log.md` shows a real recording path and no longer says `Status: pending capture`.