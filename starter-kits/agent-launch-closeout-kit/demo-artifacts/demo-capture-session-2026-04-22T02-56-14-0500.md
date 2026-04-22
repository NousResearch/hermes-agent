# Demo Capture Session — 2026-04-22 02:56 CDT

## Status
- Session state: ready to record
- Readiness packet: `starter-kits/agent-launch-closeout-kit/demo-artifacts/latest-demo-capture-readiness.md`
- Suggested raw recording path: `starter-kits/agent-launch-closeout-kit/demo-artifacts/raw-demo-capture-2026-04-22T02-56-14-0500.mov`
- Suggested edited asset path: `starter-kits/agent-launch-closeout-kit/demo-artifacts/edited-demo-capture-2026-04-22T02-56-14-0500.mp4`

## Record this exact path
1. Keep the claim narrow: closeout process only, not broader product proof.
2. Follow the one-screen trigger card in `starter-kits/agent-launch-closeout-kit/demo-trigger.md` (full detail remains in `starter-kits/agent-launch-closeout-kit/demo-capture-runbook.md`).
3. Capture the raw recording to `starter-kits/agent-launch-closeout-kit/demo-artifacts/raw-demo-capture-2026-04-22T02-56-14-0500.mov` or replace with your actual asset path.
4. After recording/editing, run:

```bash
bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture.sh --finalize \
  --recording-path /Users/hermesmasteragent/.hermes/hermes-agent/starter-kits/agent-launch-closeout-kit/demo-artifacts/raw-demo-capture-2026-04-22T02-56-14-0500.mov \
  --duration 00:01:19 \
  --edited-asset-path /Users/hermesmasteragent/.hermes/hermes-agent/starter-kits/agent-launch-closeout-kit/demo-artifacts/edited-demo-capture-2026-04-22T02-56-14-0500.mp4
```

## Surfaces to show during capture
- `starter-kits/agent-launch-closeout-kit/demo-artifacts/latest-demo-capture-readiness.md`
- `starter-kits/agent-launch-closeout-kit/demo-trigger.md`
- `starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md`
- `starter-kits/agent-launch-closeout-kit/launch-execution-log.md`
- `starter-kits/agentic-cron-orchestration-kit/launch/demo-outline.md`

## Raw preflight output
```
Demo capture preflight OK
Readiness report: starter-kits/agent-launch-closeout-kit/demo-artifacts/demo-capture-readiness-2026-04-22T02-56-14-0500.md
Canonical latest: starter-kits/agent-launch-closeout-kit/demo-artifacts/latest-demo-capture-readiness.md
```
