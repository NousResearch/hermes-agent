# Grunt-Offloaded App Review Pattern

Use this reference when the user wants an existing app reviewed against the idea-workflow framework while keeping orchestrator token use low.

## Purpose

Run a read-only existing-app review through a cheaper worker, produce idea-workflow artifacts in a separate workspace, and keep the user updated through Telegram or the originating channel.

## Proven workspace pattern

- Create a separate review workspace, e.g. `~/Hermes/FleetWorkspace/<app-slug>-review/`.
- Write a `worker-prompt.md` that includes:
  - known source/app path(s)
  - any related idea-note/doc path(s)
  - read-only constraints
  - output workspace path
  - required artifact list
  - required compact handoff report contract
  - a `status.txt` path workers must update at meaningful checkpoints
- Start the worker in the background with a wrapper script that logs to `grunt-run.log` and writes `DONE:` or `BLOCKED:` to `status.txt` on exit.
- For Telegram progress updates, create a temporary no-agent cron watcher script that only emits when `status.txt` changes. Remove that watcher when the worker completes.

## Artifact contract

For existing-app reviews, ask for:

- `README.md` — artifact map/status summary
- `01-app-review.md` — current app/source review
- `02-implementation-spec.md` — improvement plan and technical strategy
- `03-agent-build-handoff.md` — coding-agent/Superpowers-ready handoff for the next pass
- `04-spec-review.md` — PASS / PASS WITH CHANGES / FAIL readiness review
- `grunt-report.md` — compact handoff packet

## Worker prompt guardrails

Require the worker to:

- inspect source read-only
- avoid `.env`, secrets, private keys, credential stores, and generated/vendor folders
- avoid app mutation, commits, installs, and dependency changes unless explicitly approved
- mark assumptions and uncertainty
- report commands/files used, verification performed, blockers, and next action

## Orchestrator verification before final answer

Do not simply relay the worker verdict. Before presenting final recommendations:

1. Read `grunt-report.md` and `04-spec-review.md`.
2. Spot-check key factual claims against source with a cheap command, e.g. line counts, package scripts, component/test counts, key file existence.
3. Remove any temporary progress watcher/cron job.
4. Send a concise completion update with verdict, artifact path, verified key facts, and recommended next action.

## Pitfall this prevents

A background worker can finish silently or claim facts without evidence. The `status.txt` + watcher + orchestrator spot-check pattern keeps the user informed while preserving architect as the final quality gate.