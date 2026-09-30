# Durable & Background Systems

These systems run alongside the main conversation loop. Quick reference
here; full developer notes live in `AGENTS.md`, user-facing docs under
`website/docs/user-guide/features/`.

### Delegation (`delegate_task`)

Spawn a subagent with an isolated context + terminal session.

- **Single:** `delegate_task(goal, context)`.
- **Batch:** `delegate_task(tasks=[{goal, ...}, ...])` runs children in
  parallel, capped by `delegation.max_concurrent_children` (read the effective configuration).
- **Background:** `delegate_task(background=true)` returns a handle
  immediately and keeps the parent loop going; the child's result
  re-enters the conversation as a new turn when it finishes.
- **One-shot budget:** finite `hermes chat -q` sessions also have a total
  `delegation.oneshot_max_children` budget across calls; when exhausted, finish inline.
- **Roles:** `leaf` (default; cannot re-delegate) vs `orchestrator`
  (can spawn its own workers, bounded by `delegation.max_spawn_depth`).
- **Not durable.** A backgrounded child is still process-local — if the
  parent process exits, the child is lost. For work that must outlive
  the process, use a responsibility schedule. Background terminal processes
  are also process-local; they do not survive a runtime restart.

Config: `delegation.*` in `config.yaml`.

### Scheduled and event-driven work

Author responsibility packages using `../../responsibility-authoring/guide.md`.
Schedules are `schedules/*.yaml`; webhooks are `webhooks/*.yaml` inside the owning
package. The scheduler and gateway retain native execution and restart behavior.
Read the responsibility guide's schedules/webhooks references for declaration
formats, lifecycle, guards and delivery. Inspect or run jobs with the retained
`hermes cron` commands; do not create a second standalone job for a declaration.

### Background review

Native review cadence consolidates authored memory, responsibility knowledge and
service manuals. It does not author skills, send messages, delegate work or edit
schedule/webhook declarations. Read `references/memory-and-learning.md`.
