# Goal-to-Kanban Routing Pattern

Use this when converting strategic objectives, `/goal` flows, or architecture decisions into tracked Kanban work without bypassing governance.

## Durable pattern

1. Keep normal `/goal <text>` session-local unless Sahil explicitly asks for routed work or the command is `/goal route`.
2. Route explicit `/goal route <text>` into Kanban as a parent/triage task, not as immediate execution.
3. Assign routed parent goals to `orchestrator` first. The Orchestrator decomposes into sub-goals/tasks before workers execute.
4. Use Kanban status `backlog` for newly routed goals unless the board explicitly supports a stronger ready state. Do not invent unsupported statuses such as `ready`.
5. Include acceptance criteria in the task body, even if they are provisional.
6. Include open risks explicitly in the task body; do not treat known broken/unverified areas as resolved just because they are out of scope for the current implementation.
7. Webhook/event-driven inputs should create or update Kanban tasks first. They should not directly trigger destructive changes, publishing, provider/auth edits, service deactivation, or profile deletion.
8. Keep profile cleanup language audit-first: classify, reconcile, get approval, then retire/archive. Avoid direct "scrub/delete" wording unless Sahil has explicitly approved destructive cleanup.

## Verification checklist

- Unit tests cover command parsing/alias behaviour.
- Integration test creates a real Kanban task where safe.
- Created task has the intended assignee, status, title, body sections, and triage marker.
- Specialist gateways remain non-dispatching; Kensei owns cron/Kanban dispatch.
- Gateway restart is only done when approved or explicitly requested.
- Final report separates implemented behaviour from open risks.

## Common pitfalls

- Treating a strategic goal as executable work and dispatching a worker immediately.
- Creating child tasks before the parent goal is traceable.
- Claiming knowledge accumulation is solved by memory alone; structured wiki/runbook/provenance remains a separate capability.
- Claiming webhook autonomy exists without live webhook ingress, security checks, and Kanban-first routing.
- Hiding fleet hygiene risk because a small SOUL/config gap was fixed; profile sprawl and lifecycle governance can still remain open.
