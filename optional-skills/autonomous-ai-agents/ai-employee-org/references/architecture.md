# AI Employee Org — Architecture

## Layer model

```text
Human / Webhook / Cron
        │
        ▼
   secretary (orchestrator profile)
        │  kanban_create / kanban_link / decomposer
        ▼
┌───────────────────────────────────────────┐
│  Kanban board (ai-company)                │
│  SQLite + dispatcher (gateway-owned)      │
└───────────────────────────────────────────┘
        │ spawns OS process per claim
        ├── job-recruiter
        ├── job-seeker
        ├── self-improver
        └── delivery-worker
```

## When to use which primitive

| Primitive | Lifetime | Identity | Best for |
|-----------|----------|----------|----------|
| `delegate_task` sync | Parent turn | Anonymous subagent | Quick reasoning fork |
| `delegate_task(background=true)` | Until done; new turn on complete | Anonymous | Parallel research inside a worker |
| Kanban task | Durable; reclaim on crash | Named profile + memory | Cross-role handoffs |
| `cronjob` | Scheduled | Profile session | Daily scans, hygiene |
| `terminal(background=true)` | Process lifetime | Shell | Long builds, servers |

## Typical flows

### Inbound contract (受注)

1. Secretary triages → `delivery-worker` task with acceptance criteria in body.
2. Delivery worker may `delegate_task(background=true)` for parallel research.
3. `kanban_complete` with `metadata.changed_files` and deliverable paths.
4. Secretary notifies human via gateway platform.

### Recruiting loop (求人)

1. Cron or secretary creates `job-recruiter` task with role spec.
2. Recruiter drafts posting → `dir:` workspace under `~/ops/recruiting/`.
3. Comments thread holds approval; human unblocks if `kanban_block`.

### Job search loop (求職)

1. `job-seeker` cron scans sources (idempotency key per listing URL).
2. Creates child tasks for applications needing human approval.
3. Secretary aggregates weekly pipeline comment on parent epic.

### Self-improvement (自己改善)

1. `self-improver` runs `hermes curator status` and reviews `agent.log` patterns.
2. Proposes skill patches; never deletes pinned skills.
3. Files report under `_docs/` with date prefix.

## Config anchors

```yaml
kanban:
  orchestrator_profile: secretary
  dispatch_in_gateway: true
  failure_limit: 2

delegation:
  max_async_children: 5
  max_concurrent_children: 3
  max_spawn_depth: 2
  orchestrator_enabled: true

auxiliary:
  kanban_decomposer:
    provider: ""   # inherit or pin cheap model for decomposition
```

## Multi-gateway note

If you run `hermes -p job-seeker gateway run` alongside default, only default
keeps `dispatch_in_gateway: true`. Worker gateways deliver messages only.
