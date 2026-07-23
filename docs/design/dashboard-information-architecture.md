# Dashboard Information Architecture

The current dashboard problem is not only visual quality. The deeper issue is that dashboards have been accumulating tabs faster than they have accumulated a shared operating structure.

This document defines the shared Hermes/TLC dashboard information architecture.

## The Six Workspaces

### Command

Question: What needs attention now?

Use for:

- priority alerts
- live status summary
- action queue
- decisions needed
- owner handoffs

Avoid placing every metric here. Command is the top of the operating stack, not the whole dashboard.

### Operations

Question: What is running, blocked, stale, expensive, or failing?

Use for:

- queues
- schedulers
- deployments
- system health
- incidents
- logs and traces
- job freshness

### Intelligence

Question: What have we learned?

Use for:

- research reports
- findings
- evidence
- confidence
- recommendations
- segment/tag learning
- topic or market opportunity ranking

### Capacity

Question: What are we spending, consuming, scanning, generating, or storing?

Use for:

- API calls
- token use
- provider cost
- storage growth
- throughput
- rate limits
- budget breakers

### Projects

Question: How is each business unit doing?

Use for:

- readiness status
- project health
- business-unit blockers
- dashboard coverage
- deployment posture
- owner accountability

### Controls

Question: What can I start, stop, approve, tune, or deploy?

Use for:

- autopilot controls
- capacity controls
- deployment actions
- approval gates
- kill switches
- configuration changes

Controls must have audit feedback and clear permission/risk treatment.

## Dashboard Collapse Rule

If a dashboard has more than eight sidebar items, it should be reviewed for collapse into the six workspace model.

Example:

| Old Pattern | New Workspace |
| --- | --- |
| Activity, Run Monitor, System | Operations |
| Coverage, Market Data, Research Intelligence, Learning Ledger | Intelligence |
| Cost, Cost Intelligence, Persistence | Capacity |
| Command, Autopilot | Command or Controls depending on action level |
| Portfolio KPIs | Projects or Command depending on scope |

## First-Screen Rule

The first screen should answer three questions without requiring tab hopping:

1. Is anything broken or blocked?
2. Is the system doing useful work?
3. What should the operator do next?

Everything else can be a drilldown.

## Drilldown Rule

A drilldown page is justified when:

- it supports investigation
- it has a table, chart, or timeline that would crowd the command surface
- it needs filters
- it provides evidence behind a recommendation

Do not create a drilldown just because a feature exists.

## Retrofit Order

1. Kashi VC: market cartography, experiments, scheduler, cost, and research intelligence.
2. Media Engine: generation health, brand operations, social channel readiness, cost, and assets.
3. Media Business Operations: executive rollup and publishing performance.
4. Hermes OS: cross-project readiness and production rails.
5. TLC Capital Group OS: business-unit command and portfolio readiness.

