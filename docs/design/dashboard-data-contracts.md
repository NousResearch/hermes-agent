# Hermes Dashboard Data Contracts

The dashboard kit now owns a shared data language for TLC/Hermes dashboards. This is the layer Mobbin cannot replace: Mobbin can help us see stronger interface patterns, but every dashboard still needs trusted data contracts before the UI can tell the truth.

Canonical exports:

```ts
import {
  type DashboardSnapshotContract,
  type DashboardModuleContract,
  type DashboardDataSourceState,
  type DashboardMetricContract,
  type DashboardAlertContract,
  type DashboardProjectStatusContract,
  type DashboardCostContract,
  type DashboardSystemHealthContract,
  type DashboardReadinessSnapshotContract,
  summarizeDashboardSnapshot,
  assessDashboardArchitecture,
} from "@hermes/dashboard-kit";
```

## Required Dashboard Language

Every operational dashboard should be able to report:

| Contract | Purpose |
| --- | --- |
| `DashboardSnapshotContract` | One project-level snapshot that can be consumed by Hermes OS or TLC OS. |
| `DashboardModuleContract` | One dashboard module, page, or major panel mapped into a shared workspace. |
| `DashboardDataSourceState` | Data source owner, endpoint, freshness, status, and failure mode. |
| `DashboardMetricContract` | Numeric or labeled metrics with optional time windows. |
| `DashboardAlertContract` | Open risks, blockers, degraded systems, or acknowledged exceptions. |
| `DashboardDecisionContract` | Operator/business decisions with evidence and revisit dates. |
| `DashboardCostContract` | Provider, project, usage, budget, and cost posture. |
| `DashboardSystemHealthContract` | Runtime health signals such as CPU, memory, queue depth, and error rate. |
| `DashboardReadinessSnapshotContract` | Project readiness state that can roll into TLC OS. |

## Standard Severity

Use the same severity vocabulary everywhere:

```text
healthy
watch
degraded
blocked
unknown
```

Do not invent local synonyms like `good`, `bad`, `needs attention`, `red`, or `ok`. Local dashboards can render friendlier labels, but exported contracts should stay normalized.

## Six Workspace Model

Every module belongs to exactly one workspace:

| Workspace | Question |
| --- | --- |
| `command` | What needs attention now? |
| `operations` | What is running, blocked, stale, expensive, or failing? |
| `intelligence` | What have we learned? |
| `capacity` | What are we spending, consuming, scanning, generating, or storing? |
| `projects` | How is each business unit doing? |
| `controls` | What can I start, stop, approve, tune, or deploy? |

This model exists so Kashi, Media Engine, Media Business Operations, Hermes OS, and TLC OS do not each grow a different dashboard structure.

## Implementation Rule

Before redesigning a dashboard:

- [ ] Produce or update a `DashboardSnapshotContract`.
- [ ] Map existing tabs/pages into the six workspace model.
- [ ] Mark each data source with owner, endpoint/file, freshness, and failure mode.
- [ ] Convert raw metrics into explicit `healthy`, `watch`, `degraded`, `blocked`, or `unknown`.
- [ ] Use `assessDashboardArchitecture()` to identify remaining gaps.
- [ ] Use Mobbin only after the information architecture and data contract are understood.

## Example Snapshot

```ts
const snapshot = {
  id: "khashi-vc-dashboard-2026-07-23",
  projectId: "khashi-vc",
  generatedAt: new Date().toISOString(),
  modules: [
    {
      id: "market-cartography",
      label: "Market Cartography",
      workspace: "intelligence",
      route: "/coverage",
      status: "watch",
      primaryQuestion: "Which categories and tags have persistent liquid market activity?",
      dataSources: [
        {
          id: "kalshi-scan-rollups",
          label: "Kalshi scan rollups",
          owner: "khashi-vc",
          endpoint: "/api/roc/market-cartography",
          freshnessSeconds: 300,
          status: "healthy",
        },
      ],
    },
  ],
  metrics: [],
  alerts: [],
  activity: [],
};
```

