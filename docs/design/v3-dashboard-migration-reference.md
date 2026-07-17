# V3 Dashboard Migration Reference

This document records the first real dashboard migrations for the Hermes dashboard design system.

V3 has two purposes:

1. Prove the kit on the Hermes OS dashboard inside `nous-hermes-agent`.
2. Define exact migration maps for Khashi VC and Media Engine before V4 extracts the kit into a shared package.

## Hermes OS Reference Migration

Status: complete inside `web/src/pages/HermesOsPage.tsx`.

Current usage:

- `DashboardHeader` for the page header.
- `MetricGrid` and `KpiCard` for score, work graph, runtime, assignments, tasks, templates, and dry-run metrics.
- `DataTable` for task backlog.
- `InsightPanel` and `StatusPill` for architecture gaps.
- `ChartPanel` and `SimpleBarChart` for agent assignment distribution.
- `DashboardSection` for runtime modules.
- `DashboardLoadingState` and `DashboardErrorState` for route-level data states.

Remaining refinement before V5 visual QA:

- Add screenshot coverage for `/hermes-os`.
- Add route-level visual regression cases for loading, empty, and error states.

## Khashi VC ROC Migration Map

Project path: `/Users/hq/Workspace/projects/khashi-vc`

Current architecture:

- TypeScript ROC services generate dashboard contracts under `src/roc`.
- The web dashboard is not a React app yet.
- Existing plan notes say the current frontend renders many panes through generic object rendering.

V3 decision:

- Do not rewrite Khashi into React inside V3.
- Preserve current auth, APIs, scheduler behavior, and ROC contracts.
- Migrate the visual layer after V4 extracts `hermes-dashboard-kit` into a consumable package.

Required component mapping:

| Current Khashi Surface | Required Kit Primitive |
|---|---|
| ROC shell/sidebar/dashboard launcher | `DashboardShell`, `DashboardSidebar`, `DashboardLauncher` |
| Command center actions | `CommandBar`, `ActionButtonGroup` |
| Active investigations / experiments / buckets / queue cards | `MetricGrid`, `KpiCard`, `RunStatusPanel` |
| Heatmap matrix | `ChartPanel`, `HeatmapGrid` |
| Cost and operations intelligence | `MetricGrid`, `ChartPanel`, `InsightPanel` |
| Persistence, market data, activity, system tables | `DataTable`, `FilterBar`, `SearchInput`, `DateRangeToggle` |
| Knowledge/finding evidence | `InsightPanel`, `FindingCard`, `RecommendationCard` |
| Loading/unavailable/aborted API states | `DashboardLoadingState`, `DashboardErrorState`, `DashboardEmptyState` |

Migration acceptance:

- Existing ROC endpoints stay stable.
- The dashboard launcher shows Khashi VC, Media Engine, and Hermes Agent.
- The Khashi dashboard keeps all current operational controls.
- Generic object rendering is replaced with purpose-built kit surfaces for command, experiments, coverage, market data, cost, persistence, and system health.

## Media Engine Ops Migration Map

Project path: `/Users/hq/Workspace/projects/media-engine`

Current architecture:

- The ops dashboard is generated from `core/operations/unified-publishing-dashboard.js`.
- The dashboard server is launched by `npm run ops:dashboard:server`.
- It already has autopilot start/stop/capacity controls and a dashboard registry panel.

V3 decision:

- Do not rewrite the static generated dashboard in V3.
- Preserve current Discord, autopilot, generation, storage, and publishing behavior.
- Migrate the visual layer after V4 extracts `hermes-dashboard-kit` or provide a static-compatible style/token adapter if React is not adopted.

Required component mapping:

| Current Media Engine Surface | Required Kit Primitive |
|---|---|
| Operator control plane shell | `DashboardShell`, `DashboardHeader`, `DashboardSection` |
| Dashboard registry list | `DashboardLauncher`, `ProjectSwitcher` |
| Autopilot start/stop/capacity controls | `CommandBar`, `ActionButtonGroup`, `CapacityMeter` |
| Brand generation health/status | `MetricGrid`, `KpiCard`, `StatusPill`, `RunStatusPanel` |
| Generation history | `DataTable`, `FilterBar`, `DateRangeToggle` |
| Asset storage/pruning panel | `CapacityMeter`, `InsightPanel`, `RecommendationCard` |
| Discord output vs internal logs | `DashboardSection`, `ActivityTimeline`, `AuditEventList` |
| Approval queue and blockers | `QueuePanel`, `ActivityTimeline`, `StatusPill` |

Migration acceptance:

- Media Engine clearly shows autopilot status, enabled brands, due jobs, allowed jobs, capacity, storage usage, approvals, blockers, and generation history.
- Human-facing Discord output is separated from internal operational logs.
- Controls remain available from dashboard and Discord.

## Registry Updates

The Hermes Agent dashboard registry now includes:

- `nous-hermes-agent.dashboard`
- `khashi-vc.roc`
- `media-engine.ops`

This makes the V3 migration target visible before V4 shared-package extraction.
