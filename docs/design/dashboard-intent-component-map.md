# Dashboard Intent To Component Map

This map lets Hermes/Codex choose the right dashboard primitive before writing UI.

| Dashboard Intent | Data Shape | Component Pattern | Notes |
|---|---|---|---|
| Operating overview | Small set of status metrics | `MetricGrid` + `KpiCard` + `HealthBadge` | Use for first-screen summaries. |
| Command center | User-triggered actions | `CommandBar` + `ActionButtonGroup` | Separate read-only data from operator commands. |
| Work queue | Ordered tasks or jobs | `QueuePanel` or `DataTable` | Use queue for short action lists; table for dense records. |
| Event history | Timestamped events | `ActivityTimeline` + `AuditEventList` | Include actor, action, status, and entity. |
| Record table | Rows with sortable columns | `DataTable` | Use `TableToolbar`, `FilterBar`, and `SearchInput` when filtering matters. |
| Trend chart | Time-series values | `ChartPanel` + `SimpleLineChart` | Keep chart colors token-driven. |
| Category comparison | Buckets or grouped counts | `SimpleBarChart` or `HeatmapGrid` | Use heatmaps only when the matrix conveys value at a glance. |
| Insight summary | Finding, risk, recommendation | `InsightPanel`, `FindingCard`, `RecommendationCard` | Include confidence/evidence when available. |
| Cross-project scorecard | Business/project rollup | `ExecutiveProjectScorecard` | Do not duplicate full project dashboards here. |
| CEO-level command | Portfolio-level summary | `ExecutiveHealthRollup`, `ExecutiveActionQueue`, `ExecutiveCostCapacityRollup` | Use to summarize, prioritize, and route follow-up. |
| Missing data | No records yet | `DashboardEmptyState` | Tell the operator what will make data appear. |
| Loading data | Request in progress | `DashboardLoadingState` | Avoid custom spinners. |
| Failed data | Request or dependency failure | `DashboardErrorState` | Include actionable failure text. |

## Selection Rules

- Use a table when the operator must compare more than five rows.
- Use cards when the operator must scan status quickly.
- Use a chart only when movement or distribution matters.
- Use executive components only for cross-project rollups.
- Keep production actions out of summary cards; route them through command/action patterns.
