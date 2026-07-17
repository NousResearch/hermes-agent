# Dashboard Component Catalog

This catalog is the short operating reference for Hermes/TLC dashboard work. Use it with `hermes-dashboard-design-contract.md` and the `/design-system` gallery.

## Shell And Page Structure

Use `DashboardShell`, `DashboardSidebar`, `DashboardHeader`, `DashboardMain`, `DashboardSection`, and `DashboardPageTitle` for all dashboard pages.

Use these when:

- A page is an operating dashboard, command center, project dashboard, or executive rollup.
- Navigation, page title, actions, and content need consistent placement.

Avoid:

- Project-local page frames.
- Nested cards around entire page sections.
- Custom sidebar systems inside dashboard pages.

## Metrics And Status

Use `MetricGrid`, `KpiCard`, `StatusPill`, `HealthBadge`, `ProgressMetric`, `CapacityMeter`, and `TrendDelta`.

Use these when:

- The value is a count, health state, capacity limit, trend, or operational indicator.
- The dashboard needs stable sizing while values change.

Avoid:

- One-off KPI cards.
- Color-only state without a text label.
- Hero-scale type inside compact operational panels.

## Tables And Filters

Use `DataTable`, `TableToolbar`, `FilterBar`, `SearchInput`, `SegmentedControl`, and `DateRangeToggle`.

Use these when:

- Rows need sorting, scanning, filtering, or row click/detail behavior.
- The data is dense enough that cards would reduce readability.

Avoid:

- Custom table markup.
- Filters detached from the table they control.
- Tables without empty, loading, and error states.

## Charts And Insights

Use `ChartPanel`, `SimpleBarChart`, `SimpleLineChart`, `HeatmapGrid`, `InsightPanel`, `FindingCard`, and `RecommendationCard`.

Use these when:

- The dashboard needs trend, distribution, coverage, evidence, recommendation, or risk presentation.
- A finding needs evidence/confidence context.

Avoid:

- Decorative charts that do not support a decision.
- Unlabeled color scales.
- Recommendations without confidence or action text.

## Launcher And Registry

Use `DashboardLauncher`, `ProjectSwitcher`, `DashboardRegistryEntry`, and `useDashboardHealth`.

Use these when:

- A dashboard needs to list other dashboards.
- Local and production URLs need to coexist.
- Health state should be shown from manifest data or a health endpoint.

Avoid:

- Hard-coded sidebar links that bypass the dashboard registry.
- Silent missing manifests or missing launch URLs.
- Health status that cannot distinguish unknown, offline, missing, and current.

## Commands And Queues

Use `CommandBar`, `ActionButtonGroup`, `ActivityTimeline`, `QueuePanel`, `RunStatusPanel`, and `AuditEventList`.

Use these when:

- The page includes operator actions.
- Work can be queued, running, completed, failed, blocked, or stale.
- Actions need permission, disabled reason, confirmation, or risk treatment.

Avoid:

- Placing destructive actions beside read-only analytics without visual separation.
- Disabled buttons without a reason.
- Command surfaces without audit or queue feedback.

## Required State Coverage

Every new dashboard must account for:

- Normal data.
- Loading.
- Empty.
- Error.
- Warning.
- Critical.
- Disabled or permission-limited.
- Compact/mobile layout.

## Validation

Before calling a dashboard complete:

- Run `npm run dashboard:usage:audit:strict`.
- Run `npm run dashboard:registry:validate`.
- Run `npm run dashboard:visual:check` when visual surfaces changed.
- Confirm the `/design-system` gallery still renders the components used by the dashboard.
