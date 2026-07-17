# Hermes Dashboard Design Contract

This contract defines the default UI rules for Hermes/TLC dashboards. New dashboard work should use the dashboard kit first and only add local UI when the kit does not cover a real product need.

## Required Primitives

- Page frames use `DashboardShell`.
- Left navigation uses `DashboardSidebar`.
- Page headings and actions use `DashboardHeader`.
- Metric summaries use `MetricGrid` and `KpiCard`.
- Operational state uses `StatusPill`, `HealthBadge`, `ProgressMetric`, and `CapacityMeter`.
- Tabular data uses `DataTable`.
- Filters use `FilterBar`, `SearchInput`, `SegmentedControl`, and `DateRangeToggle`.
- Charts live inside `ChartPanel`.
- Research output uses `InsightPanel`, `FindingCard`, and `RecommendationCard`.

## Layout Rules

- Dashboard pages should be dense, scannable, and operational.
- Avoid landing-page hero sections inside operational dashboards.
- Do not nest cards inside cards.
- Prefer full-width page bands and grid sections over decorative floating panels.
- Use responsive grid constraints so metric cards, tables, and charts do not resize unpredictably.
- Mobile layouts must preserve the primary task before secondary analytics.

## Visual Rules

- Use theme tokens and kit components instead of hardcoded project-local color systems.
- Use status colors only to communicate state: healthy, warning, critical, neutral, unknown.
- Keep typography compact inside dashboard panels.
- Do not use viewport-scaled font sizes.
- Do not add decorative gradient orbs, bokeh, or one-off backgrounds.

## Interaction Rules

- Buttons must represent commands.
- Binary state should use switches or explicit status controls.
- Lists and tables should expose loading, empty, and error states.
- Dangerous actions must be visually distinct and require an explicit confirmation pattern.
- Icon-only controls require accessible labels.

## Migration Rules

- Replace local dashboard shells before replacing individual cards.
- Replace repeated local metric cards with `KpiCard`.
- Replace local table implementations with `DataTable`.
- Replace custom chart chrome with `ChartPanel`.
- Preserve project-specific data contracts and API behavior during migration.

## Agent Rules

- Codex/Hermes should not create new ad hoc dashboard primitives unless the kit lacks the required behavior.
- When the kit lacks a behavior needed by two or more dashboards, extend the kit first.
- When building a new dashboard, start from `DashboardShell`, then add metrics, tables, charts, and insights from the kit.
