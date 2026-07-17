# V7 Dashboard Layout Recipes

V7 turns the Hermes dashboard system from a component kit into a set of approved full-page layouts. These recipes are the default starting point for Codex, Hermes, and human developers when building or rebuilding TLC dashboards.

Status: `[x]` Complete

## Goal

Make dashboard work less improvised. A new dashboard should start from a known operating pattern, not a blank page or a generic admin template.

## Completion Criteria

- [x] Eight approved full-page dashboard recipes exist.
- [x] Each recipe defines purpose, layout anatomy, required data contract, required components, and validation.
- [x] The design-system gallery exposes the recipe catalog.
- [x] The dashboard scaffolder accepts a `--recipe` option.
- [x] A V7 validator confirms the recipe IDs exist in docs, gallery data, and scaffolder support.

## Approved Recipes

### 1. Executive Command Center

Use for: TLC/Hermes top-level rollups across companies, projects, dashboards, and action queues.

Layout anatomy:

- Header with operating period, global health, and primary escalation action.
- KPI strip for portfolio health, cost, capacity, throughput, and risk.
- Domain tabs for media, research, operations, infrastructure, and finance.
- Action-needed queue with owner, severity, age, and next step.
- Project scorecards with health, trend, blockers, and launch links.
- Evidence and recommendation column for executive decisions.

Required data contract:

- `projects[]` with health, owner, category, status, cost, capacity, throughput, and launch URLs.
- `actions[]` with severity, owner, due date, source dashboard, and command availability.
- `domains[]` with score, trend, and linked projects.

Required components:

- `DashboardHeader`
- `MetricGrid`
- `KpiCard`
- `StatusPill`
- `ProjectSwitcher`
- `DashboardLauncher`
- `DataTable`
- `InsightPanel`
- `RecommendationCard`

Validation:

- Must show at least one action-needed state.
- Must distinguish unknown, healthy, warning, and critical project health.
- Must link back to source project dashboards.

### 2. Operations Control Room

Use for: schedulers, workers, queues, autopilot systems, market polling, production commands, and run capacity.

Layout anatomy:

- Header with live/offline state and operator permission level.
- Command bar separated from analytics.
- Capacity row for running, queued, stalled, failed, and available capacity.
- Queue timeline and run monitor table.
- Worker/system health panel.
- Recent audit events.

Required data contract:

- `commands[]` with permission, risk level, disabled reason, and confirmation requirement.
- `capacity` with current, floor, ceiling, dynamic limit, and pressure reason.
- `runs[]` with status, age, owner, freshness, and retry/stop affordances.
- `workers[]` with heartbeat, lag, and assigned queue.

Required components:

- `CommandBar`
- `ActionButtonGroup`
- `CapacityMeter`
- `RunStatusPanel`
- `QueuePanel`
- `ActivityTimeline`
- `AuditEventList`
- `DataTable`

Validation:

- Dangerous commands must declare `riskLevel`.
- Disabled commands must explain why.
- Queue, run, and worker states must show stale and failed states.

### 3. Research Intelligence Dashboard

Use for: findings, evidence, experiments, promoted knowledge, tag/category research, and strategy readiness.

Layout anatomy:

- Header with research objective and evidence freshness.
- Evidence KPI strip.
- Finding cards grouped by confidence and status.
- Coverage heatmap for categories/tags/time windows.
- Experiment results table with filters.
- Recommendation panel with next research action.

Required data contract:

- `findings[]` with confidence, evidence count, freshness, source runs, and promotion state.
- `coverage[]` with category, tag, bucket, count, and quality score.
- `experiments[]` with hypothesis, status, outcome, and linked evidence.

Required components:

- `InsightPanel`
- `FindingCard`
- `RecommendationCard`
- `HeatmapGrid`
- `ChartPanel`
- `FilterBar`
- `SegmentedControl`
- `DataTable`

Validation:

- Must separate observation from recommendation.
- Must show confidence and evidence count.
- Must include a coverage or blind-spot view.

### 4. Pipeline Workflow Dashboard

Use for: Media Engine generation pipelines, approvals, publishing, Discord delivery, Search Console work, and failed package review.

Layout anatomy:

- Header with production window and autopilot state.
- Brand/job health metrics.
- Pipeline lane view for planned, generating, reviewing, approved, delivered, failed.
- Approval queue table.
- Discord/public output panel separated from internal logs.
- Failure review section with reason, artifact links, and retry guidance.

Required data contract:

- `jobs[]` with brand, format, stage, status, artifact URLs, and failure reason.
- `approvals[]` with owner, age, risk, and publish destination.
- `outputs[]` with Discord message, download links, and production URL.

Required components:

- `RunStatusPanel`
- `QueuePanel`
- `DataTable`
- `ActivityTimeline`
- `StatusPill`
- `CommandBar`
- `DashboardSection`
- `DashboardEmptyState`

Validation:

- Must keep internal logs out of the public output section.
- Must show fallback/failure reason when no deliverable package exists.
- Must show brand-level enabled/disabled state.

### 5. Cost And Capacity Dashboard

Use for: token spend, model/vendor usage, API calls, storage growth, CPU pressure, queue cost, and budget enforcement.

Layout anatomy:

- Header with selected range and budget posture.
- Range toggle for 7/30/90 days.
- Cost/capacity KPI strip.
- Trend charts for tokens, API calls, storage, and CPU.
- Breakdown table by project, model, vendor, operation, and day.
- Budget risk/recommendation column.

Required data contract:

- `usageSeries[]` with date, metric, project, operation, and amount.
- `budgets[]` with limit, actual, overage, enforcement mode, and reset window.
- `resources[]` with storage, CPU, API calls, and queue pressure.

Required components:

- `DateRangeToggle`
- `MetricGrid`
- `KpiCard`
- `CapacityMeter`
- `ChartPanel`
- `SimpleLineChart`
- `SimpleBarChart`
- `DataTable`
- `RecommendationCard`

Validation:

- Range controls must actually change the dataset.
- Budget overage must explain whether it is hard-blocking, soft-warning, or informational.
- Storage and external API usage must not be hidden under token cost.

### 6. Market Asset Explorer

Use for: Kashi markets, investing assets, stocks, ETFs, categories, tags, liquidity, open interest, close windows, and watchlists.

Layout anatomy:

- Header with universe, freshness, and filter summary.
- Filter rail for category, tag, liquidity, close window, status, and source.
- Heatmap or matrix for category/tag coverage and current activity.
- Dense result table.
- Detail drawer/panel for selected market or asset.
- Research/experiment action bar.

Required data contract:

- `assets[]` with id, title, category, tags, liquidity, open interest, close time, status, and source URL.
- `coverage[]` with category/tag counts, active experiments, completed experiments, and blind spots.
- `selectedAsset` with snapshots, linked runs, and evidence.

Required components:

- `FilterBar`
- `SearchInput`
- `SegmentedControl`
- `DateRangeToggle`
- `HeatmapGrid`
- `DataTable`
- `ChartPanel`
- `CommandBar`

Validation:

- Must expose tags/subcategories, not only categories.
- Must show data freshness.
- Must distinguish active, closing soon, closed, and unknown markets.

### 7. Brand Business Performance Dashboard

Use for: media brands, content output, channel analytics, engagement, posting consistency, and business performance.

Layout anatomy:

- Header with brand selector and active channels.
- Output consistency metrics.
- Content calendar or production cadence table.
- Channel performance charts.
- Brand health cards with blockers and next scheduled jobs.
- Recommendation panel for what to post or fix next.

Required data contract:

- `brands[]` with enabled state, cadence, backlog, last output, and health.
- `channels[]` with platform, audience, engagement, clicks, and publishing status.
- `content[]` with type, status, schedule, and artifact URLs.

Required components:

- `ProjectSwitcher`
- `MetricGrid`
- `KpiCard`
- `ChartPanel`
- `SimpleLineChart`
- `DataTable`
- `StatusPill`
- `RecommendationCard`

Validation:

- Must show enabled/disabled brand state.
- Must separate content production from channel performance.
- Must make missed cadence visible.

### 8. System Health And Deployment Dashboard

Use for: production deployments, CI failures, dashboard manifests, service health, environment variables, DNS/routes, and release readiness.

Layout anatomy:

- Header with current environment and deploy status.
- Service health cards.
- Deployment timeline.
- CI/check table.
- Manifest/registry health section.
- Environment/secrets readiness panel.
- Action bar for promote, rollback, restart, and verify commands.

Required data contract:

- `services[]` with URL, health URL, status, version, uptime, and owner.
- `deployments[]` with commit, actor, status, duration, and rollback target.
- `checks[]` with CI status, annotations, and required/failing distinction.
- `secrets[]` with presence state only, never secret values.

Required components:

- `DashboardLauncher`
- `HealthBadge`
- `ActivityTimeline`
- `DataTable`
- `CommandBar`
- `AuditEventList`
- `StatusPill`
- `DashboardErrorState`

Validation:

- Must never print secret values.
- Must distinguish production, local, and unknown URLs.
- Must show the next verification action when a deploy is incomplete.

## V7 Build Guidance

When creating a dashboard:

1. Pick exactly one recipe as the primary layout.
2. Add secondary recipe patterns only when the dashboard truly crosses domains.
3. Fill the recipe's required data contract before polishing visuals.
4. Use `@hermes/dashboard-kit` components first.
5. Add a Playwright check for the chosen recipe's required states.
6. Run `npm run dashboard:v7:validate`.

## Next Track

V8 should focus on package-native migrations for the highest-value dashboards:

- Media Engine Ops first.
- Khashi VC ROC second.
- Hermes Executive Control Plane third.
