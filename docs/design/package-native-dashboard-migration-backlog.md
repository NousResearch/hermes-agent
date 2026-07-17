# Package-Native Dashboard Migration Backlog

This backlog tracks the work required to move static adapter dashboards into full `@hermes/dashboard-kit` React implementations.

V3 does not require this rewrite. V3 proves the system through Hermes OS plus static adapters. Package-native migration belongs to later hardening once APIs, deployment rails, and dashboard contracts are stable.

## Khashi VC ROC

Current state:

- Static HTML and vanilla JavaScript dashboard.
- Uses `hdk-*` static adapter classes.
- Keeps existing auth, API calls, scheduler controls, experiment views, and operational workflows.

Package-native target:

- React dashboard shell using `DashboardShell`, `DashboardSidebar`, and `DashboardLauncher`.
- Command center using `CommandBar` and explicit command safety props.
- Experiments, runs, queues, market data, persistence, activity, cost, and system views using `DataTable`, `MetricGrid`, `KpiCard`, `StatusPill`, `RunStatusPanel`, and `ChartPanel`.
- Heatmaps using `HeatmapGrid`.
- Findings and knowledge views using `InsightPanel`, `FindingCard`, and `RecommendationCard`.
- Query/data layer using the eventual TanStack Query standard.

Recommended phases:

- K1: Extract current ROC API client from `public/roc/app.js`.
- K2: Build React shell and routing while embedding existing views behind a compatibility adapter.
- K3: Convert summary, command, and run monitor views first.
- K4: Convert table-heavy views.
- K5: Convert research intelligence, cost intelligence, and findings views.
- K6: Remove static compatibility adapter after feature parity.

## Media Engine Ops

Current state:

- Generated static HTML from `core/operations/unified-publishing-dashboard.js`.
- Uses `hdk-*` static adapter classes.
- Keeps autopilot controls, dashboard registry, generated dashboard output, Discord delivery, approval, publishing, storage, and production-run behavior.

Package-native target:

- React app or embedded React dashboard route using `@hermes/dashboard-kit`.
- Server exposes dashboard JSON separately from HTML rendering.
- Autopilot controls use `CommandBar` with permission/risk metadata.
- Brand health and generation status use `MetricGrid`, `KpiCard`, `StatusPill`, and `RunStatusPanel`.
- Generation history, approvals, publishing events, search console issues, and production runs use `DataTable`.
- Storage and pruning use `CapacityMeter`, `InsightPanel`, and `RecommendationCard`.
- Discord output and internal logs remain separate sections.

Recommended phases:

- M1: Split snapshot JSON API from generated HTML concerns.
- M2: Add package-native React dashboard shell consuming the same snapshot.
- M3: Convert operations and autopilot controls.
- M4: Convert tables and queues.
- M5: Convert storage, Discord output, and approval views.
- M6: Retire generated static HTML once deployed React dashboard reaches parity.

## Completion Criteria

- Static adapters remain drift-free until each project is fully migrated.
- Package-native routes have Playwright visual coverage.
- Each converted dashboard preserves existing auth, APIs, commands, and production behavior.
- Static HTML renderers are removed only after live parity is proven.
