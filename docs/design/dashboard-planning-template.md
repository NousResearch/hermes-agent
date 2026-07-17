# Dashboard Planning Template

Use this before creating or changing any Hermes/TLC dashboard.

## Dashboard Identity

- Dashboard name:
- Project:
- Owner:
- Category:
- Route:
- Manifest id:
- Local URL:
- Production URL:

## User Workflow

- Primary operator:
- Primary decision this dashboard supports:
- Top three tasks:
- Read-only views:
- Operator actions:
- Destructive or approval-gated actions:

## Data Model

List each data source before UI work starts.

| Source | Endpoint/File | Freshness | Failure State | Owner |
|---|---|---:|---|---|
|  |  |  |  |  |

Required entities:

- Entity:
  - Required fields:
  - Optional fields:
  - Empty state:
  - Loading state:
  - Error state:

## Component Plan

Use only approved dashboard-kit primitives unless the design contract explicitly allows an exception.

| Intent | Required Component |
|---|---|
| Page frame | `DashboardShell`, `DashboardSidebar`, `DashboardHeader`, `DashboardMain` |
| KPI/status cards | `MetricGrid`, `KpiCard`, `HealthBadge`, `StatusPill` |
| Tables/lists | `DataTable`, `TableToolbar`, `FilterBar`, `SearchInput` |
| Charts | `ChartPanel`, `SimpleLineChart`, `SimpleBarChart`, `HeatmapGrid` |
| Empty/loading/error | `DashboardEmptyState`, `DashboardLoadingState`, `DashboardErrorState` |
| Operator commands | `CommandBar`, `ActionButtonGroup`, `QueuePanel`, `RunStatusPanel` |
| Executive rollup | `ExecutiveHealthRollup`, `ExecutiveProjectScorecard`, `ExecutiveActionQueue`, `ExecutiveCostCapacityRollup`, `ExecutiveDomainTabs` |

## Layout Plan

- Desktop layout:
- Mobile layout:
- Sidebar behavior:
- Table overflow behavior:
- Long text handling:

## Accessibility And Safety

- Keyboard path:
- Icon-only controls and labels:
- Contrast-sensitive surfaces:
- Reduced-motion concerns:
- Approval copy for risky actions:

## Validation Plan

- Typecheck command:
- Build command:
- Unit/smoke command:
- Registry validation:
- Usage audit:
- Playwright route coverage:
- Manual production check:

## Done Criteria

- [ ] Manifest is valid.
- [ ] Route is registered.
- [ ] Dashboard uses approved kit primitives.
- [ ] Empty/loading/error states exist.
- [ ] Desktop and mobile layouts do not horizontally overflow.
- [ ] Operator actions are visually distinct from read-only views.
- [ ] Tests or Playwright checks cover the dashboard.
- [ ] Documentation or plan status is updated.
