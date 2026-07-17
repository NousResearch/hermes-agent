# Dashboard Design Review Checklist

Run this before shipping a dashboard change.

## Structure

- [ ] The page uses `DashboardShell` or the static `hdk-shell` adapter.
- [ ] The dashboard has a valid `hermes.dashboards.json` entry.
- [ ] Sidebar/navigation labels match the actual workflow.
- [ ] The first screen shows useful operational state, not marketing copy.

## Components

- [ ] KPI cards use `KpiCard` or `hdk-kpi-*`.
- [ ] Tables use `DataTable` or `hdk-table`/`hdk-table-wrap`.
- [ ] Empty/loading/error states use shared primitives.
- [ ] Operator actions use command/action components.
- [ ] Charts are wrapped in `ChartPanel`.

## Data And Failure States

- [ ] Every API/file source has an owner and failure state.
- [ ] Empty states explain what action creates data.
- [ ] Loading states do not block unrelated page sections.
- [ ] Error states distinguish missing data from system failure.

## Accessibility

- [ ] Keyboard focus reaches primary controls.
- [ ] Icon-only controls have `aria-label` or visible text.
- [ ] Text does not overflow buttons, cards, sidebars, or tables.
- [ ] Mobile layout has no document-level horizontal overflow.
- [ ] Motion is reduced under `prefers-reduced-motion`.

## Validation

- [ ] `npm run dashboard:registry:validate`
- [ ] `npm run dashboard:usage:audit`
- [ ] `npm run dashboard:visual:check` when route/UI changed.
- [ ] Relevant project build or smoke test passes.
