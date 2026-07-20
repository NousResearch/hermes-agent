# Hermes Dashboard Kit

Shared React dashboard primitives for Hermes/TLC operational dashboards.

## Canonical Role

This package is the source of truth for the Hermes/TLC dashboard design system. The machine-readable and human-readable design contract lives in `DESIGN.md`.

Hermes OS should enforce and report adoption of this package. It should not maintain a competing dashboard component system.

## Install From This Workspace

Inside a workspace project:

```json
{
  "dependencies": {
    "@hermes/dashboard-kit": "file:../packages/hermes-dashboard-kit"
  }
}
```

For an external local project before publication:

```json
{
  "dependencies": {
    "@hermes/dashboard-kit": "file:../nous-hermes-agent/packages/hermes-dashboard-kit"
  }
}
```

The consuming app must provide:

- React
- lucide-react
- Tailwind-compatible utility classes
- Hermes/TLC theme tokens or compatible CSS variables

For static dashboards, use:

```text
packages/hermes-dashboard-kit/static/hermes-dashboard-kit.css
```

See `docs/design/static-dashboard-adapter.md` for the migration class map.

## Core Imports

```tsx
import {
  DashboardShell,
  DashboardHeader,
  MetricGrid,
  KpiCard,
  DataTable,
  ChartPanel,
  DashboardLauncher,
  CommandBar,
} from "@hermes/dashboard-kit";
```

## Versioning Rule

- `0.x`: internal pre-release package while Khashi VC and Media Engine migrations are underway.
- Patch releases may add components or fix styling.
- Minor releases may add props.
- Breaking changes require a migration note in `CHANGELOG.md`.

## Migration Rule

Do not copy components into a project. Consume the package and extend the package when two or more dashboards need the same behavior.

Static dashboards may use `static/hermes-dashboard-kit.css` as a bridge, but copied CSS must be tracked in `docs/design/dashboard-kit-adoption.json` and checked with:

```bash
npm run dashboard:design-system:status
```
