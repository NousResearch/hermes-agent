# Hermes Dashboard Design System Ownership

## Decision

`@hermes/dashboard-kit` in `projects/nous-hermes-agent/packages/hermes-dashboard-kit` is the canonical dashboard design-system implementation.

Hermes OS is the governance and enforcement layer. It should reference the kit, validate adoption, and report dashboard quality, but it should not maintain a competing set of dashboard components.

## Why This Split Exists

Nous Hermes Agent is the agent/runtime layer where shared agent-facing product capabilities live. The dashboard kit belongs there because it is the reusable package that Codex, Hermes, and project dashboards can consume.

Hermes OS is the control plane. It should answer whether projects are following the design system, which dashboards are drifted, and what needs migration next.

## Responsibilities

| Area | Owner | Notes |
| --- | --- | --- |
| React components | Nous Hermes Agent `@hermes/dashboard-kit` | Package-native source of truth. |
| Static dashboard CSS adapter | Nous Hermes Agent `packages/hermes-dashboard-kit/static` | Temporary bridge for non-React dashboards. |
| Design contract | Nous Hermes Agent `DESIGN.md` and `docs/design` | Rules agents must read before dashboard work. |
| Adoption registry | Nous Hermes Agent `docs/design/dashboard-kit-adoption.json` | Tracks which dashboards consume the kit and how. |
| Enforcement and reporting | Hermes OS | Uses registry/status output to report maturity and drift. |
| Project-specific UX | Each project | Allowed when behavior is domain-specific and not reusable. |

## Cleanup Rule

When a dashboard uses copied CSS, it must be listed in the adoption registry. Copied CSS is acceptable only as an adapter phase. The target state is package-native consumption for React dashboards and controlled sync for static dashboards.

