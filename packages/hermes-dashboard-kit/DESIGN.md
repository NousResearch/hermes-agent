---
name: Hermes Dashboard Kit
version: 0.1.0
source_package: "@hermes/dashboard-kit"
source_css: "packages/hermes-dashboard-kit/static/hermes-dashboard-kit.css"
tokens:
  colors:
    background: "var(--hdk-bg)"
    surface: "var(--hdk-card)"
    surface_muted: "var(--hdk-card-muted)"
    border: "var(--hdk-border)"
    border_strong: "var(--hdk-border-strong)"
    text: "var(--hdk-text)"
    muted_text: "var(--hdk-muted)"
    primary: "var(--hdk-primary)"
    primary_soft: "var(--hdk-primary-soft)"
    success: "var(--hdk-success)"
    success_soft: "var(--hdk-success-soft)"
    warning: "var(--hdk-warning)"
    warning_soft: "var(--hdk-warning-soft)"
    critical: "var(--hdk-critical)"
    critical_soft: "var(--hdk-critical-soft)"
  typography:
    font_family: "var(--hdk-font)"
    card_label: "12px"
    metric_value: "28px"
    table_header: "11px"
  radius:
    default: "var(--hdk-radius)"
    control: "6px"
  spacing:
    shell: "18px"
    sidebar: "14px"
    section: "16px"
    card: "14px"
    grid_gap: "12px"
  components:
    shell: "DashboardShell / .hdk-shell"
    sidebar: "DashboardSidebar / .hdk-sidebar"
    header: "DashboardHeader / .hdk-header"
    card: "KpiCard, ChartPanel, DataTable / .hdk-card"
    table: "DataTable / .hdk-table"
    button: "Command buttons / .hdk-button"
    status: "StatusPill / .hdk-pill"
---

# Hermes Dashboard Kit

This file is the machine-readable and human-readable design contract for Hermes/TLC operational dashboards. It follows the same intent as Google Labs `design.md`: keep a persistent design-system description that coding agents can read before creating UI.

## Product Intent

Hermes dashboards are operating surfaces, not landing pages. They should help a human operator understand business state, risk, cost, work queues, experiments, automation, and decisions quickly.

The interface should be compact, scannable, and evidence-oriented. It should avoid decorative layouts that make dashboards look modern while reducing operational clarity.

## Source Of Truth

The canonical implementation lives in `packages/hermes-dashboard-kit`.

React dashboards should consume `@hermes/dashboard-kit`. Static dashboards should consume or sync from `packages/hermes-dashboard-kit/static/hermes-dashboard-kit.css` until they are migrated to package-native React.

Hermes OS governs enforcement and adoption. It does not own a competing design-system implementation.

## Layout Rules

- Use a stable left-navigation plus main-workspace shell for operational dashboards.
- Use dense metric grids, tables, charts, and insight panels.
- Do not use marketing heroes, oversized card stacks, or decorative page backgrounds in dashboard views.
- Do not nest cards inside cards.
- Preserve stable dimensions for KPI cards, tables, charts, buttons, tabs, and status controls.
- Mobile layouts must keep the primary workflow usable before secondary analytics.

## Component Rules

- Use `DashboardShell`, `DashboardHeader`, `MetricGrid`, `KpiCard`, `DataTable`, `ChartPanel`, and status primitives before creating local UI.
- Extend the kit when two or more dashboards need the same pattern.
- Keep local components only when they represent project-specific behavior or data.
- Icon-only controls require accessible names.
- Dangerous actions require explicit confirmation.

## Visual Rules

- Use the `--hdk-*` token layer or compatible app-level variables.
- Status color is semantic: success, warning, critical, neutral, unknown.
- Border radius should stay at `8px` or below unless the kit changes the token.
- Do not use one-off gradients, decorative blobs, or viewport-scaled typography.
- Letter spacing should remain normal.

## Agent Rules

Before building or changing a dashboard, agents should:

1. Read this `DESIGN.md`.
2. Read `docs/design/hermes-dashboard-design-contract.md`.
3. Check `docs/design/dashboard-kit-adoption.md`.
4. Prefer package primitives or the static adapter.
5. Update adoption status when a dashboard moves closer to package-native usage.

