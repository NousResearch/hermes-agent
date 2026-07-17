# V8-V14 Trackable Dashboard Build Plan

This plan tracks the work after the V1-V7 foundation. The goal is not more scaffolding; the goal is to turn Hermes dashboards into a live, package-native, production-tested operating system.

## Status Summary

| Version | Goal | Status | Completion |
|---|---|---:|---:|
| V8 | Package-native dashboard migrations | `[~]` | 30% |
| V9 | Live data contracts and executive signals | `[~]` | 45% |
| V10 | Premium visual QA and design review | `[~]` | 45% |
| V11 | Agent-enforced dashboard creation | `[~]` | 50% |
| V12 | Hermes central command layer | `[x]` | 100% |
| V13 | Multi-brand theme and product polish | `[x]` | 100% |
| V14 | Dashboard marketplace and plugin system | `[x]` | 100% |

## V8: Package-Native Dashboard Migrations

Goal: move priority dashboards from static adapter surfaces into package-native `@hermes/dashboard-kit` implementations.

### V8.1 Migration Workbench

- [x] Add Hermes route for tracking package-native dashboard migrations.
- [x] Show migration targets, recipe mapping, current state, target state, next step, and completion.
- [x] Add V8 validation script.
- [x] Add route/nav entry for the migration workbench.

### V8.2 Media Engine Ops Migration

- [ ] Split dashboard snapshot JSON API from generated HTML.
- [ ] Build package-native React dashboard consuming the snapshot.
- [ ] Convert autopilot controls with `CommandBar`.
- [ ] Convert generation history, approvals, publishing events, and Discord delivery to `DataTable`.
- [ ] Keep generated static dashboard until parity is verified.

### V8.3 Khashi VC ROC Migration

- [ ] Extract ROC API client from static `app.js`.
- [ ] Build package-native shell and route model.
- [ ] Convert Run Monitor and command center first.
- [ ] Convert Market Data, Coverage, Cost, Persistence, Activity, and System views.
- [ ] Keep static ROC until parity is verified.

### V8.4 Executive Summary Package-Native Upgrade

- [x] Executive summary already uses package-native components.
- [x] Executive summary now consumes standard dashboard snapshots.
- [ ] Replace fallback signals with live project signal endpoints.

### V8.5 Adapter Retirement

- [ ] Define parity checklist for each retired adapter.
- [ ] Require Playwright coverage.
- [ ] Require rollback path.
- [ ] Remove static adapter only after production verification.

## V9: Live Data Contracts And Executive Signals

Goal: standardize the data every project exposes to Hermes.

### V9.1 Signal Contract Package

- [x] Add `DashboardSignalSource`.
- [x] Add `DashboardSnapshot`.
- [x] Add `HealthSnapshot`.
- [x] Add `CostSnapshot`.
- [x] Add `CapacitySnapshot`.
- [x] Add `QueueSnapshot`.
- [x] Add `ActionNeeded`.
- [x] Add `ResearchSignal`.
- [x] Add `DeploymentSignal`.
- [x] Export signal helpers from `@hermes/dashboard-kit`.

### V9.2 Hermes Signal Adapter

- [x] Add known dashboard signal sources for Khashi VC, Media Engine, and Hermes.
- [x] Add plugin-to-dashboard-snapshot adapter.
- [x] Add Hermes `/dashboard-snapshot`, `/api/dashboard/snapshot`, and `/api/dashboard/snapshots`.
- [x] Add registry-to-DashboardSnapshot synthesis with optional health checks.
- [x] Persist aggregate snapshot health back into operating runtime evidence.
- [x] Add standard fallback snapshots with explicit missing cost/capacity signals.
- [x] Upgrade executive summary to consume server snapshots before plugin/fallback snapshots.

### V9.3 Project Signal Endpoints

- [ ] Add rich project-owned `/dashboard-snapshot` or `snapshotUrl` payload to Media Engine.
- [ ] Add rich project-owned `/dashboard-snapshot` or `snapshotUrl` payload to Khashi VC.
- [ ] Add standard health/cost/capacity/queue/action data.
- [ ] Add stale/unknown states explicitly.

### V9.4 Executive Rollups

- [x] Roll up standard snapshot health into project scorecards.
- [x] Roll up queue state into throughput and capacity metrics.
- [x] Roll up action-needed signals into executive action queue.
- [ ] Roll up real cost telemetry after project endpoints exist.

## V10: Premium Visual QA And Design Review

- [x] Add dashboard quality scorecard.
- [x] Add governed dashboard metadata with recipe, data contract, states, validation, owner, and category.
- [x] Add V10 validator.
- [x] Add visual coverage for `/dashboard-migrations`.
- [ ] Add live production screenshot checks.
- [ ] Add visual baselines for Media Engine, Khashi VC, and Hermes Executive.
- [ ] Add automated recipe compliance scoring.
- [ ] Add design review checklist gate for high-impact dashboard releases.

## V11: Agent-Enforced Dashboard Creation

- [x] Require recipe metadata for every governed dashboard route.
- [x] Require data contract metadata before dashboard implementation can pass validation.
- [x] Require state coverage metadata.
- [x] Add V11 validator for route metadata, approved recipes, data contracts, states, and validation.
- [ ] Require screenshot evidence in final handoff.
- [ ] Reject dashboard changes that bypass the design kit without documented exception.

## V12: Hermes Central Command Layer

- [x] Add `/central-command` route.
- [x] Add daily cross-project brief.
- [x] Add action-needed queue backed by project signals.
- [x] Add health/cost/capacity-style rollups from standard snapshots.
- [x] Add business impact summaries.
- [x] Add V12 validator.
- [x] Add governed route metadata.
- [x] Keep live external project endpoint work parked until Khashi/Media migrations resume.

## V13: Multi-Brand Theme And Product Polish

- [x] Add TLC base theme.
- [x] Add Khashi VC research theme.
- [x] Add Media Engine publishing theme.
- [x] Add Media Business Ops analytics theme.
- [x] Add domain-specific density rules.
- [x] Add `/theme-system` route.
- [x] Add V13 validator.
- [x] Add governed route metadata.
- [x] Leave deeper contrast matrix automation for later hardening.

## V14: Dashboard Marketplace And Plugin System

- [x] Define dashboard plugin manifest.
- [x] Add panel registry model.
- [x] Add command registry model.
- [x] Add health/cost/capacity signal registry model.
- [x] Add permission-aware command metadata.
- [x] Add `/dashboard-marketplace` route.
- [x] Add V14 validator.
- [x] Add governed route metadata.
- [x] Leave dynamic remote plugin discovery for future runtime integration.
