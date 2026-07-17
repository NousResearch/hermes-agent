# Hermes V1-V30 Runtime Consolidation

This document records the consolidation pass requested before adding any V31-V40 roadmap. It closes the biggest gaps exposed by V1-V30 without pretending the final production runtime is complete.

## Consolidated Capabilities

| Gap From Prior Versions | Built In This Pass | Remaining Production Hook |
| --- | --- | --- |
| Live data contracts existed but were not operationally visible | Server-backed runtime evidence records plus Hermes `/dashboard-snapshot` and aggregate `/api/dashboard/snapshots` endpoints | Rich project-owned `/dashboard-snapshot` payloads beyond registry/health synthesis |
| Decision/task memory was mostly static | SQLite `operating_runtime.db` store seeded from decisions and routed tasks | Project-specific memory ingestion and migrations |
| Permission policies were visible but not enforced | Server-side readiness checks create durable audit records and block high-risk execution | Middleware enforcement around real execution routes |
| Agent workbench was a dashboard model only | Workbench records persist with approval, artifacts, reports, and evidence | Real execution bridge with artifact ingestion |
| QA gates were plan-only for production | Runtime evidence for visual checks, production checks, provider scoring, kill switch, and budget breakers | Production URL screenshot checks, provider outcome history, runtime circuit breakers |

## What Is Now Real

- [x] `hermes_cli/operating_runtime.py` defines the SQLite runtime evidence, audit, and workbench store.
- [x] `hermes_cli/web_server.py` exposes `/api/operating-runtime/*` endpoints.
- [x] `hermes_cli/dashboard_snapshots.py` builds standard DashboardSnapshot payloads from Hermes and project registries.
- [x] Hermes exposes `/dashboard-snapshot`, `/api/dashboard/snapshot`, and `/api/dashboard/snapshots`.
- [x] Aggregate snapshots persist back into runtime evidence.
- [x] Executive summary consumes the server snapshot endpoint before falling back.
- [x] `web/src/pages/operating-runtime.ts` loads server-backed runtime evidence with local fallback.
- [x] V22-V30 pages show runtime evidence tied to each stage.
- [x] V22-V30 pages can run a server-backed readiness check.
- [x] permission-aware readiness checks create durable audit records.
- [x] High-risk stages remain gated instead of pretending autonomy is active.
- [x] Evidence is durable under `HERMES_HOME/operating_runtime.db`.
- [x] The dashboard remains testable without a project-specific migration.

## What Is Still Intentionally Gated

- [ ] Rich project-owned production snapshot endpoints for Khashi VC, Media Engine, and future dashboards.
- [ ] Project-specific memory ingestion into the Hermes runtime.
- [ ] Permission middleware wrapped around live execution endpoints.
- [ ] Real provider adapters and cost/quality scoring.
- [ ] Scheduled loop execution.
- [ ] Runtime kill switch and budget-breaker enforcement.
- [ ] Production URL screenshot and health validation.

## Why This Is The Right Boundary

This pass gives Hermes a working bridge from static planning to runtime supervision. It is useful immediately because the operator can see server-backed readiness evidence, gated items, audit checks, and workbench records. It is still safe because production-affecting execution remains blocked until live execution routes, permission middleware, project snapshot producers, and kill switches are wired.

## Post-Consolidation Runtime Progress

The V31-V40 work is now allowed because the production runtime path moved beyond synthesized registry/health signals:

1. Khashi VC and Media Engine expose project-owned `/dashboard-snapshot` or `snapshotUrl` payloads.
2. Hermes can ingest runtime evidence, workbench records, and audited permission decisions into durable storage.
3. Permission decisions run through `require_permission` and write durable audit records before high-risk actions.
4. Workbench tasks can persist artifacts and completion reports.
5. Production verification, incident command, deployment promotion, finance attribution, learning records, provider evals, and autonomy controls have runtime endpoints.

Remaining live-production gaps are narrower now: production route checks still need real network/screenshot execution, permission decisions still need to wrap every command handler, deployment promotion still needs the shared Hetzner runner, and autonomy controls still need kill-switch/budget-breaker enforcement at execution time.
