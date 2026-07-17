# Hermes Executive Operating System Build Plan V31-V40

This plan extends the V22-V30 autonomy-readiness layer into the next practical operating layer for TLC Capital Group OS. The goal is to make Hermes act less like a collection of dashboards and more like a governed executive control plane: project truth, telemetry, incidents, deployments, secrets, data lineage, finance attribution, learning, model evals, and executive cockpit.

| Version | Capability | Status | Built Surface | Remaining Runtime Gap |
| --- | --- | --- | --- | --- |
| V31 | Production project registry | `[x]` | `/project-registry`, registry ownership model, `/api/operating-runtime/production-checks`, route, metadata, validation | Live DNS/Caddy/screenshot execution |
| V32 | Telemetry fabric | `[x]` | `/telemetry-fabric`, telemetry families, runtime evidence, freshness model, route, metadata, validation | Project-owned telemetry adapters |
| V33 | Incident command | `[x]` | `/incident-command`, `/api/operating-runtime/incidents`, severity/owner/rollback model, route, metadata, validation | Automated incident ingestion/remediation |
| V34 | Deployment promotion rail | `[x]` | `/deployment-promotion`, `/api/operating-runtime/deployments`, promotion evidence model, route, metadata, validation | Shared live deployment runner |
| V35 | Secrets and access posture | `[x]` | `/secrets-posture`, secret-presence runtime evidence, permission-decision audit, route, metadata, validation | Vault/rotation backend and live GitHub/Hetzner checks |
| V36 | Data source catalog | `[x]` | `/data-source-catalog`, `/api/operating-runtime/data-sources`, source/freshness/retention model, route, metadata, validation | Automated lineage discovery and project reports |
| V37 | Finance and cost attribution | `[x]` | `/finance-attribution`, `/api/operating-runtime/costs`, cost bucket/business unit model, route, metadata, validation | Actual invoice reconciliation |
| V38 | Learning engine | `[x]` | `/learning-engine`, `/api/operating-runtime/learning`, evidence promotion model, route, metadata, validation | Automatic outcome ingestion |
| V39 | Agent evaluation lab | `[x]` | `/agent-eval-lab`, `/api/operating-runtime/evals`, provider eval matrix, route, metadata, validation | Larger golden task execution history |
| V40 | Executive cockpit | `[x]` | `/executive-cockpit`, `/api/operating-runtime/autonomy-controls`, executive rollup model, route, metadata, validation | Fully live V31-V39 signal feed and approval agenda |

## V31 Production Project Registry

- [x] Register a first-class route for the production project registry.
- [x] Track registry ownership, production URL, health URL, and snapshot URL expectations.
- [x] Add runtime evidence for root registry coverage and route verification.
- [x] Add `/api/operating-runtime/production-checks` to record production route verification evidence.
- [ ] Run a live DNS/Caddy/health/snapshot/screenshot sweep after production routes are final.

## V32 Telemetry Fabric

- [x] Define required telemetry families: health, logs, cost, capacity, storage, and queues.
- [x] Add stale/missing signal states to the operating evidence layer.
- [x] Add route, metadata, and validator coverage.
- [ ] Add full usage/storage/queue instrumentation inside each production project.

## V33 Incident Command

- [x] Define incident severity, owner, next step, and rollback path.
- [x] Add incident command route and runtime evidence.
- [x] Add `/api/operating-runtime/incidents` for runtime incident records.
- [x] Keep auto-remediation gated by explicit permissions.
- [ ] Ingest real project health failures into an incident store.

## V34 Deployment Promotion Rail

- [x] Define the promotion gates: validate, build, test, migrate, deploy, health-check, screenshot, rollback.
- [x] Track migration-aware deployment requirements.
- [x] Add `/api/operating-runtime/deployments` for deployment promotion evidence.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Wire the shared Hermes/Hetzner promotion runner as the live deploy rail.

## V35 Secrets And Access Posture

- [x] Track secret presence and access scope without exposing values.
- [x] Model SSH keys, app env, dashboard auth, API keys, and webhooks.
- [x] Add audited `/api/operating-runtime/permission-decision` for high-risk access/deploy actions.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Choose a vault or managed secret backend before automating rotation.

## V36 Data Source Catalog

- [x] Define owner, cadence, freshness, cost, retention, and consumer fields.
- [x] Add retention governance for generated media, market data, and telemetry stores.
- [x] Add `/api/operating-runtime/data-sources` for data-source catalog records.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Automate project data-lineage reports.

## V37 Finance And Cost Attribution

- [x] Define model/API/storage/hosting/labor-proxy cost buckets.
- [x] Map costs to project and business unit.
- [x] Add `/api/operating-runtime/costs` for runtime cost attribution records.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Import actual invoice or manual monthly rate assumptions.

## V38 Learning Engine

- [x] Define candidate, finding, policy proposal, rejected, and gated evidence states.
- [x] Model evidence promotion and counterevidence requirements.
- [x] Add `/api/operating-runtime/learning` for learning evidence ingestion.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Feed outcomes from Khashi experiments, Media Engine generations, deploys, and model choices.

## V39 Agent Evaluation Lab

- [x] Define provider eval dimensions: correctness, tests, design, cost, and latency.
- [x] Model golden tasks by task family.
- [x] Add `/api/operating-runtime/evals` for provider evaluation records.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Persist provider outcome history before automatic model routing.

## V40 Executive Cockpit

- [x] Combine health, cost, incidents, deployments, learning, revenue, and autonomy into one executive view.
- [x] Track approval agenda and board narrative readiness.
- [x] Add `/api/operating-runtime/autonomy-controls` for kill-switch, budget breaker, and project autonomy-limit records.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Feed the cockpit from live V31-V39 runtime signals.
