# Hermes Boundary Closure Build Plan V51-V60

This plan closes the V41-V50 important boundary by turning gated concepts into governed runtime paths. The posture is still approval-first: dry runs and evidence recording are built, while live network, Hetzner, secret, provider-spend, and remediation actions remain blocked unless command gates and circuit breakers allow them.

| Version | Capability | Status | Built Surface | Remaining Runtime Gap |
| --- | --- | --- | --- | --- |
| V51 | Production DNS and health sweep | `[x]` | `/production-sweep`, sweep endpoint, production-check evidence | Actual network/TLS/screenshot execution from approved runner |
| V52 | Hetzner promotion execution | `[x]` | `/hetzner-promotion-execution`, promotion execution endpoint, deployment evidence | Remote SSH/docker execution and rollback automation |
| V53 | Command gate coverage auditor | `[x]` | `/command-gate-coverage`, gate coverage endpoint, missing-gate incidents | Patch every project-specific live handler |
| V54 | Project adapter rollout | `[x]` | `/project-adapter-rollout`, adapter rollout endpoint, telemetry evidence | Implement rich adapters in every production project |
| V55 | Incident automation engine | `[x]` | `/incident-automation`, batch incident endpoint, source-to-incident rules | Connect live sweep/adapter failures automatically |
| V56 | Live secret presence scan | `[x]` | `/live-secret-scan`, presence-only scanner endpoint, secrets evidence | Approved GitHub/Hetzner credential scans |
| V57 | Cost reconciliation import | `[x]` | `/cost-reconciliation`, rate/invoice import endpoint, finance evidence | Direct provider invoice API imports |
| V58 | Outcome learning feeds | `[x]` | `/outcome-learning-feeds`, batch learning endpoint, learning evidence | Project-owned automatic outcome emitters |
| V59 | Golden eval execution | `[x]` | `/golden-eval-execution`, golden eval batch endpoint, eval evidence | Real provider execution harness and artifact scoring |
| V60 | Hard circuit breaker enforcement | `[x]` | `/hard-breaker-enforcement`, breaker-check endpoint, autonomy block evidence | Patch every live execution path through breaker checks |

## V51 Production DNS And Health Sweep

- [x] Add a production sweep dashboard.
- [x] Add a sweep endpoint that records dry-run and approved live-attempt evidence.
- [x] Store DNS/TLS/Caddy/health/snapshot/screenshot check intent without requiring network access.
- [ ] Execute real network and screenshot checks from an approved runner.

## V52 Hetzner Promotion Execution

- [x] Add a Hetzner promotion execution dashboard.
- [x] Add an endpoint that records promotion plans, migration intent, audit records, and deployment evidence.
- [x] Keep live promotion gated behind admin role and explicit approval.
- [ ] Run the shared Hermes/Hetzner promotion script remotely and attach rollback artifacts.

## V53 Command Gate Coverage Auditor

- [x] Add a command gate coverage dashboard.
- [x] Add coverage records for deploy, secrets, scheduler, autonomy, and spend handlers.
- [x] Auto-create an incident when a high-risk handler is reported as uncovered.
- [ ] Patch all project-specific live handlers through the permission primitive.

## V54 Project Adapter Rollout

- [x] Add an adapter rollout dashboard.
- [x] Record manifest URL, snapshot URL, and missing telemetry fields.
- [x] Keep missing fields visible without blocking the whole dashboard.
- [ ] Implement the adapter contract inside each production project.

## V55 Incident Automation Engine

- [x] Add an incident automation dashboard.
- [x] Add batch incident ingestion from failed operational signals.
- [x] Keep remediation manual and approval-gated.
- [ ] Subscribe the incident engine to live sweep and telemetry failures.

## V56 Live Secret Presence Scan

- [x] Add a live secret scan dashboard.
- [x] Add a presence-only scan endpoint that stores names, counts, missing names, and scope.
- [x] Never store raw secret values.
- [ ] Wire approved GitHub/Hetzner scan adapters.

## V57 Cost Reconciliation Import

- [x] Add a cost reconciliation dashboard.
- [x] Add manual rate-sheet/invoice import records.
- [x] Distinguish missing rates from actual cost records.
- [ ] Add direct billing-provider imports once credentials and cost policy are approved.

## V58 Outcome Learning Feeds

- [x] Add an outcome learning feeds dashboard.
- [x] Add batch learning ingestion from project outcome events.
- [x] Keep policy promotion gated by evidence and approval.
- [ ] Add project-owned emitters for Khashi, Media Engine, deployments, incidents, and evals.

## V59 Golden Eval Execution

- [x] Add a golden eval execution dashboard.
- [x] Add batch eval result recording by provider and task family.
- [x] Keep automatic model routing gated.
- [ ] Run real provider tasks and attach scored artifacts.

## V60 Hard Circuit Breaker Enforcement

- [x] Add a hard breaker enforcement dashboard.
- [x] Add an execution-path breaker check endpoint.
- [x] Record block/allow evidence for matching active breakers.
- [ ] Patch every live deploy, scheduler, provider, and autonomy path through the breaker check.
