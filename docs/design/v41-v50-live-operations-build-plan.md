# Hermes Live Operations Build Plan V41-V50

This plan turns the V1-V40 control-plane spine into a live operations layer. The goal is not more conceptual dashboards; it is production verification, command enforcement, project telemetry, incident ingestion, deployment promotion, secret posture, cost attribution, learning ingestion, model evals, and runtime circuit breakers.

| Version | Capability | Status | Built Surface | Remaining Runtime Gap |
| --- | --- | --- | --- | --- |
| V41 | Live production verification runner | `[x]` | `/production-verification`, production check contract, route, metadata, validation | Real DNS/Caddy/screenshot sweep execution |
| V42 | Command gate runtime | `[x]` | `/command-gates`, permission decision and audit contract, route, metadata, validation | Wrap every live command handler |
| V43 | Project telemetry adapter kit | `[x]` | `/telemetry-adapters`, adapter contract, route, metadata, validation | Adopt adapters in every production project |
| V44 | Incident ingestion and escalation | `[x]` | `/incident-ingestion`, incident rule model, route, metadata, validation | Automatic ingestion from V41/V43 failures |
| V45 | Shared deployment promotion runner | `[x]` | `/promotion-runner`, promotion gate model, route, metadata, validation | Actual Hetzner/GitHub deploy runner execution |
| V46 | Secrets posture scanner | `[x]` | `/secret-scanner`, secret presence/scope model, route, metadata, validation | Live GitHub/Hetzner secret scans with approved credentials |
| V47 | Cost attribution engine | `[x]` | `/cost-attribution-engine`, cost input model, route, metadata, validation | Invoice/rate import and reconciliation |
| V48 | Learning ingestion pipeline | `[x]` | `/learning-ingestion`, learning event model, route, metadata, validation | Automatic outcome feeds from projects |
| V49 | Agent and model eval harness | `[x]` | `/model-eval-harness`, golden task/eval model, route, metadata, validation | Real golden-task execution history |
| V50 | Runtime circuit breakers | `[x]` | `/circuit-breakers`, kill-switch/budget-breaker model, route, metadata, validation | Hard enforcement in execution paths |

## V41 Live Production Verification Runner

- [x] Add a first-class production verification dashboard.
- [x] Track DNS, Caddy, health, snapshot, auth, and screenshot check classes.
- [x] Connect the stage to production-check runtime evidence.
- [ ] Execute live production network and screenshot sweeps through an approved runner.

## V42 Command Gate Runtime

- [x] Add a command gate dashboard.
- [x] Track deploy, secret, scheduler, and autonomy command risk classes.
- [x] Connect the stage to durable permission decision and audit records.
- [ ] Wrap every production-affecting command handler with the gate.

## V43 Project Telemetry Adapter Kit

- [x] Define health, cost, storage, API, queue, deployment, and action-needed fields.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Adopt the adapter kit inside every production project.

## V44 Incident Ingestion And Escalation

- [x] Define incident source, severity, owner, next step, rollback, and status.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Automatically create incidents from failed production checks and stale telemetry.

## V45 Shared Deployment Promotion Runner

- [x] Define validate, build, migrate, deploy, health-check, screenshot, and rollback gates.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Wire the shared Hermes/Hetzner promotion runner behind V42 command gates.

## V46 Secrets Posture Scanner

- [x] Define secret presence, scope, deploy-key posture, and rotation status.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Run live GitHub/Hetzner checks without exposing values.

## V47 Cost Attribution Engine

- [x] Define model, API, storage, hosting, and manual-rate cost inputs.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Import monthly rates or invoices for reconciliation.

## V48 Learning Ingestion Pipeline

- [x] Define learning events from experiments, content, deploys, incidents, model choices, and operations.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Feed Khashi, Media Engine, deployment, incident, and eval outcomes automatically.

## V49 Agent And Model Eval Harness

- [x] Define golden task, provider run, eval score, and routing recommendation contracts.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Execute and persist golden-task runs before automatic provider routing.

## V50 Runtime Circuit Breakers

- [x] Define kill switch, budget breaker, provider-spend cap, loop stop, and project autonomy limit controls.
- [x] Add route, metadata, runtime evidence, and validation.
- [ ] Enforce breakers inside every live execution path.

