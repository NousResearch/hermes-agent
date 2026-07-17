# Hermes Live Adapter Build Plan V61-V70

This plan handles the remaining important boundary after V51-V60: turning governed evidence endpoints into approved live adapters that can safely touch networks, servers, providers, projects, and release trains. The build keeps a hard line between adapter contracts and live execution; real external execution still requires command gates, circuit breakers, and operator approval.

| Version | Capability | Status | Built Surface | Remaining Runtime Gap |
| --- | --- | --- | --- | --- |
| V61 | Network runner adapter | `[x]` | `/network-runner-adapter`, adapter-run contract, registry evidence, approved DNS/TLS/HTTP probes | Screenshot capture remains artifact-pointer based until a production browser runner is attached |
| V62 | Hetzner SSH adapter | `[x]` | `/hetzner-ssh-adapter`, remote command plan evidence, breaker-gated promotion command execution switch | True remote SSH execution depends on configured production credentials and operator approval |
| V63 | Secret provider adapter | `[x]` | `/secret-provider-adapter`, presence-only adapter contract, GitHub secret/variable name scanner | Hetzner and provider-native scanners still need their provider-specific CLIs/APIs |
| V64 | Billing provider adapter | `[x]` | `/billing-provider-adapter`, billing import adapter contract, manual billing import API | Direct provider billing API integrations |
| V65 | Project outcome emitter | `[x]` | `/project-outcome-emitter`, outcome adapter contract, Khashi VC and Media Engine `/api/hermes/outcomes` adapters | Remaining projects need the same adapter endpoint |
| V66 | Provider eval runner | `[x]` | `/provider-eval-runner`, breaker-aware provider run contract | Real model calls and artifact scoring |
| V67 | Breaker middleware SDK | `[x]` | `/breaker-middleware`, reusable execution-check contract, sweep/promotion breaker enforcement | Patch every remaining live project entry point through middleware |
| V68 | Incident subscription bus | `[x]` | `/incident-subscriptions`, source subscription endpoint, automatic incidents from sweep/deploy/secret/breaker failures | Notification fanout still needs channel subscriptions |
| V69 | Evidence artifact store | `[x]` | `/evidence-artifact-store`, artifact pointer endpoint, deterministic artifact fingerprints | Production artifact storage backend and retention jobs |
| V70 | Release train orchestrator | `[x]` | `/release-train-orchestrator`, train plan endpoint | Approved multi-project release train execution |

## V61 Network Runner Adapter

- [x] Add a network runner adapter dashboard.
- [x] Add adapter-run evidence for DNS/TLS/HTTP/snapshot/screenshot probe plans.
- [x] Keep live network probes behind explicit approval and breaker checks.
- [x] Implement real DNS/TLS/HTTP probing in the approved runner.
- [x] Index screenshot/log artifacts by pointer when supplied.
- [ ] Attach a production browser screenshot runner.

## V62 Hetzner SSH Adapter

- [x] Add a Hetzner SSH adapter dashboard.
- [x] Add remote command plan evidence.
- [x] Keep SSH execution behind explicit approval.
- [x] Build validated command plans for build/test/migrate/compose/verify.
- [x] Enforce circuit breakers before live promotion.
- [x] Allow command execution only when live, approved, and explicitly requested.
- [ ] Attach remote SSH transport and rollback artifact capture.

## V63 Secret Provider Adapter

- [x] Add a secret provider adapter dashboard.
- [x] Add presence-only provider scan evidence.
- [x] Keep raw secret values out of all payloads.
- [x] Wire live GitHub secret/variable name scans with approved credentials.
- [ ] Add Hetzner and provider-native secret presence adapters.

## V64 Billing Provider Adapter

- [x] Add a billing provider adapter dashboard.
- [x] Add billing import adapter records.
- [x] Preserve manual import fallback.
- [x] Add manual billing import API for invoice/rate-sheet reconciliation.
- [ ] Add direct billing APIs per provider.

## V65 Project Outcome Emitter

- [x] Add a project outcome emitter dashboard.
- [x] Add outcome adapter evidence.
- [x] Standardize event source and evidence linking.
- [x] Adopt the emitter inside Khashi VC.
- [x] Adopt the emitter inside Media Engine.
- [ ] Adopt the emitter inside the remaining production projects.

## V66 Provider Eval Runner

- [x] Add a provider eval runner dashboard.
- [x] Add breaker-aware provider-run evidence.
- [x] Keep paid model calls gated.
- [ ] Execute real provider tasks and attach scored artifacts.

## V67 Breaker Middleware SDK

- [x] Add a breaker middleware dashboard.
- [x] Add middleware integration evidence.
- [x] Define deploy/scheduler/provider/autonomy hooks.
- [x] Patch production sweep and promotion paths through breaker middleware.
- [ ] Patch scheduler, provider, and project-owned deploy entry points through the middleware.

## V68 Incident Subscription Bus

- [x] Add an incident subscription dashboard.
- [x] Add source subscription records with dedupe keys.
- [x] Keep remediation approval-gated.
- [x] Create incidents from failed/blocked sweeps, deploys, missing secrets, and breaker blocks.
- [ ] Add notification fanout for subscribed incident sources.

## V69 Evidence Artifact Store

- [x] Add an evidence artifact store dashboard.
- [x] Add artifact pointer records for screenshots, logs, traces, receipts, invoices, and evals.
- [x] Store pointers instead of large blobs.
- [x] Add deterministic artifact fingerprints for pointer records.
- [ ] Choose and implement the production artifact backend and retention jobs.

## V70 Release Train Orchestrator

- [x] Add a release train orchestrator dashboard.
- [x] Add release train plan records.
- [x] Keep multi-project release trains manual-first.
- [ ] Execute approved release trains after adapters prove reliable.
