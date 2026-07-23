# Dashboard Design System Spine Plan

This plan tracks the work needed to move Hermes/TLC dashboards from project-specific tab piles into a shared, modern operating system. Mobbin MCP is part of this workflow, but it is a reference source, not the design system itself.

## Current Status

| Area | Status | Notes |
| --- | ---: | --- |
| Shared data contracts | 100% | Implemented in `@hermes/dashboard-kit/src/contracts.ts`. |
| Six-workspace information architecture | 100% | Implemented in `@hermes/dashboard-kit/src/workspaces.ts`. |
| Architecture assessment helper | 100% | Implemented in `@hermes/dashboard-kit/src/strategy.ts`. |
| Workspace overview component | 100% | Implemented in `@hermes/dashboard-kit/src/workspace-overview.tsx`. |
| Mobbin reference workflow | 100% | Documented in `docs/design/mobbin-reference-workflow.md`. |
| Project dashboard retrofits | 55% | Kashi VC and Media Engine emit architecture contracts, expose workspace overviews, validate architecture snapshots in tests, and have declared downstream consumers; package-native React migration remains open. |
| Prototype lab | 100% | Prototype contracts, starter registry, selected variants, preview evidence, static gallery, shared review component, and Hermes app route exist. |
| Production governance | 90% | Adoption registry, prototype registry, downstream feed registry, static gallery, and proving-project snapshot validation are enforced; CI gating remains deferred until package-native migration. |

Spine infrastructure readiness: **100% for the current proving-project boundary**

Rollout adoption readiness: **72%**

## V1 - Shared Dashboard Language

- [x] Define shared operational status vocabulary.
- [x] Define project status, cost, system health, readiness, alert, activity, and decision contracts.
- [x] Export contract helpers from `@hermes/dashboard-kit`.
- [x] Build package output.

## V2 - Shared Information Architecture

- [x] Define Command, Operations, Intelligence, Capacity, Projects, and Controls.
- [x] Add module grouping helper.
- [x] Document collapse rules for dashboards with long sidebars.
- [x] Add Hermes OS source-of-truth pointer.

## V3 - Assessment And Gap Detection

- [x] Add `assessDashboardArchitecture`.
- [x] Score coverage across the six workspaces.
- [x] Report gaps across data contracts, IA, business rules, implementation, and design-system usage.
- [x] Add reusable workspace overview UI component.

## V4 - Mobbin-Assisted Prototype Workflow

- [x] Document when to use Mobbin.
- [x] Define reference brief template.
- [x] Define 3-4 prototype direction requirement.
- [x] Document promotion rules and runtime boundaries.

## V5 - Dashboard Retrofit Targets

- [x] Kashi VC exports a shared dashboard architecture snapshot at `/api/dashboard-architecture`.
- [x] Media Engine exports a shared dashboard architecture snapshot at `/api/dashboard-architecture`.
- [x] Kashi VC has a visible first-screen workspace overview.
- [x] Media Engine has a visible first-screen workspace overview in the generated ops dashboard.
- [x] Media Business Operations declares Media Engine as its downstream dashboard snapshot producer.
- [x] Hermes OS declares proving-project snapshot producers through `dashboard-downstream-snapshot-feed.json`.
- [x] TLC Capital Group OS maps Khashi VC and Media Engine into `project-feed.v1`.
- [ ] Kashi VC migrates the visible dashboard shell to package-native `@hermes/dashboard-kit` components.
- [ ] Media Engine migrates the visible dashboard shell to package-native `@hermes/dashboard-kit` components.

## V6 - Prototype Lab

- [x] Define prototype set, variant, data requirement, and assessment contracts in `@hermes/dashboard-kit`.
- [x] Support 3-4 variants per dashboard redesign as a validation requirement.
- [x] Store reference notes, intended operator workflow, and data requirements in the prototype contract.
- [x] Define promotion actions for selected variants.
- [x] Add a persistent starter prototype registry for Kashi VC and Media Engine.
- [x] Add a local static prototype gallery at `docs/design/prototype-gallery/index.html`.
- [x] Add a generator command: `npm run dashboard:prototype:build`.
- [x] Generate Hermes app prototype data from the canonical prototype registry.
- [x] Add Hermes app route `/dashboard-prototypes`.
- [x] Store selected prototype records with preview evidence in the project-level registry.
- [x] Promote selected visual patterns back into dashboard-kit components through `DashboardPrototypeReview`.
- [x] Promote the static gallery concept into an app route.

## V7 - Enforcement

- [x] Add dashboard contract validation in `@hermes/dashboard-kit`.
- [x] Add workspace mapping validation.
- [x] Add adoption registry fields for contract coverage and IA coverage.
- [x] Add `npm run dashboard:spine:validate` for docs and adoption metadata.
- [x] Wire dashboard snapshot validation into Kashi VC and Media Engine tests.
- [x] Validate downstream snapshot feed producers and consumers.
- [x] Fail local quality checks when dashboards add unclassified modules, missing owners, invalid workspaces, or stale prototype gallery entries.
- [ ] Add CI gating once Kashi VC and Media Engine stop using static adapters.

## Boundary

Do not retrofit every project until Kashi VC and Media Engine prove the model. They are the best first two cases because they already expose the pain: too many dashboard tabs, unclear priority surfaces, live data confusion, and separate cost/intelligence/operations views.

## Completion Boundary

The design-system spine infrastructure is complete for the current proving-project boundary. The remaining work is adoption and production cutover, not missing spine primitives:

- migrate Kashi VC and Media Engine from static adapters to package-native dashboard shells
- add CI gating after the package-native migration removes static-adapter exceptions
- collect production screenshot/parity evidence before retiring the old static shells
