# Dashboard Prototype Lab

The prototype lab is the controlled place to explore better dashboards before changing production UI. Mobbin can provide references, but every prototype must be grounded in Hermes data contracts, workspace mapping, and operator questions.

## Prototype Contract

Use `DashboardPrototypeSet` from `@hermes/dashboard-kit`.

Starter prototype sets live in:

```text
docs/design/dashboard-prototype-registry.json
```

The generated review surface lives in:

```text
docs/design/prototype-gallery/index.html
```

The Hermes app route is:

```text
/dashboard-prototypes
```

`npm run dashboard:prototype:build` also regenerates:

```text
web/src/pages/dashboard-prototype-data.ts
```

Do not hand-edit that generated app data file. Edit `docs/design/dashboard-prototype-registry.json`, then rebuild.

Each set must include:

- project id and dashboard name
- objective
- operator questions
- at least three comparable variants
- workspace focus for each variant
- intended operator workflow
- reference notes
- data requirements
- selected variant and selection rationale before promotion

## Variant Rules

Each variant should be meaningfully different:

- **Command-first:** prioritizes attention, alerts, next actions, and blockers.
- **Operations-first:** prioritizes live runs, queues, failures, freshness, and deployment state.
- **Intelligence-first:** prioritizes findings, evidence, confidence, and recommendations.
- **Capacity-first:** optional fourth variant for cost, throughput, storage, and API consumption.

Do not compare superficial skin changes. The variants should answer different operator workflow assumptions.

## Promotion Rules

A prototype can move toward production only when:

- it has at least three variants
- operator questions are explicit
- required data is available or deliberately deferred
- selected variant has a written rationale
- promoted components or static adapter changes are listed
- production implementation keeps the six Hermes workspaces intact

Validate the registry and adoption metadata with:

```bash
npm run dashboard:prototype:build
npm run dashboard:spine:validate
```

## Current Boundary

Kashi VC and Media Engine are still the first proving grounds. Do not force this process across every project until those two dashboards prove that the pattern reduces sidebar sprawl and improves operational clarity.
