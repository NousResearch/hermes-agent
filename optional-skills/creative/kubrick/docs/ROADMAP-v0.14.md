# Kubrick v0.14 Status and Continuation Roadmap

## Current state

**v0.14.0 consolidates Kubrick's deterministic contract as a standalone Hermes skill.**

The release freezes ontology growth long enough to make the existing compiler surface
machine-readable, reproducible, failure-stable, provider-auditable, install-safe, and
portable into the official Hermes optional-skills tree.

## Shipped in v0.14

- `kubrick.manifest.yaml` is the canonical registry for runtime profiles, intents,
  actions, aliases, recipes, artifacts, schemas, providers, authority classes, and
  exit codes.
- `pyproject.toml` documents Python 3.10–3.14 and optional validation/dev tooling
  without turning the skill into a required package.
- Compile receipts identify the Kubrick release, corpus, schema bundle, provider
  adapter, normalized command, and normalized semantic inputs with stable digests.
- Structured diagnostics use stable exit codes for invalid commands, unavailable
  dependencies, and `NOT_COMPUTABLE` results.
- Provider adapters emit semantic-preservation reports and fail closed on critical
  loss of graph identity, required content, ownership, geometry, state change,
  residue, continuity, or negative constraints.
- The Hermes installer stages and validates before activation, swaps atomically,
  records receipts, preserves external backups, supports dry-run and rollback, and
  restores the prior installation on activation failure.
- `SKILL.md` follows the modern Hermes skill section contract and remains a concise
  behavior document rather than a duplicate engineering manual.
- Official Hermes packaging exports repository-level contract tests and the complete
  storyboard recipe fixture set.
- CI separates the `stdlib` and `validation` runtime profiles across Linux, macOS,
  Windows, and Python 3.10–3.14.

## Verified invariants

- Weak evidence returns `NOT_COMPUTABLE` rather than invented correspondence.
- Ordinary execution cannot promote authority or automatically mutate corpus
  confidence.
- Audience-facing outputs expose observable cinematic constraints, not private graph
  labels or named esoterica.
- Canonical storyboard outputs are byte-identical after normalization.
- Provider syntax changes do not silently alter critical semantic constraints.
- Continuity Forge, MCP, model providers, and vision providers remain optional.

## Post-v0.14 priorities

### P0

1. Complete review and merge of the official Nous Research Hermes Agent PR.
2. Align the OpenClaw branch with the v0.14 contracts without erasing its host-specific
   packaging and state model.

### P1

1. Add a Windows-native `install.ps1` with validation, receipt, and rollback parity.
2. Introduce five task-oriented workflow aliases: `create`, `revise`, `storyboard`,
   `inspect`, and `validate`, while retaining `do <intent>` for advanced operation.
3. Establish enforceable domain, application, adapter, and infrastructure boundaries.
4. Introduce frozen typed models for the highest-risk internal artifacts.
5. Add claim-level provenance and an explicit collision taxonomy.

### P2

1. Add role-specific guides for writers, directors, storyboard artists, generative
   artists, agent engineers, and corpus maintainers.
2. Add a visual end-to-end transformation example with passing and failing frames.
3. Add optional vision-provider normalizers and syntax-only video adapters.
4. Add privacy-preserving cross-project analytics from approved receipts only.
5. Expand the corpus only with provenance, misuse boundaries, mutation requirements,
   production-cost analysis, and regression coverage.

## Non-goals

- Kubrick is not becoming a mandatory Python package.
- External providers are not required for core creative work.
- The corpus does not evolve automatically.
- Host-specific overlays do not gain authority over the shared symbolic contract.
