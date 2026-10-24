# Kubrick v0.14.0 Release Notes

## Deterministic Contract Consolidation

Kubrick v0.14 turns the existing symbolic-cinematic compiler into a more dependable
standalone Hermes skill. This release emphasizes reproducibility, dependency clarity,
structured failure, provider invariants, installation safety, and official Hermes
packaging rather than adding new symbolic ontology.

## Highlights

### One canonical manifest

`kubrick.manifest.yaml` now governs runtime profiles, intents, actions, aliases,
recipes, artifact and schema registries, providers, authority classes, and exit codes.
The CLI router derives its public registries from this file.

### Explicit runtime profiles

- `stdlib`: manifest, routing, skill validation, diagnostics, and release checks
- `validation`: YAML and JSON Schema validation plus compile/adaptation integration
- `dev`: full regression, governance, repeatability, installer, and corpus tooling

Kubrick remains a Hermes skill loaded from its directory. `pyproject.toml` documents
tooling contracts and does not make package installation mandatory.

### Reproducible receipts

Compile receipts now include Kubrick, corpus, schema-bundle, provider-adapter,
normalized-input, and command identities. Mutation controls prove that each digest
changes when its governed input changes, while canonical normalized outputs remain
byte-identical across repeated runs.

### Structured failure

The router and compiler emit schema-valid diagnostics with stable exit codes:

- `2`: invalid command
- `3`: required optional dependency unavailable
- `4`: `NOT_COMPUTABLE`

Human-readable stderr remains the default. Set `KUBRICK_DIAGNOSTICS=json` for machine
consumers.

### Provider semantic preservation

Generic, Grok Imagine, Flux, SD3, and Midjourney adapters emit preservation reports
covering graph identity, required content, ownership, geometry, state change, residue,
continuity, and negative constraints. Critical semantic loss fails closed as
`PROVIDER_LOSS`.

### Atomic Hermes installation

The Bash installer now supports staged validation, atomic activation, external
backups, install receipts, `--dry-run`, `--rollback`, and `--version`. A failed staged
validation or activation leaves the prior installation intact.

### Modern Hermes skill contract

`SKILL.md` now uses the required Hermes section order and a concise activation-focused
description. Official packaging includes the storyboard recipe fixtures and a
repository-level contract test.

## Verification summary

The release gates cover:

- Linux, macOS, and Windows
- Python 3.10 through 3.14
- stdlib and validation profiles
- 22 Hermes behavioral and adversarial evaluations
- manifest/router parity and stable diagnostics
- deterministic receipt identity and repeatability
- provider semantic preservation and loss injection
- installer fresh install, upgrade, rollback, and failure safety
- outcome governance, Wave 2/3, design-specification, and authority boundaries
- canonical and exported Hermes skill layouts

## Compatibility

- Existing `do <intent>` commands and legacy aliases remain supported.
- Continuity Forge, MCP, image providers, and vision providers remain optional.
- OpenClaw remains on its permanent host-specific branch and requires a follow-up
  contract-alignment pass tracked by issue #32.

## Release documents

- `docs/ROADMAP-v0.14.md`
- `docs/RELEASE-CHECKLIST-v0.14.md`
- `CHANGELOG.md`
