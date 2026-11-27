# Kubrick v0.13.0 Release Notes

## Summary

Wave 2 and Wave 3 delivery: Forge multi-signal feedback, multi-provider generative adaptation, closed-loop differential visual QA, and optional MCP/CLI operators — without weakening fail-closed, proposal-only, or Forge-canonical governance.

**Tracking:** Issues [#3](https://github.com/scrimshawlife-ctrl/Kubrick/issues/3) and [#4](https://github.com/scrimshawlife-ctrl/Kubrick/issues/4) closed via [PR #24](https://github.com/scrimshawlife-ctrl/Kubrick/pull/24).

Also included on `main` in this generation of the skill: deterministic design-specification compilation (`design-build`).

## Wave 2 — Forge feedback, multi-signal evolution, project ledgers

- `forge-signals` extracts deterministic observation bundles from ledger diffs, revision records, saturation trends, collisions, ingestion results, and payoff realization.
- Every evolution event emits a multi-signal receipt covering confidence evidence, mutation success, production feasibility, anti-slop compliance, cultural-boundary respect, and payoff realization.
- Large confidence deltas and any structural or lifecycle mutation require a human review gate.
- Deprecation and retirement proposals are emitted for patterns that repeatedly create debt, collisions, or failed payoffs.
- Project symbolic ledgers persist pattern history (evidence-of-use, source projects, outcome confidence) and feed retrieval/evolution snapshots.
- Ledger commands: `init`, `audit`, `mutate`, `rehydrate`, `apply-forge`, `export-retrieval`, `record-pattern`.
- Structural changes never apply automatically. Forge remains canonical.

**Schemas:** `schemas/forge-signal-bundle.schema.yaml`, `schemas/multi-signal-evolution-receipt.schema.yaml`  
**Fixture:** `references/examples/forge-signals/ledger-before-after.yaml`

## Wave 3 — Model adapters, closed-loop QA, MCP operators

- Shared latent graph adapter path for Grok Imagine, Flux, SD3, and Midjourney.
- Adapters change provider syntax only; canonical symbolic intent and graph identity are immutable.
- Closed-loop QA pipeline: normalize observation → differential scoring → targeted correction → optional iteration governance.
- Fidelity reports expose geometry, state, residue, and convergence separately.
- CLI operators: saturation scoring, counterpoint, convergence-site locking, surface-occult audit, motif mutation, symbolic-architecture export.
- Optional stdio MCP server wraps the same CLI; MCP is never authoritative.
- Contemporary memetic/cultural-signal packs ship with provenance and `TIME_SENSITIVE` validity windows.

**Packs:** `references/patterns/contemporary/cultural-signal-packs/`  
**Contract:** `references/hermes-model-adapters.md`

## Design specification compiler

- `design-build` compiles heterogeneous creative/technical evidence into a schema-valid design specification.
- Weak or contradictory evidence fails closed; generated interpretation is never silently promoted to canonical fact.

**Contract:** `references/design-specification-compiler.md`  
**Schema:** `schemas/design-specification.schema.yaml`  
**Template:** `templates/design-specification.yaml`

## Compatibility

- Existing `adapt-grok`, outcome, storyboard, and compile workflows remain.
- New CLI commands are additive.
- Compiler provider choices expand to: `none`, `generic`, `grok-imagine`, `flux`, `sd3`, `midjourney`.
- External model APIs, vision APIs, MCP, and Continuity Forge remain optional.

## Verification

```bash
python scripts/kubrick.py validate-skill
python scripts/kubrick.py validate-corpus
python scripts/test_wave2_wave3.py
python scripts/test_outcome_governance.py
python scripts/test_design_specification.py
python scripts/kubrick.py repeatability --output out/kubrick/repeatability-report.json
python scripts/audit_release_version.py --strict
```

CI runs the above family of gates plus multi-provider storyboard compilation on pull requests and `main`.

## Documentation

| Doc | Purpose |
|---|---|
| `README.md` | Public overview and architecture diagram |
| `QUICKSTART.md` | Full operator workflows |
| `SKILL.md` | Hermes operating contract |
| `docs/README.md` | Documentation index |
| `docs/ROADMAP-v0.13.md` | Status and next priorities |
| `docs/RELEASE-CHECKLIST-v0.13.md` | Tag/publish gates |
