# Kubrick Documentation

Current release: **0.14.0** (see root `VERSION`).

## Start here

| Doc | Audience | Purpose |
|---|---|---|
| [`../README.md`](../README.md) | Everyone | Public overview, install, architecture |
| [`../QUICKSTART.md`](../QUICKSTART.md) | Operators | Intent-first command workflows + recipes |
| [`../SKILL.md`](../SKILL.md) | Hermes runtime | Canonical skill operating contract |
| [`HERMES-OFFICIAL-SUBMISSION.md`](HERMES-OFFICIAL-SUBMISSION.md) | Maintainers | Official optional-skills PR + community publish checklist |
| [`OPENCLAW.md`](OPENCLAW.md) | OpenClaw users | OpenClaw Agent Skill edition (Prabu; branch `openclaw`) |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Maintainers | Release history |

## Runtime editions

| Edition | Branch | Notes |
|---|---|---|
| **Hermes** | [`main`](https://github.com/scrimshawlife-ctrl/Kubrick/tree/main) | Canonical Hermes skill; deterministic v0.14 contract |
| **OpenClaw** | [`openclaw`](https://github.com/scrimshawlife-ctrl/Kubrick/tree/openclaw) | Agent Skill packaging by **Prabu** ([PR #1](https://github.com/scrimshawlife-ctrl/Kubrick/pull/1)); external state dir; doctor / portability tests |

See [`OPENCLAW.md`](OPENCLAW.md) for install commands, credit, and what differs between editions.

## Operator surface (intent router)

Primary CLI (Hermes + humans):

```bash
python scripts/kubrick.py do <intent> [--action <action>] [flags]
```

**12 intents:** `compile` · `retrieve` · `ledger` · `design` · `storyboard` · `adapt` · `visual` · `learn` · `check` · `operate` · `mcp` · `bundle`

| Doc | Purpose |
|---|---|
| [`superpowers/specs/2026-07-30-operator-intent-router-design.md`](superpowers/specs/2026-07-30-operator-intent-router-design.md) | Design: intents, aliases, UX, MCP |
| [`superpowers/plans/2026-07-30-operator-intent-router.md`](superpowers/plans/2026-07-30-operator-intent-router.md) | Implementation plan |
| [`../scripts/intent_router.py`](../scripts/intent_router.py) | Registry + resolve + help |
| [`../QUICKSTART.md`](../QUICKSTART.md) | Human recipes and examples |

Legacy flat command names remain soft aliases. MCP exposes a single tool: `kubrick_do`.

## Current release (v0.14)

| Doc | Purpose |
|---|---|
| [`ROADMAP-v0.14.md`](ROADMAP-v0.14.md) | Shipped scope + next priorities |
| [`RELEASE-NOTES-v0.14.md`](RELEASE-NOTES-v0.14.md) | What landed in 0.14.0 |
| [`RELEASE-CHECKLIST-v0.14.md`](RELEASE-CHECKLIST-v0.14.md) | Release gates and tag procedure |

Wave 2 ([#3](https://github.com/scrimshawlife-ctrl/Kubrick/issues/3)) and Wave 3 ([#4](https://github.com/scrimshawlife-ctrl/Kubrick/issues/4)) shipped via [PR #24](https://github.com/scrimshawlife-ctrl/Kubrick/pull/24).

## Historical release docs

| Doc | Notes |
|---|---|
| [`ROADMAP-v0.13.md`](ROADMAP-v0.13.md) | v0.13 Wave 2/3 and router context |
| [`RELEASE-NOTES-v0.13.md`](RELEASE-NOTES-v0.13.md) | v0.13 release notes |
| [`RELEASE-CHECKLIST-v0.13.md`](RELEASE-CHECKLIST-v0.13.md) | v0.13 release gates |
| [`ROADMAP-v0.12.md`](ROADMAP-v0.12.md) | v0.12 status; points forward to 0.13 |
| [`RELEASE-NOTES-v0.12.md`](RELEASE-NOTES-v0.12.md) | v0.12 compiler / storyboard / outcome learning |
| [`RELEASE-CHECKLIST-v0.12.md`](RELEASE-CHECKLIST-v0.12.md) | v0.12 gates (historical) |
| [`ROADMAP-v0.11.md`](ROADMAP-v0.11.md) | Original Wave 1–3 production-hardening plan |

## Reference contracts

| Doc | Purpose |
|---|---|
| [`../references/hermes-runtime-contract.md`](../references/hermes-runtime-contract.md) | Runtime, artifacts, dependencies, canon |
| [`../references/hermes-model-adapters.md`](../references/hermes-model-adapters.md) | Neutral packet + provider adapters |
| [`../references/hermes-visual-qa.md`](../references/hermes-visual-qa.md) | Observation, fidelity, correction loop |
| [`../references/hermes-storyboard-state.md`](../references/hermes-storyboard-state.md) | Multi-frame state propagation |
| [`../references/hermes-graph-operators.md`](../references/hermes-graph-operators.md) | Graph construction operators |
| [`../references/continuity-forge-integration.md`](../references/continuity-forge-integration.md) | Forge handoff + multi-signal feedback |
| [`../references/design-specification-compiler.md`](../references/design-specification-compiler.md) | Design-spec compiler |
| [`../references/retrieval-and-continuity.md`](../references/retrieval-and-continuity.md) | Retrieval discipline |
| [`../references/anti-slop-patterns.md`](../references/anti-slop-patterns.md) | Anti-slop gates |

## Schemas

| Schema | Purpose |
|---|---|
| `schemas/forge-signal-bundle.schema.yaml` | Multi-signal Forge observation bundle |
| `schemas/multi-signal-evolution-receipt.schema.yaml` | Deterministic evolution receipt |
| `schemas/design-specification.schema.yaml` | Governed design specification |
| `schemas/project-symbolic-ledger.schema.yaml` | Project ledger (+ pattern history) |
| `schemas/pattern-evolution-proposal.schema.yaml` | Proposal-only evolution (+ review gates) |

Full schema set: [`../schemas/`](../schemas/).

## Examples and fixtures

| Path | Purpose |
|---|---|
| `examples/authority-transfer-storyboard/` | Canonical three-frame storyboard |
| `references/examples/forge-signals/` | Forge multi-signal extraction fixture |
| `references/patterns/contemporary/cultural-signal-packs/` | Time-sensitive cultural packs |
| `templates/design-specification.yaml` | Design-spec template |

## Verification commands

```bash
python scripts/kubrick.py validate-skill
python scripts/kubrick.py validate-corpus
python scripts/kubrick.py coverage
python scripts/kubrick.py eval
python scripts/test_outcome_governance.py
python scripts/test_wave2_wave3.py
python scripts/test_design_specification.py
python scripts/kubrick.py repeatability --output out/kubrick/repeatability-report.json
python scripts/audit_release_version.py --strict
```
