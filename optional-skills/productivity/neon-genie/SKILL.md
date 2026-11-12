---
name: neon-genie
description: Build evidence-bound product and opportunity packets.
version: 3.25.0
author: Daniel Meyer (@scrimshawlife-ctrl), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
dependencies: []
metadata:
  hermes:
    tags: [Product, OpportunityIntelligence, ZeroOption, EvidenceBound, AdvisoryOnly]
    category: productivity
    related_skills: []
triggers:
  - neon genie
  - product audit
  - opportunity mining
  - zero option
  - wayfinder handoff
  - commercial simulation
  - evidence intelligence
  - capital sprint
---

# Neon Genie Skill

Neon Genie turns weak signals and blocked transitions into evidence-bound product and opportunity packets. It is **advisory only**: draft and recommend; never spend, publish, contact, or mutate repositories. Full releases and packaging live upstream: https://github.com/scrimshawlife-ctrl/NeonGenie

## When to Use

- Product audits, boundaries, and Wayfinder-ready handoffs
- Opportunity mining, zero-capital first-cash loops, fragmentation scans
- Roadmaps and approaches for constrained / solo builders
- Commercial framing when buyer roles may be missing
- Evidence gaps that need host research or a private `DataRequest`

Do **not** use for cinematic work (Kubrick), code execution, spend, or publication.

## Prerequisites

- Hermes Agent with optional skills enabled
- Python 3.11+ (stdlib packaging CLI only)
- Optional: host tools such as `web_search` / `web_extract` for public facts
- Optional: Wayfinder as handoff consumer only

Profiles and schemas: `references/profiles/`, `references/schemas/`. Privacy contract: `PRIVACY.md` and `references/PRIVACY.md`. Gates: `references/gates.yaml`.

## How to Run

Skill directory = folder containing this `SKILL.md`.

```bash
python scripts/neon_genie.py do doctor
python scripts/neon_genie.py do run --recipe product-audit --out out/neon-genie/demo
python scripts/neon_genie.py do run --brief examples/product-audit.brief.yaml --out out/neon-genie/demo
python scripts/neon_genie.py do route --text "first cash zero capital" --json
python scripts/neon_genie.py do privacy --json
python scripts/neon_genie.py do validate --packet out/neon-genie/demo/run-envelope.json --type envelope --strict-authority
python scripts/neon_genie.py do capabilities --json
```

In chat, load the skill and describe the job in plain language. Always run **OPEN → ALIGN → ASCEND → CLEAR → SEAL**. Prefer the smallest profile set (`core` + `privacy` always).

**Default job (transitional builders):** name the stuck point and “done”; capture constraints without inventing resources; find public → request private → label claims; emit roadmap and/or approaches with completion proof; seal as drafts only.

Example prompt:

```text
Use Neon Genie. I'm between jobs with limited money and an app idea.
I need a realistic roadmap and first approaches I can actually run.
Do not invent buyers, capital, or skills I did not declare.
Research public facts if you can; request private facts with DataRequest.
Label every important claim. Advisory only — do not modify any repo.
```

Open **`run-envelope.json`** first when resuming packaging work.

## Quick Reference

| Item | Value |
|------|--------|
| Authority | `advisory_only` — `grants_execution: false` |
| Claim labels | `OBSERVED` · `INFERRED` · `SPECULATIVE` · `NOT_COMPUTABLE` |
| Missing public fact | research via host tools, then cite or drop |
| Missing private fact | emit `DataRequest` |
| Still missing | `NOT_COMPUTABLE` — never invent |
| Privacy default | packaging `local_only`; see `do privacy --json` |
| Entry artifact | `run-envelope.json` |
| Profiles | `references/profiles/core.md` (+ siblings; `privacy` always on) |
| Schemas | `references/schemas/run-envelope.schema.json` (+ siblings) |

Recipes: `product-audit`, `zero-option`, `zero-option-executable`, `fragmentation`, `commercial`, `audit`, `agentic`, `memetic`, `evidence`, `opportunity`, `capital-sprint`.

## Procedure

1. **OPEN** — Request, actor, current/desired state, constraints, artifact type. Authority is advisory only.
2. **ALIGN** — Operator evidence → workspace → host research → model prior as `SPECULATIVE` only. Gap-detect; privacy egress check before host tools; auto-load `evidence_intelligence` when external facts change the answer.
3. **ASCEND** — Topology, thesis, intervention, scorecard, route. Label every material claim. Smallest sufficient profiles.
4. **CLEAR** — Fail closed on authority leaks, uncited OBSERVED, missing DataRequest for private decision-critical fields, missing completion proof at TESTABLE+, buyer/beneficiary conflation, privacy gates S–Y as applicable.
5. **SEAL** — Packet(s) + receipt + **`run-envelope.json`**. List `data_requests`, `research_attempts`, privacy provenance. Wayfinder handoffs freeze product intent.

**Wayfinder boundary:** Neon Genie owns what/why/user/boundary/proof. Wayfinder owns decomposition/milestones/status. Intent changes return here.

## Pitfalls

- Treating model prior as `OBSERVED`
- Inventing buyers, capital, access, or credentials under zero-option constraints
- Granting spend/publish/mutate in packets or chat
- Letting Wayfinder rewrite product intent
- Skipping research on public fetchable gaps (Gate P) or inventing private facts without `DataRequest` (Gate Q/R)
- Claiming absolute privacy without matching mode and receipt evidence
- Opening many profiles “just because”

## Verification

```bash
python scripts/neon_genie.py do check
python scripts/neon_genie.py do doctor
python scripts/neon_genie.py do eval
python scripts/neon_genie.py do privacy --json
python scripts/neon_genie.py do run --recipe zero-option --out out/neon-genie/verify-zero
python scripts/neon_genie.py do validate --packet out/neon-genie/verify-zero/run-envelope.json --type envelope --strict-authority
```

Expect: doctor green; zero-option path does not invent resources; every envelope has `authority: advisory_only` and `grants_execution: false`.

Upstream monorepo and releases: https://github.com/scrimshawlife-ctrl/NeonGenie
