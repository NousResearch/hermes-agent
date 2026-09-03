# Changelog — Kubrick

Kubrick is the primary symbolic cinematic narrative engineering skill. It replaces the earlier scriptwriting skill as the recommended system for premise-to-production development with deep motif, geometric, cinematic, storyboard, and visual-fidelity encoding.

## [Unreleased]

## [0.14.0] - 2026-07-30

### Added — Deterministic contract consolidation

- Added `kubrick.manifest.yaml` as the stdlib-readable canonical Hermes registry for runtime profiles, intents, actions, aliases, recipes, schemas, providers, artifacts, authority classes, and exit codes.
- Added `pyproject.toml` to declare Python compatibility and optional `validation` / `dev` tooling without turning Kubrick into a required Python package.
- Added manifest validation, router parity tests, release-version alignment, and a cross-platform stdlib CI matrix.
- Added deterministic compile-receipt identities for the Kubrick release, corpus, schema bundle, provider adapter, command options, and normalized semantic input.
- Added mutation-sensitive identity tests and compile-receipt repeatability coverage.
- Added a canonical structured diagnostic schema and failure matrix for router errors, unavailable validation dependencies, and compiler `NOT_COMPUTABLE` paths.
- Added provider semantic-preservation reports covering graph identity, required observable content, ownership, geometry, state change, residue, continuity, and negative constraints.
- Added isolated lifecycle tests for dry-run, staged validation, fresh install, upgrade backup, rollback, receipts, and validation-failure safety.

### Changed

- The unified Hermes intent router now derives its registry, aliases, and recipes from the canonical manifest.
- The smoke gate now validates the manifest before the skill and corpus.
- The unified router now supports `KUBRICK_DIAGNOSTICS=json`; compiler `NOT_COMPUTABLE` consistently exits with manifest code `4`, while missing required optional tooling exits `3`.
- Grok Imagine now uses the shared provider adaptation path; all adapters fail closed when a critical semantic invariant is lost.
- The Hermes installer now validates before activation, atomically swaps installations, writes receipts, restores on activation failure, and supports `--dry-run`, `--rollback`, and `--version`.

### Changed — Operator intent router

- Primary CLI surface is `kubrick do <intent> [--action …]` (12 intents) for Hermes and humans.
- Soft aliases preserve all prior top-level command names (`adapt-flux`, `closed-loop-qa`, `validate-skill`, …).
- Help lists intents only; sugar includes `help`, `recipe`, and `aliases`.
- Recipes: `recipe storyboard-example`, `recipe verify`.
- MCP: single tool `kubrick_do` over the same router (never authoritative).
- Implementation: `scripts/intent_router.py`; design/plan under `docs/superpowers/`.

### Documentation — OpenClaw edition pointer

- Added `docs/OPENCLAW.md` documenting the permanent **`openclaw`** branch Agent Skill packaging contributed by **Prabu** ([@prabu-openclaw](https://github.com/prabu-openclaw); [PR #1](https://github.com/scrimshawlife-ctrl/Kubrick/pull/1), commits `77fa721`, `15defae`).
- Cross-links from README, QUICKSTART, SKILL, and docs index so Hermes `main` and OpenClaw users can find the correct edition.

## [0.13.0] - 2026-07-30

### Added — Wave 2: Forge Feedback, Multi-Signal Evolution, Project Ledgers

- `scripts/extract_forge_signals.py` extracts multi-signal observation bundles from ledger diffs, revision records, saturation trends, collisions, ingestion results, and payoff realization.
- `schemas/forge-signal-bundle.schema.yaml` and `schemas/multi-signal-evolution-receipt.schema.yaml`.
- Multi-signal evolution proposals covering mutation success, production feasibility, anti-slop compliance, cultural-boundary respect, and payoff realization.
- Human review gates for large confidence deltas and structural or lifecycle mutations.
- Retirement/deprecation proposals for patterns that repeatedly create debt, collisions, or failed payoffs.
- First-class project symbolic ledgers with pattern history (evidence-of-use, source projects, outcome confidence), forge rehydrate/apply, and retrieval snapshots.
- Example Forge signal fixture under `references/examples/forge-signals/`.

### Added — Wave 3: Model Adapters, Closed-Loop QA, MCP Operators

- Shared latent-graph adapters for Flux, SD3, and Midjourney (plus existing Grok Imagine) via `adapt_provider.py`.
- Compiler provider support for `flux`, `sd3`, and `midjourney`.
- `closed_loop_visual_qa.py` orchestration with separate geometry, state, residue, and convergence fidelity reporting.
- CLI operators: saturation scoring, counterpoint, convergence-site locking, surface-occult audit, motif mutation, symbolic-architecture export.
- Optional stdio MCP server (`mcp_kubrick_server.py`) wrapping the same CLI without becoming authoritative.
- Time-sensitive contemporary cultural-signal packs with explicit provenance and validity windows.

### Governance

- Forge remains canonical authority; Kubrick emits observations and proposals only.
- No structural change applies automatically.
- Adapters never alter canonical symbolic intent.
- Operators fail closed on weak evidence and emit auditable receipts.
- Audience-facing prompts remain free of named esoterica unless explicitly requested.

### Added — Design Specification Compiler

- `design-build` compiles heterogeneous creative and technical evidence into a schema-valid design specification.
- Schema `schemas/design-specification.schema.yaml`, template `templates/design-specification.yaml`, and contract `references/design-specification-compiler.md`.

### Changed

- Skill version advanced from `0.12.0` to `0.13.0`.
- Unified CLI exposes forge-signals, multi-provider adapt, closed-loop-qa, operator, mcp-server, and design-build commands.
- README, QUICKSTART, SKILL, and `docs/` fully document the Wave 2/3 operator surface and shipped issue status.

## [0.12.0] - 2026-07-29

### Added — Unified Symbolic Storyboard Compiler

- Unified `scripts/kubrick.py` operator CLI.
- Compiler `0.3.0` orchestration from brief through retrieval, private graph, structured audit, audience translation, storyboard propagation, provider adaptation, and compile receipt.
- Draft 2020-12 artifact validation with exact failing paths.
- Canonical authority-transfer storyboard example and CI compilation.
- Storyboard state propagation for ownership, node, object, light, material, residue, and convergence continuity.
- Frame-transition comparison and prohibited-reset enforcement.
- Provider-neutral model-adapter packet contract.
- First Grok Imagine prompt adapter without mandatory credentials, network access, or model invocation.

### Added — Closed-Loop Visual QA

- Structured visual-observation schema and normalization for human, generic JSON, and optional Grok Vision outputs.
- Dimension-specific visual-fidelity reports for geometry, state, ownership, object, light, material, residue, convergence, and continuity.
- Targeted correction packets that preserve passing dimensions and change only observed mismatches.
- Bounded correction-loop governance with progress, regression, iteration-limit, and human-review decisions.
- Complete Grok generation/review bundle manifests and observation templates.

### Added — Governed Outcome Learning

- Pattern-use receipts recording retrieval confidence, compile validity, production feasibility, visual fidelity, correction count, payoff realization, cultural-boundary compliance, and operator outcome.
- Proposal-only pattern-evolution packets with bounded confidence deltas, misuse-risk additions, mutation-variable additions, and lifecycle review actions.
- Explicit prohibition on automatic corpus mutation or autonomous authority promotion.

### Added — Release and Runtime Hardening

- Deterministic repeatability check across two clean canonical compiles.
- Stable hashes for private graph, structured packet, storyboard state, transition report, neutral adapter packet, and Grok Imagine packet.
- Release checklist, release notes, current roadmap, authoritative `VERSION` manifest, and strict release-version audit.
- CI coverage for Hermes evals, outcome governance, canonical storyboard compilation, schema validation, and repeatability.
- Legacy sidecar normalization, qualitative production-cost support, structured ledger motif support, timestamp-safe deterministic hashing, and underscore-delimited route matching.

### Changed

- Skill version advanced from `0.11.0` to `0.12.0`.
- README and SKILL contract now describe the unified storyboard, adapter, visual-QA, and outcome-governance workflow.
- Outcome learning is observation- and proposal-based; ordinary execution never rewrites corpus confidence or lifecycle state.
- Continuity Forge, MCP, model APIs, and vision APIs remain optional extensions.

### Governance

- Local creative output remains `PROPOSED`.
- Production-use evidence remains `OBSERVATION`.
- Weak or invalid evidence returns `NOT_COMPUTABLE`.
- Pattern evolution cannot be applied automatically.
- Private pattern and lexicon semantics remain excluded from audience-facing packets.

## [0.10.0] - 2026-07-29

### Added — Neuro-Symbolic & Compositional Upgrades

- **Internal Motif/Structure Graph Layer**: Lightweight neuro-symbolic graph (nodes = motifs + observed states; edges = relational pressures and state transformations) used internally by the Dynamic Selection Engine. Translated exclusively into observable constraints.
- **Explicit Disentanglement of Cinematic Systems**: Layout/Geometry, Semantics/Function, and Attributes/States factored (SL-VAE style) to reduce leakage and enable cleaner interlocking.
- **Compositional Layered Encoding**: Analogs to Compositional Masked Attention (CMA) and Multi-Layered Sampler (MLS) — convergence masking, independent layers with controlled interaction, prioritized constraints.
- **Conditioning-Style Prompt Engineering**: Prompt sections act as precise control signals (geometry, state differentials, dual light, object participation).
- **Neuro-Symbolic Predicate Validation**: Internal combinatorial coherence checks treating lexicon entries as observable predicates.
- **Enhanced Single-Frame State Modeling**: Stronger before/after differentials, persistent residue as charge, layered time within one frame.

- Added section 17 "Neuro-Symbolic Motif Graph Layer" to references/symbolic-dramaturgy.md.
- Added "Neuro-Symbolic Conditioning Patterns (0.10.0+)" major section to references/cinematic-symbolism-corpus.md.
- SKILL.md bumped to 0.10.0 with full upgrade description.
- Goal: Higher relational precision, convergence strength, and controllable symbolic density for narrative + single-frame work while preserving full latency and anti-slop rules.

## [0.9.0] - 2026-07-29

### Added — Dynamic Latent Esoteric–Alchemical Encoding

- Full **Kubrick Esoteric–Alchemical Encoding Lexicon** (200+ entries across Hermetic Principles, Classical Alchemical Operations, Magnum Opus stages, Substances, Vessels, Polarity, Geometry, Elements, Planetary Forces, Initiatory Structures, Mythic Archetypes, Magical Operations, Chaos Magic, Theurgy, Kabbalah, Gnosticism, Neoplatonism, Time/Recursion, Shadow work, Dream/Astral, Divination, Ritual, Color, Sound, and Kubrick-native hidden structures).
- **Dynamic Esoteric–Alchemical Selection Engine**:
  - Analyzes prompt content or resulting image description.
  - Dynamically selects and interlocks multiple lexicon concepts.
  - Applies exclusively as observable structural constraints (state differentials, convergence points, relational pressure, recurrence-with-mutation, etc.).
  - **Latent operation**: No automatic disclosure or naming. Outputs are significantly more symbolically dense by default.
  - Explicit user query only ("what esoteric concepts were used?") triggers private report.
- New reference file: `references/esoteric-alchemical-lexicon.md`.
- Updated cinematic-symbolism-corpus.md with full Dynamic Selection rules, triggers, density mandates, and feedback loop for image analysis.
- SKILL.md updated with 0.9.0 capabilities and latent encoding behavior.
- Goal: Produce much higher symbolic density through content-driven, interlocking esoteric grammar without wasting tokens on explanation.

See `examples/beach-threshold.md` for baseline; 0.9.0+ runs will be denser.

## [0.8.0] - 2026-07-29

### Added — Profound Esoteric & Single-Frame Integration

- **Esoteric Structural Translation** in `references/cinematic-symbolism-corpus.md`
  - Formal translation layer for ancient magical and esoteric concepts (liminal/threshold, trace as binding/erasure, witness objects, erasure as active operation, inversion as crossing, dual preservation/dissolution, residue/charge).
  - All concepts enter exclusively via enforceable structural rules and constraint — never naming, iconography, or occult collage.
  - New "Core Translation Principles", "Rules for Profound Esoteric Work", and "Single-Frame Esoteric Mappings".

- Expanded **Cinematic Systems** in corpus: Light Systems, Geometry & Negative Space, Material & Trace, Inversion & Reflection, Perceptual Layering & Residue (with profound interlocking requirements).
- **Single-Frame & Generative Image Translation** section with concrete mappings, test criteria, and enforcement via Corpus Integration Rules.
- Integration points added across symbolic-dramaturgy.md, SKILL.md, and templates/production-handoff.md for esoteric and profound single-frame work.
- Stronger requirement for image prompts: interlocking from Light + Geometry + Material/Trace + Inversion at convergence point with state differentials.

## [0.7.1] - 2026-07-29
### Added — Autonomous Evolution from Use
- `scripts/evolve_from_use.py` — self-improvement engine
  - Aggregates retrieval receipts (auto-logged by retrieve script)
  - Incorporates project outcomes (success/failure signals)
  - Adjusts `confidence` in sidecars
  - Appends `usage_history` with performance data
  - Re-orders suggestions in `corpus-index.yaml` based on observed results
  - Emits auditable `evolution-*.json` receipts
- Auto-logging added to retrieval script (`references/usage/receipts/`)
- Seeded example usage data (receipts + outcomes)
- New procedures documented in SKILL.md under "Evolution from Use"
- Sidecars now carry usage-driven metadata

The corpus now improves from real application in projects and Forge workflows without manual curation for every pattern.

> **Superseded in 0.12.0:** automatic corpus changes are no longer part of the supported workflow. Outcome evidence now produces human-reviewed proposals only.

## [0.7.0] - 2026-07-29
### Added — Executable Retrieval (P0 of next campaign)
- `scripts/retrieve_symbolic_patterns.py` — deterministic retrieval helper
  - Loads index + sidecars
  - Scores with decomposition
  - Applies exclusions/prohibited
  - Emits structured retrieval_receipt
  - Fails closed below threshold
- Initial sidecar patterns in `references/patterns/` (9 high-value: alchemical nigredo, Kubrick monolith, Tarkovsky reflection, Bresson hands, Propp, Peirce, liminal, acousmatic, Denis marching)
- `evals/retrieval/` structure with inputs/ and expected/ golden fixtures
- 3 new formal tradition packs: Soviet montage, Japanese cinema, Animation
- SKILL.md documentation for the retrieval script

This begins the shift from reference skill to operational symbolic compiler.

## [0.6.2] - 2026-07-28 (further continuation)
### Added
- Additional genre packs: melodrama, comedy, low-budget/TV/short-form
- More populated schema examples (symbolic counterpoint, project ledger)
- Forge round-trip validation example and test
- Deeper film-pattern provenance depth for key examples (Kubrick Monolith, Tarkovsky, Bresson)
- Production feasibility test
- Expanded validation for round-trip and feasibility
- Updated SKILL.md with Forge round-trip procedure

### Expanded
- Genre and constrained-production guidance
- Transferable structure documentation in cinematic corpus

## [0.6.1] - 2026-07-28 (continuation after merge of #42)
### Added
- Populated schema examples (alchemical-nigredo, kubrick-monolith)
- Dedicated genre packs: horror, noir-thriller, science-fiction (under references/corpus/genre/)
- Concrete revision diff example (scene deletion)
- Additional validation tests for sequence/character arc and cultural review gates
- Concrete "How to Use" procedures in SKILL.md for scoring, ledger, sequence, character arc, revision, and handoff

### Expanded
- Film-pattern provenance depth examples integrated
- Cultural review trigger documentation and test
- Production feasibility notes in genre packs

This continues the P1/P2 hardening after the P0 merge.

## [0.6.0] - 2026-07-28
### Added — Symbolic Retrieval and Continuity Hardening (P0)
- Machine-readable schemas/ (symbolic-narrative-pattern, cinematic-pattern, transformation-grammar, narrative-affordance, symbolic-architecture, continuity-forge-symbolic-export)
- Retrieval scoring model with composite formula and NOT_COMPUTABLE threshold
- Negative retrieval / exclusion_profiles
- Project symbolic ledger (governing/supporting grammars, active/retired/prohibited motifs, symbolic_debt, saturation_score)
- Symbolic saturation control + SYMBOLIC_OVERLOAD
- Motif collision detection (REDUNDANT, CONTRADICTORY, etc.)
- Symbolic counterpoint rules
- Sequence-level symbolic_architecture and symbolic_character_arc compilation
- Symbolic revision diff engine
- Production feasibility weights
- Cultural review gates
- Corpus versioning and source_status (DRAFT → VALIDATED → DEPRECATED)
- Expanded cinematic corpus (genre packs, production scales, formal traditions: Soviet montage, Neorealism, Japanese, Hong Kong action, etc.)
- Film-pattern provenance depth (transferable_structure vs non_transferable_surface)
- references/retrieval-and-continuity.md
- Updated corpus-index.yaml with scoring, exclusions, ledger templates
- New validation cases for scoring, exclusion+collision, ledger+saturation+revision
- Forge round-trip validation expectations

### Changed
- SKILL.md now documents deterministic retrieval procedures as first-class
- All pattern work now expected to be schema-normalized + scored + collision-checked + ledger-tracked
- Cinematic examples now include production_cost and transferable vs surface distinction

This sprint converts the rich corpus into a reliable, auditable retrieval-and-continuity system.

## [0.5.0] - 2026-07-28
### Added
- Full provenance-linked Symbolic Narrative Pattern System
- `SymbolicNarrativePattern` core YAML schema with observed_structure, cinematic_affordances, mutation_rules, misuse_risks, and full source_records provenance
- Narrative Affordance Registry (BIND, DIVIDE, INITIATE, CONCEAL/REVEAL, INVERT, REPEAT, CONTAMINATE, MIRROR, SACRIFICE, CROSS, ENCLOSE, DESCEND, RETURN, HAUNT, ERASE, RESTORE + full mappings)
- Transformation Grammar Registry (alchemical processes, initiation, contamination, fragmentation, descent/return mapped to narrative + cinematic forms)
- Dedicated Cinematic Symbolism Corpus (techniques tracked as patterns with no fixed meanings)
- 10 corpus domains with starter PRIMARY/SCHOLARLY-anchored patterns (alchemical, ritual-liminal, cinematic)
- Source Hierarchy (PRIMARY / EARLY_COMMENTARY / SCHOLARLY / PRACTITIONER / COMPARATIVE / POPULAR / INTERNET)
- Cross-Tradition Relationship Types (HISTORICALLY_DERIVED, SHARED_FUNCTION, FORMAL_RESEMBLANCE, MODERN_SYNTHESIS, CONTESTED, UNSUPPORTED, etc.)
- Skill Retrieval Rules, Quality Gates, and 10 Validation Tests
- `corpus-usage.md`, `source-hierarchy.md`, `source-registry.md`, `cross-tradition-relationships.md`

### Changed
- kubrick positioned and documented as the primary/replacement symbolic cinematic skill
- Enhanced documentation across README, hermes docs, and internal references
- Updated retrieval discipline to prioritize dramatic problem → one primary grammar → at most two secondary → cinematic form → provenance

Kubrick now provides a rigorous, auditable bridge from historically grounded symbolic systems to subtle scene, character, blocking, composition, editing, and sound structures.

## [0.4.0] - 2026-07-28

### Added — Public-Ready Improvements

- Fixed frontmatter to Hermes standards.
- Added symbolic-specific eval rubric and regression cases.
- Strengthened observed-form-first, mutation, relational geometry, and anti-slop enforcement.
- Restructured SKILL.md closer to canonical Hermes skill format.
- Added concrete YAML examples of symbolic intent, motif lifecycle, and cinematic encoding.

## [0.3.0] - 2026-07-28

Initial public packaging of symbolic upgrade, Module 5B, Gates M–W, and Forge integration.

## Earlier

See scriptwriting history for base narrative engineering foundations. Kubrick supersedes that base for projects requiring cinematic symbolic precision.
