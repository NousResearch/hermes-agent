# Retrieval, Synthesis, Validation, and Continuity Hardening (v0.5.0+)

This document operationalizes the corpus for deterministic use by Hermes / Kubrick.

## 1. Pattern Schemas (Machine-Readable)

All patterns must be representable in the schemas/ directory:
- symbolic-narrative-pattern.schema.json
- cinematic-pattern.schema.json
- transformation-grammar.schema.json
- narrative-affordance.schema.json
- symbolic-architecture.schema.json
- continuity-forge-symbolic-export.schema.json

When retrieving, output normalized records matching these schemas rather than free prose.

## 2. Retrieval Scoring

Before returning any pattern, compute:

```yaml
retrieval_score:
  dramatic_fit: 0-1
  character_fit: 0-1
  cultural_fit: 0-1
  cinematic_fit: 0-1
  source_quality: 0-1
  mutation_potential: 0-1
  continuity_compatibility: 0-1
  cliché_risk: 0-1   # higher = worse
```

Composite:
score = (dramatic_fit * 0.25) + (character_fit * 0.15) + (cultural_fit * 0.15) + (cinematic_fit * 0.15) + (source_quality * 0.10) + (mutation_potential * 0.10) + (continuity_compatibility * 0.10) - (cliché_risk * 0.20)

If best available score < 0.55 → return NOT_COMPUTABLE and fall back to simpler non-symbolic approach or request more context.

## 3. Negative Retrieval / Exclusion Profiles

Always run an exclusion pass.

Example exclusion_profile for "identity fragmentation":
- overused_patterns: ["broken mirror", "doubled face reflection", "black-and-white split", "unexplained twins"]
- visual_clichés: ["literal shattered glass", "two-sided face lighting"]
- genre_mismatches: ["pure comedy"]

Return explicit "excluded because..." with the profile.

## 4. Project Symbolic Ledger

Maintain across the project (handoff to Forge for persistence):

```yaml
project_symbolic_ledger:
  governing_grammar: "..."
  supporting_grammars: []
  active_motifs: [ {motif_id, state, last_scene, debt: true/false} ]
  retired_motifs: []
  prohibited_motifs: []
  unresolved_payoffs: []
  symbolic_debt: []          # seeded but not transformed
  saturation_score: 0.0
```

Before proposing a motif:
- Check if it is in prohibited or retired.
- Check symbolic_debt.
- Check saturation_score against budget.

## 5. Symbolic Saturation Control

Enforce budget:
- 1 governing grammar per project
- ≤2 supporting grammars
- ≤3 active motifs per sequence
- ≤2 symbolic channels per beat communicating the same function
- Every recurrence must mutate (state, ownership, geometry, sound, consequence)
- SYMBOLIC_OVERLOAD failure if violated.

In DIAGNOSE/REVISION mode, compute current saturation and flag.

## 6. Motif Collision Detection

Before finalizing a beat or sequence, run collision check:

collision_type enum: REDUNDANT | CONTRADICTORY | CULTURALLY_INCOMPATIBLE | VISUALLY_CONFUSING | RHYTHMICALLY_OVERLAPPING | PAYOFF_COMPETITION

## 7. Symbolic Counterpoint (Controlled Misalignment)

Do not force every channel to reinforce.

Valid counterpoint patterns:
- Image contradicts dialogue
- Music resists emotional valence
- Blocking suggests intimacy while dialogue creates distance
- Stable color palette while identity fractures
- Recurring shot returns without the expected sound
- Ritual form succeeds while emotional transformation fails

Record in symbolic_architecture.

## 8. Sequence-Level Symbolic Architecture

For every major sequence produce symbolic_sequence with governing_operation, motif_distribution, recurrence_schedule, inversion_point, cinematic_progression, exit_symbolic_state, residue.

## 9. Symbolic Character Arc Compilation

Translate internal change into operations using symbolic_character_arc: initial_relation, pressure_operations, threshold, irreversible_choice, retained_residue.

## 10. Revision Diffing

On any revision produce symbolic_revision_diff with preserved, removed, weakened, orphaned_setups, broken_payoffs, new_collisions, required_repairs.

## 11. Production Feasibility Weights

Attach production_cost scores. Prefer performance, blocking, sound, recurring props, controlled composition for constrained budgets.

## 12. Cultural Review Gates

Trigger when living_sacred_practice, marginalized_tradition, funerary_symbol, initiation_rite, deity_or_spirit_representation, indigenous_symbol, historically_persecuted_practice.

## 13. Corpus Versioning & Freshness

Every pattern must carry version, last_reviewed, source_status (DRAFT | PROVENANCE_COMPLETE | VALIDATED | CONTESTED | DEPRECATED | PROJECT_LOCAL), supersedes.

## Continuity Forge Round-Trip

Export using the forge export schema. On re-ingest, re-hydrate the project ledger. Verify motif identity and mutation history survived.
