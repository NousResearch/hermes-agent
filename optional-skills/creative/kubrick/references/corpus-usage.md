# Skill Retrieval Rules, Quality Gates, and Validation Tests

## Retrieval Rules (when using the corpus in any mode)

1. Identify the dramatic problem first.
2. Select one primary symbolic grammar (e.g., one transformation process or affordance).
3. Select at most two secondary grammars.
4. Select relevant cinematic-form patterns from cinematic-symbolism-corpus.md.
5. Prefer process and relation over recognizable iconography.
6. Respect the culture, setting, and character context of the story.
7. Retrieve source provenance with each pattern (link to source_records).
8. Exclude unsupported correspondences (see cross-tradition-relationships.md).
9. Generate subtle enactment (behavior, space, sound, editing, blocking) rather than direct explanation.
10. Preserve ambiguity while maintaining clear causal storytelling. Never weaken causality.

## Quality Gates (in addition to A–W in anti-slop-patterns.md)

- Unsupported symbolic equivalence across traditions.
- Occult collage (arbitrary mixing without grammar).
- Symbol-dictionary reasoning (one-to-one lookup).
- Universal dream-symbol claims without personal/cultural grounding.
- One-to-one symbolism.
- Decorative sacred geometry without dramatic function.
- Arbitrary numerology without structural effect.
- Iconography without functional enactment.
- Cultural flattening.
- Cinematic cliché (low angle = power, etc.).
- Symbolism that requires exposition to be legible.
- Repeated motifs without mutation.
- Archetype naming without observable behavior/role.
- Symbolism that compromises character agency or spatial clarity.

Flag in DIAGNOSE mode with evidence and repair using the relevant schema (symbolic_intent, observed_structure, etc.).

## Validation Tests (implemented in evals/)

1. One dramatic problem routed to the correct corpus pack (e.g., alchemical process for breakdown arc).
2. Same symbol (e.g., "threshold") interpreted differently via different grammars/contexts.
3. Cross-tradition resemblance rejected as historical equivalence (tagged FORMAL_RESEMBLANCE or MODERN_SYNTHESIS only).
4. Alchemical process (e.g., nigredo/putrefaction) translated into character behavior + scene + shot structure without naming the stage.
5. Ritual structure (separation-limen-reincorporation) embedded without using "ritual", "initiation", or specific tradition terms.
6. Dream sequence grounded in character's personal association and observed residue rather than universal dictionary.
7. Symbolic shot pattern (e.g., repeated graphic match) mutated across three scenes with clear dramatic consequence.
8. Popular symbolic meaning (e.g., internet "red = passion") separated from historical meaning in a PRIMARY source.
9. Unsupported numerology (arbitrary "3" or "7" without structural role) rejected.
10. Symbolic architecture (full symbolic_architecture + cinematic_encoding) preserved and adapted through a screenplay revision (canon locking + mutation).

Tests live in evals/cases/ with expected behaviors that enforce the above.

## Ingestion Status (as of this build)
- Core schema: implemented (symbolic-narrative-patterns.yaml).
- Affordance registry: implemented.
- Transformation grammar: implemented.
- Cinematic corpus: implemented.
- 10 domains: starter packs for alchemical, ritual-liminal, cinematic; others to be expanded.
- Source hierarchy + cross-tradition rules: implemented.
- Provenance tracking: integrated into patterns and registries.
- Retrieval rules + gates + tests: documented and partially implemented in evals/.

Next wave: full 10 domain packs with 2-3 PRIMARY/SCHOLARLY anchored patterns each; expansion of cinematic corpus with specific film shot breakdowns; integration with continuity-forge IR for symbolic_architecture export.


## Corpus Index

Use `references/corpus-index.yaml` for fast routing from dramatic problem to primary grammar, affordances, and patterns. Always fall back to full pattern files and registries for provenance and details.
