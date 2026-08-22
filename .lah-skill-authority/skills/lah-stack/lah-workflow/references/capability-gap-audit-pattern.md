# Capability Gap Audit Pattern (READ_ONLY)

Proven during BEHAVIORAL_DIAGNOSTIC_CONTRACT_AND_GAP_AUDIT_V1 (2026-08-17, lah-brain).

Use for missions of the class: "is X genuinely missing / does the stack already do X / should we build X". These are READ_ONLY_AUDIT missions where the deliverable is a defensible evidence-based decision, INCLUDING a decision not to implement.

## Golden rule: absence is a git-refs claim, not a working-tree claim

A discovery sub-agent that searches the checked-out working tree can report `MODULE_X — NOT FOUND` while the module exists on an unmerged feature branch. This produced a false "genuinely missing" verdict for the math layers:

- `git grep -l -E "<id>" $(git for-each-ref --format='%(refname:short)' refs/heads) 2>/dev/null` — search all local branches
- `git log --all --oneline --grep="<id>" -i | head` — find the commits that created the capability
- `git branch --merged main` vs `git branch --no-merged main` — what is actually deployed vs shadow/feature work
- `git ls-tree -r --name-only <branch> | grep -iE "math|..."` — enumerate files on a candidate branch

Canonical state includes feature branches. Unmerged shadow-stage layers, receipts, and Lot D work are real evidence. Always run the refs-wide search BEFORE concluding absence.

## Gap audit structure (10 phases)

1. **Canonical state audit** — repo, default branch, main HEAD, working branch+HEAD, remote, working tree state. Record "on main" vs "on branch" per component.
2. **Capability inventory** — for each candidate component: purpose, inputs, outputs, authority level, computes-vs-interprets, contextual-vs-global, revisable?, can influence routing/spend?, overlap with proposed layer.
3. **Duplication search** — search the FULL identifier list, plus semantically equivalent implementations. Never conclude from a single keyword search. Classify each overlap: ALREADY_CANONICAL / PARTIALLY_PRESENT / HEURISTIC_ONLY / OBSERVABILITY_ONLY / DISCOVERY_ONLY / DECISION_ONLY / GENUINELY_MISSING.
4. **Gap analysis** — distinguish A feature extraction / B mismatch diagnosis / C structural clustering / D behavioral interpretation / E causal validation / F economic decision. Determine whether D exists as its OWN canonical responsibility. If yes → STOP. If no → prove with specific evidence (write-only fields are strong proof: labels defined but consumed nowhere).
5. **Value test** — per downstream area: NONE/LOW/MEDIUM/HIGH. Only credit value that connects to an existing or planned contract.
6. **Minimal contract** — propose fields justified by the ACTUAL stack (each field maps to an existing primitive/contract). Drop fields that "sound useful" but map to nothing.
7. **Taxonomy feasibility** — prefer fewer defensible profiles over speculative taxonomy. Each profile must map to ≥2 existing signals.
8. **Integration point** — producer/consumer modules, canonical contracts, where NOT to integrate, smallest change surface.
9. **No-duplication guarantees** — explicit DO NOT BUILD matrix: each responsibility + canonical owner (analytics, memory, discovery, decision, causal, bankroll, autocut, routing, provider execution).
10. **Decision** — A CONFIRMED / B PARTIALLY_REDUNDANT / C NOT_JUSTIFIED. B means: extend an existing module, do NOT create a new layer.

## Read-only evidence conventions

- When the mission text forbids repository modification (stricter than the default READ_ONLY_AUDIT), write the report + continuity JSON to an EXTERNAL evidence dir: `/home/deploy/lah-audit-evidence/<MISSION>/`.
- Do NOT write to `docs/superpowers/plans/` or `docs/mcporter/` when the mission says "DO NOT modify repository files" — the mission text wins.
- Verify zero mutation: `git status --short` identical before/after, HEAD unchanged.
- Final gate: explicit booleans, one per decision axis, with the verdict identifier matching the mission's allowed list.

## Worked example (BEHAVIORAL_DIAGNOSTIC_CONTRACT_AND_GAP_AUDIT_V1)

- Verdict: B — PARTIALLY_REDUNDANT. mismatch-engine (heuristic pointwise interpreter) + source-dna tags partially cover behavioral interpretation; the provisional/contextual/revisable profile FORM is missing; Lot D0 labels (friction_class, intent_stage) are defined in event-schema but consumed NOWHERE on main (write-only debt).
- Key evidence commands that settled it: refs-wide grep for 18 profile identifiers (zero matches on main + 4 feat branches + docs); `git grep -n "friction_class\|intent_stage" main -- src/` returning empty (write-only labels).
- Report: /home/deploy/lah-audit-evidence/BEHAVIORAL_DIAGNOSTIC_CONTRACT_AND_GAP_AUDIT_V1/REPORT.md
- Companion artifact: /home/deploy/lah-brain-architecture-map.json (sub-agent output — always re-verify its absence claims per the golden rule).
