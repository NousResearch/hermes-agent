# Continuity Forge Integration

This skill is the **creative / structural** layer. Continuity Forge is the **deterministic kernel** that owns canonical state.

## When to Handoff

- After premise, characters, structure, or scene contracts are approved by the user.
- Before or instead of writing full prose pages when the goal is production use.
- On any material change to canon (new approved scenes, character traits that affect continuity, structure revisions).

## Recommended Handoff Flow

1. Produce clean artifacts from this skill (project brief, scene contracts, character bibles, approved canon list, symbolic_architecture and cinematic_encoding when applicable).
2. Ingest via Forge (prefer MCP or CLI with proper mutation envelope):
   - `acquire_write_lease`
   - `ingest_script` (or `compile_script` + ingest)
   - `build_ledger` / `build_shot_contracts`
3. Capture receipt (hashes, document_key, shot IDs).
4. Reference Forge state in future work (use `get_project_status`, `inspect_scene`, etc. for grounding).

## CLI Examples

```bash
# Basic compile from fountain or structured text
continuity-forge compile path/to/outline.fountain --out out/

# Or from a scene contract / brief you produced here
continuity-forge ingest --document-key myfilm --source structured-outline.md
```

## MCP Tools (via companion `hermes-continuity-forge` skill)

Typical tools you will call after creative work:
- `compile_script`
- `ingest_script` (with mutation contract)
- `build_ledger`
- `build_shot_contracts`
- `get_project_status`
- `audit_drift`

Always include:
- `document_key`
- `actor_id` (e.g. hermes-kubrick-<session>)
- Full mutation envelope when writing

## Mutation Contract Requirements (when ingesting changes)

From the operator skill:
- actor_id
- authorization_scope
- idempotency_key
- rationale
- expected_state_hash (when updating existing)

This skill produces the *rationale* and *content*. The operator skill (or direct MCP call) supplies the envelope.

## Boundaries

- This skill may generate **PROPOSED** narrative material and scene contracts (including symbolic proposals).
- Forge owns the canonical ledger, IR, and shot contracts.
- Never claim "this is now in the film" until you have a Forge receipt with committed status.
- For drift or contradictions discovered here: run local CONTINUITY pass, then cross-validate with Forge `audit_drift`.

## Feedback Into Kubrick (Wave 2)

Extract multi-signal observations from Forge outcomes without promoting authority:

```bash
python scripts/kubrick.py forge-signals \
  --project-id myfilm \
  --input path/to/ledger-diff-or-revision.yaml \
  --output out/forge-signal-bundle.yaml

python scripts/kubrick.py ledger apply-forge \
  --ledger project/symbolic-ledger.yaml \
  --forge-bundle out/forge-signal-bundle.yaml

python scripts/kubrick.py evolution-propose \
  --pattern-id some_pattern \
  --forge-bundle out/forge-signal-bundle.yaml \
  --output out/evolution-proposal.yaml
```

Rules:
- Forge signal bundles are `OBSERVATION` only.
- Multi-signal evolution receipts never set `automatic_application_allowed: true`.
- Structural mutations and large confidence deltas require human review.
- Project ledgers remain local until Forge (or an authorized reviewer) promotes them.

## Recommended Pairing

Load both skills:
- `kubrick` for creative development, diagnosis, anti-slop, voice, structure, symbolic dramaturgy and cinematic encoding.
- `hermes-continuity-forge` for leases, ingestion, proof, shot repair, approvals.

See main repo `docs/hermes/README.md` and the companion skill for full operator rules.

## Export Mapping: symbolic_architecture → Continuity Forge IR / Shot Contracts

When symbolic work is approved, the following mapping should be used when handoff artifacts are prepared for Forge (via MCP, CLI ingest, or structured brief).

### Core Mapping Table

symbolic_architecture:
  governing_tension:          → thematic_tension in project brief or scene contract
  symbolic_intent:            → scene_contract.symbolic_intent (dramatic_function, intended_payoff, prohibited_explanation)
  motif_registry:             → continuity_ledger.motifs[] and/or shot_contracts.recurrence[]
    - motif_id + observed_form → ledger entry with state, recurrence_plan, provenance
    - transformations          → mutation history in ledger
    - candidate_functions      → narrative functions attached to motif
  archetypal_functions:       → character or role fields in scene contracts (as observable behaviors only)
  correspondence_map:         → shot_contracts.geometry or production_notes (only if provenance is PRIMARY/SCHOLARLY)
  tradition_boundaries:       → metadata on any symbolic element (do not allow unsupported equivalence)
  recurrence_plan / inversion_plan / convergence_plan / residue_plan:
                               → shot_contracts.recurrence, mutation, convergence, residue fields
  cinematic_encoding:
    composition_patterns:     → shot_contracts.composition (e.g., symmetry, negative_space, occlusion)
    geometric_patterns:       → shot_contracts.geometry (circle, grid, spiral, broken symmetry, etc.)
    blocking_patterns:        → shot_contracts.blocking (power_center, threshold, prohibited_zone, formation)
    camera_patterns:          → shot_contracts.camera (height, distance, movement, lens_behavior)
    shot_recurrence:          → explicit recurrence in shot contracts with mutation required
    edit_cadence:             → editing notes in shot contracts (rhythm, ellipsis, match cuts)
    sonic_motifs:             → sound design parameters (acousmatic, bridges, silence, residue)
    production_design_states: → production design states tied to motifs

### Handoff Rules
- Only include symbolic_architecture when it has passed quality gates and has provenance.
- For every motif or pattern exported, include at minimum: observed_form, mutation history, provenance reference, and intended dramatic function.
- Never export "meaning" — export the structural instructions (what must recur, how it must mutate, which cinematic parameters carry the charge).
- Forge will treat these as constraints on the Production IR and shot contracts, not as interpretive notes.

Example (simplified):
```yaml
handoff:
  symbolic_architecture:
    motif_registry:
      - motif_id: "red_carpet"
        observed_form: "geometric red pattern underfoot"
        recurrence_plan: "must mutate ownership, scale, or context each appearance"
        provenance: "PRIMARY via film analysis (Kubrick Shining)"
    cinematic_encoding:
      geometric_patterns: ["broken symmetry", "repeating grid with contamination"]
      blocking_patterns: ["power_center moves along the pattern"]
  forge_target:
    ledger.motifs: ["red_carpet"]
    shot_contracts.geometry: ["grid", "contamination"]
    shot_contracts.recurrence: {mutation_required: true}
```
