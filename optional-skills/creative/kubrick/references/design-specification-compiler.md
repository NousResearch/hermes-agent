# Design Specification Compiler

Kubrick may compile heterogeneous creative and technical evidence into a governed `design.md` without treating generated interpretation as canonical fact.

## Purpose

The compiler extracts the invariant architecture of a project rather than summarizing its files. It converts repositories, prototypes, scripts, assets, transcripts, operator notes, and existing documentation into a schema-valid design specification that can be rendered as stable Markdown and compared against future revisions.

## Routing

Use this workflow when the request includes:

- create or update `design.md`;
- infer design intent from a repository or prototype;
- consolidate visual, interaction, audio, narrative, or motion rules;
- define design pillars, invariants, or continuity constraints;
- detect design drift across revisions;
- produce a canonical-candidate creative specification.

Do not use it to invent missing product requirements. Weak or contradictory evidence must remain an open question or return `NOT_COMPUTABLE`.

## Deterministic Operations

```text
RUNE.DESIGN.INGEST(sources)
Normalize source identity, revision, kind, and provenance.

RUNE.DESIGN.EXTRACT(evidence)
Extract observed intent, pressures, systems, languages, constraints, and unresolved boundaries.

RUNE.DESIGN.SYNTHESIZE(extracted_state)
Construct a schema-valid design specification without collapsing provenance.

RUNE.DESIGN.VALIDATE(specification)
Reject missing pillars, unverifiable invariants, unsupported locked facts, schema drift, and silent authority promotion.

RUNE.DESIGN.DIFF(previous, current)
Classify preserved intent, approved mutation, unresolved conflict, and prohibited drift.

RUNE.DESIGN.EXPORT(specification)
Render `design.md` and optional domain documents from the validated artifact.
```

## Required Separation

Every claim is routed as one of:

- `OBSERVED` — directly present in a source;
- `INFERRED` — supported by multiple observations or an explicit design relationship;
- `SPECULATIVE` — plausible but not sufficiently supported;
- `NOT_COMPUTABLE` — evidence is missing, contradictory, or outside the declared boundary.

A generated document defaults to `PROPOSED`. It must not claim `LOCKED` or canonical authority without an external authorization receipt.

## Compilation Sequence

```text
source inventory
→ observed facts
→ design intent and governing tension
→ pillars with verification tests
→ invariants with verification methods
→ intended experience and core loop
→ system boundaries and dependencies
→ visual / interaction / audio / narrative / motion languages
→ continuity contract
→ risks and open questions
→ schema validation
→ design.md rendering
→ optional revision delta
```

## Quality Gates

A design specification fails closed when:

1. a pillar has no observable verification test;
2. an invariant cannot be checked;
3. an inferred claim is presented as observed;
4. a locked fact lacks source support;
5. system responsibilities overlap without an explicit boundary;
6. design language is purely adjectival and has no operational rule;
7. a revision silently changes identity, function, chronology, ownership, interaction grammar, or continuity;
8. the document confuses implementation detail with governing design intent;
9. open uncertainty is converted into confident prose;
10. the generated Markdown and source artifact disagree.

## Output Package

Minimum:

```text
design-specification.yaml
design.md
```

Optional:

```text
technical-design.md
audio-design.md
art-direction.md
ui-design.md
level-design.md
narrative-design.md
production-plan.md
continuity-contract.md
revision-delta.yaml
```

The structured artifact remains the source of truth for deterministic comparison. Markdown is a rendered operator surface.

## CLI

```bash
python scripts/kubrick.py design-build \
  --input templates/design-specification.yaml \
  --output out/kubrick/design.md

python scripts/kubrick.py artifact-validate \
  --schema schemas/design-specification.schema.yaml \
  --artifact templates/design-specification.yaml
```

## Drift Semantics

A future revision should classify each material change as:

- `PRESERVED` — design intent and invariant remain intact;
- `MUTATED` — an allowed variable changed with lineage preserved;
- `CONFLICT` — two active rules cannot both remain true;
- `DRIFT` — a locked fact, invariant, pillar, or prohibited-drift rule changed without authorization;
- `NOT_COMPUTABLE` — evidence is insufficient to classify the change.
