# Hermes Graph Operators

Kubrick constructs the motif graph privately, validates it, then translates only observable constraints into the user-facing artifact.

## Required sequence

1. **RUNE.GRAPH** — derive nodes from observed forms and explicit state changes.
2. **RUNE.CONVERGE** — select one or two sites where multiple relations become observable together.
3. **RUNE.CHORONZON** — detect leakage, arbitrary density, named esoterica, and disconnected complexity.
4. **RUNE.COAGULA** — translate the valid graph into scene, storyboard, or single-frame constraints.
5. **RUNE.TIKKUN** — repair invalid graphs without changing approved intent.

## Operator contracts

### RUNE.GRAPH

Input: dramatic function, desired state change, observed forms, relations, project ledger.

Output: `motif-structure-graph.schema.yaml` compliant graph.

Rules:
- minimum two nodes and one edge,
- every node begins with observable form,
- every edge carries pressure and transformation,
- `SPECULATIVE` nodes cannot become audience-facing facts,
- lexicon and pattern links stay private.

### RUNE.CONVERGE

Prioritize one site; allow two only when the second creates distinct causal work.

A valid site requires:
- at least two nodes,
- at least one real edge,
- an observable effect,
- a mask priority,
- no symbolic explanation in the effect text.

### RUNE.CHORONZON

Fail or repair when:
- edge density is below 0.5,
- a node or edge reference is unknown,
- the same descriptive attribute leaks across disentangled layers,
- more than two convergence sites exist,
- named esoterica appears in `surface_output`,
- geometry, material, or light has no dramatic function.

### RUNE.COAGULA

Translation modes:
- `single-frame`: geometry, light, material, state differential, convergence, residue,
- `scene`: blocking, action, sound, recurrence, residue,
- `storyboard`: shared graph identity plus per-frame state mutations,
- `diagnostic`: includes private semantic/function layer for operator review.

The audience-facing prompt must never include lexicon labels by default.

### RUNE.TIKKUN

Repair order:
1. remove unknown references,
2. reduce convergence sites,
3. restore layer separation,
4. replace named esoterica with observable constraints,
5. add or remove edges until density is intentional,
6. revalidate before translation.

## Hermes invocation examples

```text
Load kubrick. Apply RUNE.GRAPH to this scene. Keep pattern and lexicon links private.
```

```text
Load kubrick. Apply RUNE.CONVERGE and lock the highest-pressure site. Then apply RUNE.COAGULA in single-frame mode.
```

```text
Load kubrick. Run RUNE.CHORONZON on this symbolic packet and repair it with RUNE.TIKKUN. Return only the repaired observable constraints and validation receipt.
```
