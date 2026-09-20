# Pattern Languages

Christopher Alexander et al., *A Pattern Language* (1977). 253 patterns for designing buildings, towns, rooms — structured as a generative grammar with explicit cross-references. Spawned the Gang of Four software design patterns (1994) and many domain adaptations.

## Pattern format

A pattern has three parts:
1. **Context** — the situation in which it applies
2. **Problem** — a recurring tension in that context
3. **Solution** — a *generative* principle (not a specific design — capable of many instantiations)

A pattern *language* is a network of patterns at different scales, with explicit links: which patterns *contain* this one, which patterns *complete* it.

## When to use

- Designing physical environments (buildings, rooms, gardens, neighborhoods)
- Designing interactional environments (UX, software architecture)
- Building shared design vocabulary with a team
- Documenting design intuitions for transmission
- Civic / community design

## Don't use when

- You want to break with tradition (patterns are conservative — they encode what has worked)
- Domain has no established practice yet (no patterns to extract)
- Pure conceptual / artistic work
- You'd be implementing patterns literally (collapses generative → rule)

## Selected patterns from Alexander's 253

For texture. Real use means buying or borrowing the book.

- **8. Mosaic of Subcultures** — a region needs distinct subcultures with their own ecology, separated by zones of disuse, not homogenized.
- **53. Main Gateways** — mark every entrance with a substantial visible threshold.
- **60. Accessible Green** — green outdoor space within 3 minutes' walk.
- **105. South-Facing Outdoors** — most-used outdoor space to the south of the building.
- **111. Half-Hidden Garden** — garden right at street is too public; behind house is unused. Place it half-hidden.
- **159. Light on Two Sides of Every Room** — windows on at least two sides. Single-sided rooms are uncomfortable, rarely used.
- **179. Alcoves** — rooms with no place to retreat are unsettling. Build niches, bays, window seats.
- **188. Bed Alcove** — bed in the open is exposed. Build at least a partial enclosure.
- **191. Shape of Indoor Space** — simple, mostly orthogonal; deviate only for clear local reason.
- **230. Radiant Heat** — radiant heat (fireplace, radiator) is qualitatively different from forced air.

The patterns are arguably true and arguably false; what matters is the *form*.

## Procedure

### Using an existing language
1. Identify the relevant scale (region / neighborhood / building / room / detail).
2. Read patterns at and above your scale; note which apply.
3. Compose: apply higher-scale patterns first; let them constrain lower-scale ones.
4. Adapt to your specifics. Patterns are generative, not literal.

### Developing your own language (more useful for software, org, pedagogy)
1. Identify recurring problems in your domain. Look across many cases.
2. Name each (short, memorable, describes the *solution* shape — "Light on Two Sides", not "Insufficient Daylight").
3. State each in: context — problem — solution — therefore: [generative principle] — see also: [related patterns].
4. Map containment relations between patterns.
5. Test by applying to a fresh problem; revise.

## Worked example (software, in Alexander's form)

**Iterator pattern** (Gang of Four, 1994)

*Context*: a collection of objects must be traversable by client code.
*Problem*: client shouldn't need to know the internal structure (array vs tree vs linked list); collection shouldn't have traversal logic scattered across clients.
*Solution*: provide an Iterator object with `next()`, `hasNext()`, `current()` that encapsulates traversal state. Collection produces an Iterator on request.
*Therefore*: separate "what is being traversed" from "how it is traversed."
*See also*: Composite (tree traversal), Visitor (operations during traversal), Factory Method (producing the right Iterator).

## Anti-slop notes

- Bullet-list "design tips" are not patterns. A pattern has context, problem, generative solution, and place in a network.
- Don't generate patterns to seem comprehensive. Real patterns come from many cases.
- Don't apply Alexander's residential patterns to non-residential domains literally.
- Patterns are conservative *and* generative. They don't anti-novelty; they shape novelty.

Source: Alexander et al., *A Pattern Language* (Oxford UP, 1977); *The Timeless Way of Building* (Oxford UP, 1979). For software: Gamma et al., *Design Patterns* (Addison-Wesley, 1994).
