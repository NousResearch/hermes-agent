# Pataphysics

Alfred Jarry, *Gestes et opinions du docteur Faustroll, pataphysicien* (1898/1911). The science of imaginary solutions and particular cases.

Where physics is general laws applied to common cases, **pataphysics studies particular cases and imaginary solutions** — the *one-offs*, the *exceptions*, the *imagined entities whose virtuality* (potential being) can be described as lawfully as actual objects.

The OuLiPo was founded as a sub-committee of the Collège de 'Pataphysique. Marcel Duchamp, Eugène Ionesco, Boris Vian, Italo Calvino, Umberto Eco were members. Borges, Lem, Calvino, Roussel are pataphysical writers in this sense.

## When to use

- Push past plausibility; specify the impossible thing in detail
- Parodic / satirical work that needs rigorous form
- Producing fictional artifacts (encyclopedias of non-existent civilizations, manuals for non-existent devices, reviews of non-existent books)
- Stuck and the realistic solutions feel exhausted — specify the impossible solution
- Highlighting that a "natural" framing is actually a choice

## Don't use when

- You need an actually-implementable proposal on the first pass
- Audience requires sincerity (drifts toward irony)
- Avoiding harder analysis (slop variant: pataphysical-flavored dodge)
- You don't actually have anything to say (form requires content)

## Operating moves

### Specify an imaginary object
1. Pick the object. A device, organism, institution, place, work, person — something that cannot exist.
2. Specify its **lineaments** in concrete material detail. What is it made of? How does it operate? What are its parts?
3. Identify its laws — internal consistency rules. What can it do? What can't it?
4. Describe consequences if it existed.
5. **Stop short of asking whether it could exist.** That question is not pataphysical.

### Exception-finding
1. State the general rule in your domain.
2. Find the actually-existing case that doesn't fit.
3. Describe it on its own terms — not as deviation, but as what it is.
4. Resist generalizing back into a modified rule.
5. The particular case is the result.

### Pataphysical fiction
1. Adopt the form of a serious genre (encyclopedia, manual, technical paper, museum catalog, book review).
2. Apply the form rigorously to a non-existent subject.
3. Don't break frame. Don't wink.

## Worked example

**Problem**: file synchronization software. Realistic solutions all involve some compromise on conflict resolution.

**Pataphysical specification**: a file system in which two simultaneous edits to the same file produce a *third* file containing both edits as "ghosts" — versions visible to and editable by readers but not committed until a quorum of readers reads them and chooses one. The file exists in superposition until observation.

**Lineaments**: ghost-files have an "observation count"; below threshold they are interactive but not committed; above, they collapse to chosen version.

**Consequences**: editing a popular file is fast (quorum collapses quickly); editing an obscure file is slow (no quorum). The file system has *audience-dependent commit semantics*.

The specification is impossible. But *audience-dependent commit semantics*, surfaced by the pataphysical move, is in fact a useful concept with plausible implementations.

## Anti-slop notes

- Whimsical incoherence is not pataphysics. "What if cows could fly" without the cow's wing-loading and lift coefficient = sloppy fantasy.
- Don't generate fake-Borges or fake-Calvino. Their work is grounded in deep specifics. Generated "in the style of" is decorative.
- The dry, committed register matters. Comedic SF is not pataphysics.
- Don't walk back to "of course this is just a thought experiment" at the end. That undoes the operation.

Sources: Jarry, *Gestes et opinions du docteur Faustroll, pataphysicien* (Fasquelle, 1911); Borges, *Ficciones* (1944); Lem, *A Perfect Vacuum* (1971).
