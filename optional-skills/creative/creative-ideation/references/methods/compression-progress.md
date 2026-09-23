# Compression Progress

Jürgen Schmidhuber, *Formal Theory of Creativity* (1990–2010). Beauty = compressibility given prior knowledge. Interestingness = the *change* in compressibility as you learn. A worthwhile project is one that, on completion, would compress your model of the world.

## Core formula

```
I(D, O(t)) = B(D, O(t)) − B(D, O(t−1))
```

Interestingness = first derivative of beauty over time. Pure noise (no learnable pattern) and fully-known pattern (already compressed) are both boring. Beauty lives between.

## When to use

- Picking a research question
- Selecting between candidate projects ("which would teach me the most?")
- Diagnosing aesthetic dissatisfaction ("this is fine but not interesting")
- Choosing what to read

## Don't use when

- Fast generation (this is reflective, not generative)
- Group decisions where audiences differ (single-observer model)

## Procedure

### For picking a research question
1. List 5–10 things you currently *cannot predict well* in your domain. Be specific: not "the future of AI", but "why X 7B model trained with technique A performs worse than Y 1.3B model with technique B on benchmark Z".
2. For each: would understanding it compress only this fact, or re-organize a broader domain? Prefer the latter.
3. For each: is the answer learnable from where you are? (Not noise; not too far above your prior.)
4. Pick the highest learnable compression-progress potential.

### For evaluating ideas
For each candidate, ask:
- What would I understand differently if this were complete?
- Would that understanding compress this domain or only this idea?
- Is it currently learnable from where I am?

Highest answers across all three = pursue.

### For aesthetic critique
Where is the work entirely predictable? (too known) Entirely unpredictable? (too random) Where does it sit in the learnable-but-not-yet-learned zone? Strong work has more of the third.

## Worked example

User has three options:
- A. Build a habit tracker.
- B. Build a tool that explains why a `git rebase --interactive` produced its conflicts, by reconstructing the commit graph mid-rebase.
- C. Read Lacan.

Analysis:
- A: no compression progress; user already has model of habit trackers. Reject.
- B: high. User doesn't currently have strong model of how rebase constructs intermediate states; building this requires learning that, and the resulting model re-organizes how the user thinks about all VCS internals.
- C: real compression-progress potential, but prior is missing. Long path to get there. Worthwhile if on the prerequisite track; otherwise read Žižek/Bruce Fink first as scaffolding.

Recommend B.

## Anti-slop notes

- "Compression progress" as slogan ≠ doing the analysis. State the actual model gaps you'd close.
- Don't claim every idea has high compression-progress. Most don't. The framework is useful because it discriminates.
- Don't impose this lens on artistic work without acknowledging its limits.

Source: people.idsia.ch/~juergen/creativity.html
