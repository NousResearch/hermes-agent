# Affinity Diagrams

Jiro Kawakita, *Hassōhō* (1967). The KJ method (Kawakita's initials, Japanese order). Bottom-up procedure for finding structure in qualitative items without imposing it beforehand.

## When to use

- After volume generation (100+ ideas from Crazy 8s or brainwriting need clusters)
- Qualitative research synthesis (interview transcripts, ethnographic notes, observations)
- Requirements gathering (pile of user requests / bug reports / suggestions)
- Sense-making after a workshop (whiteboard full of stickies)
- Bottom-up taxonomy when no good existing one fits
- Diagnosing what's missing — gaps between clusters often reveal what the data set lacks

## Don't use when

- Few items (under ~15 — overkill, hold them in mind instead)
- The right structure is already known (use deductive coding)
- Time pressure — done well takes hours
- Solo without enough cognitive distance from items (you'll produce the categories you'd have produced anyway)
- Highly quantitative data (use stats)

## Procedure

1. **Atomize items.** One observation per card. Items must be self-contained, specific, comparable in granularity.
2. **Make them physically separable.** Sticky notes; index cards; or a shared canvas (Miro/Mural/FigJam). Free movement matters; a list in a doc doesn't work.
3. **Spread out.** Distribute across a flat surface. No structure yet.
4. **Cluster silently.** Each participant moves items into proximity with similar ones. **Silently** — talking shapes group thinking, defeats bottom-up. If two participants disagree on placement, *duplicate the item* and let it appear in both.
5. **Continue until movement slows.**
6. **Name each cluster.** Specific names ("requests for offline functionality"), not generic ("technical issues"). Resist generic names.
7. **Look at orphans and gaps.**
   - Orphans: items not fitting any cluster — often the most surprising data.
   - Gaps: spaces between clusters — suggest categories the data lacks (questions like "why didn't anyone mention X?").
   - Cluster sizes: very large = items not differentiated enough; very small = specialized concerns worth noting.
8. **Look for relationships between clusters.** Some depend on others. Some conflict.
9. **Narrative test (Kawakita).** Write a 1–2 paragraph narrative using the cluster names to tell a coherent story about the domain. If you can't, the clusters are misapprehension.

## Worked example

50-person team brainwrites about "what would make the codebase more maintainable" — 108 raw ideas.

After 45 minutes silent clustering:

- **Dependency hygiene** (~22 items)
- **Test coverage and CI speed** (~18)
- **Documentation drift** (~14)
- **Onboarding friction** (~12)
- **Implicit knowledge** ("only Sara knows how X works") (~10)
- **Tooling fragmentation** (~9)
- **Technical debt visibility** (~8)
- **Orphans** (~15 — scattered specific concerns)

**Gap**: noticeably absent — almost no items about *production reliability*, *security review*, or *cross-team API contracts*. The team's perception of "maintainability" is internal-developer-facing; user-facing reliability is not surfaced.

**Narrative**: "Maintainability concerns cluster around (1) dependencies, (2) tests, (3) docs-code drift, with secondary concerns around onboarding and implicit knowledge. The team experiences maintainability as a developer-experience problem rather than a reliability problem."

The diagram has produced a *map of perceived maintainability problems*. Decisions about which to address require additional inputs (impact, cost, owner). But the map shows what the team thinks the problem is — and the gap is itself useful.

## Anti-slop notes

- **Fast affinity grouping that produces familiar categories = deductive coding pretending to be inductive.** If the categories are the same as you'd have written before looking at the items, you've performed deductive coding.
- Don't generate fake observations to populate clusters.
- Avoid generic cluster names ("things to improve", "various concerns").
- Don't compress too aggressively. Real data has variable cluster sizes (5–25 typical); uniform sizes suggest forced grouping.
- Affinity diagrams are sense-making, not proof. Clusters represent *the researcher's perception* of items, not objective truth.
- For LLM-driven affinity grouping: models impose familiar taxonomies. After clustering, ask "what's the most surprising cluster?" If nothing surprising, redo or supplement with human eyes.

Source: Kawakita, *Hassōhō* (Chuko Shinsho, 1967, in Japanese). Mizuno (ed.), *Management for Quality Improvement: The Seven New QC Tools* (Productivity Press, 1988).
