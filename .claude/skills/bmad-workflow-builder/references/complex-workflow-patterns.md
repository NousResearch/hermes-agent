# Complex Workflow Patterns

Routing mechanics for workflows whose SKILL.md grew past its token budget and had to carve work out to `references/`. The carve conventions themselves — descriptive names, standalone files, what stays in SKILL.md — live in `references/skill-quality-principles.md`, and the portable producing-skill patterns live in `references/producing-workflow-patterns.md`. This file is only what multi-stage routing adds.

## Multi-Stage Routing as an Earn-It Surface

Multi-stage routing is structure, and structure has to earn its place against a flatter alternative. Before splitting a workflow into routed stages, ask whether a single goal-driven SKILL.md with named sections would have produced the same result. Usually it would, so reach for explicit stages only when the workflow is large enough that SKILL.md cannot hold it within budget, or when stages have genuinely different resume and memory behavior.

When stages earn their place, name them descriptively and route by intent. The stage table near the bottom of SKILL.md is a reading aid that maps an intent to a location:

```markdown
## Stages

| Stage | Intent it serves | Location |
|-------|------------------|----------|
| Ignition | Capture the raw concept, enforce customer-first thinking | SKILL.md (above) |
| Press Release | Iterative drafting with hard coaching | `references/press-release.md` |
| Customer FAQ | Surface devil's-advocate customer questions | `references/customer-faq.md` |
```

The intent routing table is what makes the split worth its cost, because the model reads the user's intent and jumps straight to the stage that serves it rather than walking a fixed sequence. Stage order is a routing decision SKILL.md makes per run rather than something baked into the file names.

## Carved Files and the Memlog

Carved files reach the memlog by its resolved path rather than assuming in-context state, because compaction can drop SKILL.md before the carved file runs.

## Module Metadata Reference

BMad module workflows carry extended frontmatter metadata; see `references/standard-fields.md` for the field conventions. The workflow-builder captures module-capability metadata as handoff fields only and never authors module.yaml.
