---
name: idea-to-design
description: 'Collaborative design dialogue before any implementation. Use before creating features, building components, adding functionality, or modifying behavior. Hard gate: no code until design is approved.'
version: 1.0.0
author: Hermes Agent (adapted from obra/superpowers brainstorming)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [design, brainstorming, planning, specification, collaboration]
    related_skills: [writing-plans, spec-driven-development, blueprint]
---

# Idea to Design

Turn ideas into fully formed designs and specs through natural collaborative dialogue.

**HARD GATE: Do NOT write any code, scaffold any project, or take any implementation action until you have presented a design and the user has approved it.**

## Anti-Pattern: "This Is Too Simple To Need A Design"

Every project goes through this process. A todo list, a single-function utility, a config change — all of them. "Simple" projects are where unexamined assumptions cause the most wasted work.

## Checklist

Complete these in order:

1. **Explore project context** — check files, docs, recent commits with `read_file` and `search_files`
2. **Ask clarifying questions** — one at a time, understand purpose/constraints/success criteria
3. **Propose 2-3 approaches** — with trade-offs and your recommendation
4. **Present design** — in sections scaled to their complexity, get user approval after each section
5. **Write design doc** — save to `docs/designs/YYYY-MM-DD-<topic>-design.md`
6. **Spec self-review** — quick inline check for placeholders, contradictions, ambiguity, scope
7. **User reviews written spec** — ask user to review the spec file before proceeding
8. **Transition to implementation** — invoke `writing-plans` skill to create implementation plan

## Process Flow

```
Explore project context → Ask clarifying questions → Propose 2-3 approaches
→ Present design sections → User approves design → Write design doc
→ Spec self-review → User reviews spec → Invoke writing-plans
```

**The terminal state is invoking writing-plans.** Do NOT invoke any implementation skill. The ONLY skill you invoke after this is `writing-plans`.

## The Process

**Understanding the idea:**

- Check out the current project state first (files, docs, recent commits)
- Before asking detailed questions, assess scope: if the request describes multiple independent subsystems, flag this immediately
- If the project is too large for a single spec, help the user decompose into sub-projects
- For appropriately-scoped projects, ask questions one at a time to refine the idea
- Prefer multiple choice questions when possible, but open-ended is fine too
- Only one question per message — if a topic needs more exploration, break it into multiple questions

**Presenting the design:**

- Present design sections one at a time, scaled to complexity
- Each section ends with an explicit approval check
- For simple projects, design can be a few sentences

**Spec format:**

```markdown
# Design: [Topic]

## Objective

[What we're building and why]

## Approach

[Chosen approach with rationale]

## Alternatives Considered

[2-3 alternatives with why they were rejected]

## Architecture

[Key components and their relationships]

## Risks

[Known risks and mitigation strategies]

## Open Questions

[Anything unresolved that needs human input]
```

## Hermes Integration

- Use `read_file` and `search_files` to explore project context
- Use `write_file` to save design docs to `docs/designs/`
- Use `terminal` for git context
- Handoff to `writing-plans` for implementation planning
- Combine with `spec-driven-development` for full SDD workflow
