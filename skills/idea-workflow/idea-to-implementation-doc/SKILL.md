---
name: idea-to-implementation-doc
description: "Use when reviewing one specific idea/design doc, researching similar products, and producing a separate technical implementation plan or roadmap."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [idea-workflow, implementation, roadmap, research, planning]
    related_skills: [idea-superpowers-suite, idea-to-design-doc, writing-plans]
---
# Idea to Implementation Doc Workflow

Use this skill when the user wants to take *one specific idea note* and turn it into a technical implementation document.

This is **not** the top-level ideas index workflow. It works on a *single idea file* and produces a separate implementation plan.

## Goal

Given a specific idea/design doc, create a detailed implementation document that:
- reviews the idea carefully
- researches similar products or existing apps
- identifies gaps or overlap with existing solutions
- proposes a concrete build approach
- outlines the system design, major components, and development phases
- converts the plan into an executable set of tasks and/or prompts for Hermes, Claude Code, Codex, OpenClaw, or another coding agent
- includes testing, verification, and acceptance criteria so “done” means the program works and is ready to use
- produces a single markdown file that can be handed to an agent as the build source of truth

For physical-product, hardware-accessory, 3D-printable, or digital STL/3MF ideas, adapt the implementation plan away from app/database/backend defaults and use `references/physical-product-cad-workflow.md` for CAD, material, tolerance, export, prototype, and validation planning.

## Important constraints

- Focus on *implementation strategy*, not just product brainstorming.
- Research existing apps only to the extent needed to avoid reinventing something obvious.
- Do not over-engineer the stack. Choose the simplest viable approach.
- Keep the output practical and actionable.
- If the user asks for the doc to stay non-technical, stop at product-level architecture and planning.
- If the user wants code-level detail, include it in a separate section rather than mixing it into the product description.

## Inputs

Primary input:
- one specific idea note, e.g. `ideas/my-app.md`

Optional input:
- associated design doc, if it exists separately
- user constraints about stack, hosting, or platform
- deployment/data-location preference: local-only, self-hosted, Cloudflare, AWS, another cloud, or undecided
- platform targets: browser/web-only, Windows desktop, Mac desktop, cross-platform desktop, mobile web, iOS, Android, or explicitly no mobile
- accepted technical defaults or overrides for database, backend/runtime, frontend, auth, file/object storage, queues/jobs, realtime/sync, search, observability/logging, tests, and deployment/CI

## Workflow

### 1) Read the source note

Review the target idea/design doc and extract:
- the core problem
- intended user
- must-have behavior
- non-goals
- any unresolved questions

### 2) Research similar products

Look for apps/products with similar goals or workflows.

Identify:
- comparable products
- what they do similarly
- where the user’s idea differs
- what this idea should avoid copying or duplicating

Treat this as lightweight product reconnaissance, not exhaustive market analysis.

### 3) Convert to an implementation strategy

Translate the idea into an implementation plan covering:
- product scope
- main user flows
- data model / database needs
- recommended database/storage default and accepted/changed decision
- hosting/deployment target and where data should live
- front end structure
- backend/service needs
- platform target decisions: web-only, desktop apps, mobile apps, responsive web, and which are MVP vs later
- recommended technical defaults for frontend, backend/runtime, auth, storage, queues/jobs, realtime/sync, search, observability/logging, tests, and deployment/CI
- integration points
- file/module organization
- development phases
- executable task breakdown
- agent-ready prompts or handoff instructions
- testing and verification strategy
- acceptance criteria and “done means” checklist
- risks and open questions

### 4) Produce implementation spec and final handoff docs

Save the implementation work near the source note using the staged idea-workflow layout when appropriate:

- `README.md` — status/index page for the full idea package.
- `02-implementation-spec.md` — the detailed implementation strategy.
- `03-agent-build-handoff.md` — the final single-file handoff for a coding agent or Superpowers.
- `04-spec-review.md` — readiness review before execution.

For lightweight ideas, a single `<idea-name>.implementation.md` file is acceptable, but full app builds should use the staged layout.

Suggested naming:
- `ideas/<idea-slug>/README.md`
- `ideas/<idea-slug>/02-implementation-spec.md`
- `ideas/<idea-slug>/03-agent-build-handoff.md`
- `ideas/<idea-slug>/04-spec-review.md`

## Recommended output structure

```markdown
# <Idea Title> Implementation Plan

## Summary

## Source Idea Review

## Similar Products / Market Check

## Product Scope

## System Overview

## Data Model

## Database / Storage Recommendation

## Hosting / Data Location / Deployment

## Technical Stack Recommendation

## Frontend Plan

## Platform Targets

## Backend / Services Plan

## Key Workflows

## Milestones / Phases

## Build Tasks

## Agent Build Prompt

## Testing and Verification Plan

## Acceptance Criteria / Done Means

## Risks and Tradeoffs

## Open Questions

## Next Build Prompt
```

## Final agent handoff format

After the implementation spec is drafted, create `03-agent-build-handoff.md` using this required structure:

```markdown
# <Project Name> Agent Build Handoff

## Mission

## Product Vision

## User Experience Goals

## Non-Negotiable Requirements

## Out of Scope

## Technical Architecture

## Data Model

## Database / Storage Recommendation

State the recommended database/storage approach, why it fits the product, and whether the user accepted it or overrode it. Mention relevant alternatives considered when helpful.

## Hosting / Data Location / Deployment

Specify whether data stays local, is self-hosted, uses Cloudflare, AWS, another cloud, or remains undecided. If undecided, list the assumptions used for the MVP and what would change later.

## Platform Targets

Specify whether the MVP is browser/web-only, Windows desktop, Mac desktop, cross-platform desktop, mobile web, iOS, Android, or a combination. Explicitly separate MVP platforms from future platforms.

## Technical Stack Recommendation

List recommended defaults and user overrides for frontend, backend/runtime, auth, database, file/object storage, queues/jobs, realtime/sync, search, observability/logging, testing, and deployment/CI. The handoff should not require the build agent to guess these choices unless the spec explicitly marks them as flexible.

## Integrations

## Implementation Phases

## Build Tasks

## Testing Requirements

## Verification Commands / Checks

## Acceptance Criteria

## Done Means

## Known Risks

## Open Questions

## Prompt for Build Agent
Use a prompt that asks the build agent to inspect the project, preserve scope, create or follow an implementation plan, run tests, and verify evidence before claiming done. Do not duplicate Superpowers' detailed planning protocol; hand off to it.

## Superpowers Build Handoff
Use the `superpowers-gpt` workflow on this document.

Start with:
1. `superpowers-using-superpowers` to route the work.
2. `superpowers-writing-plans` to inspect the target repo and turn this handoff into exact implementation tasks, files, commands, tests, and expected results.
3. `superpowers-executing-plans` or `superpowers-subagent-driven-development` to implement the approved plan without scope drift.
4. `superpowers-requesting-code-review` to review the implementation.
5. `superpowers-verification-before-completion` to prove the work satisfies the acceptance criteria before claiming completion.

The goal is to convert this spec into an executable implementation plan, build it with tests, review the result, and verify before claiming completion.
```

## Spec review gate

Before saying the handoff is ready, create `04-spec-review.md` with this verdict structure:

```markdown
# <Project Name> Spec Review

Verdict: PASS | PASS WITH CHANGES | FAIL

## Summary

## Readiness Checklist
- [ ] Product goal is clear
- [ ] Requirements are testable
- [ ] Product decisions are resolved or explicitly listed
- [ ] Technical decisions are resolved or explicitly listed
- [ ] Database/storage recommendation is present or explicitly unnecessary
- [ ] Recommended technical defaults are documented, with user overrides if any
- [ ] Acceptance criteria are concrete
- [ ] Done Means is specific
- [ ] A fresh agent could build from the handoff without obvious missing context
- [ ] Testing requirements are included
- [ ] Verification commands/checks are included
- [ ] Non-goals prevent likely scope creep

## Required Changes Before Build

## Optional Improvements

## Superpowers Handoff Recommendation
```

Only use `PASS` when the build handoff is ready to feed into Superpowers without major clarification.


## Lite vs Full mode and progression rules

If this skill is invoked for a lightweight idea, a single `<idea-name>.implementation.md` may be enough. Do not force the full handoff package unless the user asks for build readiness.

For Full mode, follow these rules:
- Do not move to implementation if unresolved product questions would change architecture or MVP scope.
- Do not create `03-agent-build-handoff.md` until testing requirements, verification checks, acceptance criteria, and Done Means are concrete.
- Do not mark `04-spec-review.md` as `PASS` when major product or technical choices are still unstated.

The user can override completeness gates with the exact phrase **GREENLIGHT NEXT STAGE**. When they do, proceed to the next artifact, record unresolved items as assumptions/open questions, and continue.

## Research guidance

When researching similar products, answer:
- What existing apps already solve this problem?
- Which parts are commodity features?
- Which parts are unique or differentiated?
- Is the idea actually a variation of a well-known product category?
- What would make users choose this over existing tools?

If the idea appears very close to an existing app, note that clearly and suggest repositioning or narrowing scope.

## Review guidance

When reviewing the idea doc, explicitly check for:
- ambiguity
- scope creep
- missing user flows
- unclear data ownership
- missing hosting/data-location choice
- missing platform target decisions for web, desktop, and mobile
- missing database/storage recommendation
- missing technical default recommendations or user overrides
- vague success criteria
- hidden assumptions
- duplicate features that already exist elsewhere

## Implementation planning guidance

For the technical plan:
- Be concrete about likely components.
- Prefer a simple architecture first.
- Explain why a database is or is not needed.
- Recommend a default technical stack first, then allow the user to accept or change it. Do not make the user pick every technical component from scratch.
- Cover database/storage, backend/runtime, frontend/UI, auth, file/object storage, queues/jobs, realtime/sync, search, observability/logging, tests, and deployment/CI when relevant.
- Record accepted defaults as decisions and user changes as explicit overrides.
- If the app is small, mention a minimal viable architecture rather than a full enterprise stack.
- If the app needs auth, persistence, search, sync, or real-time features, call that out.
- For cost-aware runs, cheaper workers may handle bounded research, first-pass implementation specs, task breakdowns, and critique/spec-review passes, but the orchestrator must resolve tradeoffs and approve the final handoff. Follow `idea-superpowers-suite` Cost-Aware Worker Delegation rules when available.

## Stop conditions

If the user says to stop or only wants a high-level review, produce whatever you have and mark remaining uncertainty clearly.

## Behavior when invoked

This skill is intended to be run as a task or by explicit request after the idea doc already exists.

Typical response style:
- concise summary of what the idea is
- similar products found
- implementation strategy
- file path to the new implementation doc

## Quality bar

A good implementation doc should let a developer or AI coding agent immediately understand:
- what to build
- why this approach makes sense
- what the major system pieces are
- how to start building without guessing
- what existing products to compare against
- which tasks to complete in what order
- what tests to run
- how to verify the app actually works
- what must be true before the agent can honestly say “it’s done”
