---
name: grill-with-docs
description: "Use when the user wants to stress-test a plan against an existing codebase's domain language, docs, code, CONTEXT.md glossary, and ADRs before building."
version: 1.0.0
author: Hermes Agent, adapted from Matt Pocock
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [engineering, codebase, documentation, domain-model, adr, ubiquitous-language]
    homepage: https://github.com/mattpocock/skills/blob/main/skills/engineering/grill-with-docs/SKILL.md
    related_skills: [grill-me, writing-plans, systematic-debugging, codebase-inspection]
---

# Grill With Docs

## Overview

Use this skill to run a grilling session against an existing project. Interview the user relentlessly about their plan, but ground the interview in the codebase, project documentation, domain model, `CONTEXT.md` glossary, and architectural decisions.

This starts with the same one-question-at-a-time pressure testing as `grill-me`, then adds active documentation work: challenge terminology against existing language, sharpen fuzzy terms, cross-reference claims with code, update `CONTEXT.md` when domain terms are resolved, and offer ADRs only when a decision deserves one.

Core principle: align the user, codebase, and documentation before building.

## When to Use

Use when:
- The user wants to stress-test a plan in an existing codebase.
- The plan depends on domain language, architecture, business concepts, or historical decisions.
- The repo may contain `CONTEXT.md`, `CONTEXT-MAP.md`, ADRs, docs, tests, specs, or source code that should constrain the answer.
- The user mentions `grill-with-docs`, project language, ubiquitous language, domain model, context docs, or ADRs.
- You need to clarify what terms mean before writing code or an implementation plan.

Don't use when:
- There is no existing codebase or documentation context; use `grill-me`.
- The user asks for a quick isolated edit that does not touch domain language or architecture.
- The task is pure debugging with a known failure and no planning component; use `systematic-debugging`.

## Required Behavior

### Ask One Question at a Time

Interview the user relentlessly, but ask exactly one question at a time and wait for feedback before continuing.

For each question, provide your recommended answer or default direction.

```markdown
Question: [single focused question]

Recommended answer: [your suggested answer based on docs/code/current constraints]
```

### Explore Instead of Asking When Possible

If a question can be answered by exploring the codebase or documentation, inspect the codebase instead of asking.

Use Hermes file tools for evidence:

```python
search_files("CONTEXT.md|CONTEXT-MAP.md|ADR|README|AGENTS.md|CLAUDE.md", path=".")
search_files("*.md", target="files", path="docs", limit=50)
read_file("CONTEXT.md")
search_files("Order|Customer|Invoice", path="src")
```

When code and user statements conflict, surface the contradiction immediately.

Example:

```markdown
Your code appears to cancel entire Orders in `src/orders/cancel.ts`, but you just described partial cancellation. Which one should be the domain truth?

Recommended answer: if partial cancellation is a real business capability, we should introduce a separate term for it and update the relevant code/docs rather than overloading "cancellation".
```

## Domain Documentation Model

### Single-Context Repos

Most repos have one bounded context and can use a root `CONTEXT.md`:

```text
/
├── CONTEXT.md
├── docs/
│   └── adr/
│       ├── 0001-event-sourced-orders.md
│       └── 0002-postgres-for-write-model.md
└── src/
```

### Multi-Context Repos

If `CONTEXT-MAP.md` exists at the root, the repo has multiple contexts. Read it to understand where each context lives and how they relate.

```text
/
├── CONTEXT-MAP.md
├── docs/
│   └── adr/                 # system-wide decisions
├── src/
│   ├── ordering/
│   │   ├── CONTEXT.md
│   │   └── docs/adr/        # context-specific decisions
│   └── billing/
│       ├── CONTEXT.md
│       └── docs/adr/
```

When multiple contexts exist, infer which context the current topic belongs to. If unclear, ask one question to choose the context.

### Create Documentation Lazily

Do not create files just because the skill loaded. Create documentation only when there is resolved content to record.

- If no `CONTEXT.md` exists, create it when the first domain term is resolved.
- If no `docs/adr/` exists, create it when the first ADR is needed.
- If a `CONTEXT-MAP.md` exists, put glossary updates in the appropriate context's `CONTEXT.md`.

## During the Session

### Challenge Against the Glossary

When the user uses a term that conflicts with existing language in `CONTEXT.md`, call it out immediately.

Example:

```markdown
Your glossary defines "cancellation" as a full Order cancellation, but you seem to mean removing one item from an Order. Which should this plan use?

Recommended answer: keep "Cancellation" for the full Order operation and introduce "Line Removal" or another precise term for item-level changes.
```

### Sharpen Fuzzy Language

When the user uses vague or overloaded terms, propose a precise canonical term.

Example:

```markdown
You're saying "account". In this repo, do you mean Customer, User, Organization, or Billing Account?

Recommended answer: use "Customer" for the buyer relationship and "User" for login identity, unless the codebase already defines different terms.
```

### Discuss Concrete Scenarios

When domain relationships are unclear, invent specific scenarios that force precision.

Examples:
- A customer places an order, then removes one item before fulfillment.
- A manager approves a request, then the requester changes it.
- A payment succeeds but fulfillment fails.
- A user belongs to two organizations with different permissions.

Ask which concept owns the scenario and what words should be used.

### Cross-Reference With Code

When the user states how something works, check whether code agrees. Search for the domain terms, read the implementation, and compare.

If you find contradiction, surface it as a decision point. Do not silently let the plan drift away from the implementation.

### Update `CONTEXT.md` Inline

When a term is resolved, update `CONTEXT.md` immediately. Do not batch these updates until the end; capture language while it is fresh.

`CONTEXT.md` is a glossary, not a spec. It should be devoid of implementation details.

Use this format, also available as `references/CONTEXT-FORMAT.md`:

```markdown
# {Context Name}

{One or two sentence description of what this context is and why it exists.}

## Language

**Order**:
{A one or two sentence description of the term}
_Avoid_: Purchase, transaction

**Invoice**:
A request for payment sent to a customer after delivery.
_Avoid_: Bill, payment request
```

Rules:
- Be opinionated. Pick one canonical term and list alternatives under `_Avoid_`.
- Keep definitions tight: one or two sentences max.
- Define what the term is, not what it does.
- Only include project/domain-specific terms. Do not add general programming concepts.
- Group terms under subheadings only when natural clusters emerge.

### Offer ADRs Sparingly

Only offer to create an ADR when all three are true:

1. **Hard to reverse** — changing the decision later has meaningful cost.
2. **Surprising without context** — a future reader will wonder why this path was chosen.
3. **Real tradeoff** — there were genuine alternatives and a specific reason for choosing one.

If any condition is missing, skip the ADR.

ADRs live in `docs/adr/` and use sequential numbering: `0001-slug.md`, `0002-slug.md`, etc. Use the compact format in `references/ADR-FORMAT.md`:

```markdown
# {Short title of the decision}

{1-3 sentences: what's the context, what did we decide, and why.}
```

Optional sections like status, considered options, and consequences are allowed only when they add real value. Most ADRs should be short.

## Grilling Flow

1. Find project instructions and docs: `AGENTS.md`, `README`, docs, existing context docs, ADRs.
2. Identify whether the repo is single-context or multi-context.
3. Read the relevant `CONTEXT.md` and existing ADRs.
4. Ask one plan-grilling question at a time.
5. For each answer, compare against glossary, docs, and code.
6. When language is resolved, update `CONTEXT.md`.
7. When a hard-to-reverse, surprising tradeoff is resolved, offer an ADR.
8. Continue until the plan and language are aligned enough to write a brief or implementation plan.

## Output Pattern

During the session:

```markdown
Question: ...

Recommended answer: ...

Why this matters: ...
```

After the session is sufficiently aligned:

```markdown
# Alignment Summary

## Plan
- ...

## Domain Language Updates
- `CONTEXT.md`: added/changed ...

## Decisions
- ADR offered/created/skipped: ...

## Remaining Questions
- ...

## Recommended Next Step
- ...
```

If implementation should follow, load/use `writing-plans` and write a bite-sized plan grounded in the resolved language and docs.

## Common Pitfalls

1. **Turning the session into a questionnaire.** Ask one question at a time, with a recommended answer.

2. **Updating `CONTEXT.md` with implementation details.** It is a glossary only. Specs, algorithms, and implementation decisions belong elsewhere.

3. **Creating ADRs too often.** Most decisions do not deserve ADRs. Require all three criteria: hard to reverse, surprising, real tradeoff.

4. **Letting user language override code silently.** If the code says one thing and the user says another, make the conflict explicit.

5. **Ignoring multi-context boundaries.** In a repo with `CONTEXT-MAP.md`, terms may mean different things in different contexts. Update the right context.

6. **Forgetting to write the docs.** This skill is not just a conversation pattern. When language or architectural decisions crystallize, update the files.

7. **Over-documenting generic concepts.** Do not add terms like "timeout", "repository", or "service" unless they are domain-specific concepts in this project.

## Verification Checklist

Before ending a grill-with-docs session:

- [ ] You inspected available docs/code instead of asking answerable questions.
- [ ] You asked one question at a time.
- [ ] Each question included a recommended answer.
- [ ] Existing `CONTEXT.md` / `CONTEXT-MAP.md` / ADRs were considered if present.
- [ ] Conflicts between user language, docs, and code were surfaced.
- [ ] Resolved domain terms were written to the appropriate `CONTEXT.md`.
- [ ] ADRs were offered only for hard-to-reverse, surprising, tradeoff-based decisions.
- [ ] The final summary names updated docs, remaining questions, and the recommended next step.
