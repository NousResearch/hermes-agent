---
name: grill-me
description: "Use when the user wants to stress-test a new plan, product, feature, architecture, or design through a relentless one-question-at-a-time interview until shared understanding is reached."
version: 1.0.0
author: Hermes Agent, adapted from Matt Pocock
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [planning, product, requirements, discovery, decision-tree, interview]
    homepage: https://github.com/mattpocock/skills/blob/main/skills/productivity/grill-me/SKILL.md
    related_skills: [writing-plans, ideation]
---

# Grill Me

## Overview

Use this skill to interview the user relentlessly about a plan or design until you and the user reach shared understanding. Walk down each branch of the decision tree, resolving dependencies between decisions one by one.

This skill is for new work: a new app, project, product, feature concept, integration, workflow, automation, or design. If the plan is being evaluated against an existing codebase and project documentation, use `grill-with-docs` instead.

Core principle: ask the next highest-leverage question, not every question at once.

## When to Use

Use when the user:
- Says "grill me" or asks to be grilled on an idea.
- Wants to stress-test a plan, product, design, or architecture.
- Has a new application or project idea that needs requirements discovery.
- Wants help turning a vague concept into a shared understanding.
- Needs tradeoffs and assumptions surfaced before implementation planning.

Don't use when:
- The user has already supplied a complete spec and asks for execution.
- The task is inside an existing repo and can be answered by inspecting docs/code; use `grill-with-docs`.
- The user asks for pure brainstorming without interrogation; use `ideation`.
- The user explicitly says to skip discovery and build.

## Required Behavior

### Ask One Question at a Time

Do not send a giant questionnaire. Ask exactly one focused question, wait for the user's answer, then ask the next question.

Bad:

```markdown
Here are 20 questions we need to answer...
```

Good:

```markdown
First question: who is the first real user of this, and what are they trying to get done?

My recommended answer, if you're building for the fastest useful MVP: pick one concrete initial user group rather than "everyone".
```

### Provide Your Recommended Answer

For each question, include your recommended answer or default direction. The point is not to make the user do all the thinking; the point is to pressure-test the plan together.

Pattern:

```markdown
Question: [single focused question]

Recommended answer: [your suggested default, with reasoning]
```

If the answer depends on facts you do not have, say what assumption your recommendation is based on.

### Resolve Dependencies in Order

Walk the design tree in dependency order:

1. User and problem.
2. Desired outcome.
3. MVP workflow.
4. Data and state.
5. Roles and permissions.
6. Integrations and runtime environment.
7. UX and operational constraints.
8. Success criteria and failure modes.
9. Implementation plan.

Do not jump to tech stack before the user, workflow, and constraints are clear.

### Explore Instead of Asking When Possible

If a question can be answered by exploring available artifacts, use tools instead of asking the user.

Examples:
- If the user says there is a repo, inspect it.
- If the user links docs, read them.
- If a config file reveals deployment target, use that evidence.
- If a design file or screenshot is available, inspect it before asking UX questions.

## Question Strategy

Ask questions that collapse uncertainty. Favor questions that change scope, architecture, or risk.

### Start Here

First question usually:

```markdown
Who is the first real user for this, and what painful job are they trying to get done?

Recommended answer: choose the narrowest user group that would benefit from the first version. Avoid "everyone" until the workflow is proven.
```

If the user already gave the user/problem, ask:

```markdown
What is the smallest end-to-end workflow that would make this useful?

Recommended answer: define one demoable path from input to valuable output, and postpone everything outside that path.
```

### High-Leverage Questions

Use these as the decision tree unfolds:

- Who is the first user who would be annoyed if this did not exist?
- What are they doing today instead?
- What is the first moment where the project creates value?
- What is the smallest demo that would make you say "yes, that's it"?
- What must be in v1, and what must definitely not be in v1?
- What can be manual behind the scenes for the first version?
- What data does the app need before it can be useful?
- What data is created, edited, archived, exported, or deleted?
- Who can view, create, approve, edit, or delete each thing?
- What should happen when the user makes a mistake?
- What should be audited or impossible to silently change?
- What external services must this depend on?
- What should the first screen show?
- What does success look like after the first week of use?
- What would make the project a failure even if the software works?

## Synthesis After Answers

After each answer, briefly update the shared understanding before asking the next question.

Use this compact shape:

```markdown
Got it. Current understanding:
- User: ...
- Problem: ...
- MVP boundary: ...
- Open uncertainty: ...

Next question: ...

Recommended answer: ...
```

Keep the synthesis short. The session should feel like a guided interview, not a report after every answer.

## When Enough Is Known

When the major branches are resolved, stop grilling and produce a short planning artifact:

```markdown
# Project Brief: [Name]

## One-Sentence Concept
[Thing] for [user] to [outcome].

## MVP Workflow
1. ...
2. ...
3. ...

## In Scope
- ...

## Out of Scope
- ...

## Key Decisions
- ...

## Open Questions
- ...

## Recommended Next Step
...
```

If the user wants implementation next, switch to `writing-plans` and write a bite-sized implementation plan. If a codebase becomes involved, switch to `grill-with-docs` before implementation planning.

## Common Pitfalls

1. **Asking a wall of questions.** The original point of the skill is one-question-at-a-time pressure testing. Do not turn it into a questionnaire unless the user asks.

2. **Withholding recommendations.** The user asked to be grilled, not abandoned. Every question should include a recommended answer or default direction.

3. **Jumping to implementation.** Do not choose frameworks, schemas, or APIs before the core workflow and constraints are clear.

4. **Treating every idea as MVP.** Aggressively separate first useful version from nice-to-have scope.

5. **Ignoring available evidence.** If files, docs, links, screenshots, or repos can answer a question, inspect them instead of asking.

6. **Letting vague language pass.** Words like "user", "account", "admin", "document", "approval", and "status" often hide multiple concepts. Ask for precision.

## Verification Checklist

Before ending a grill-me session:

- [ ] You asked one question at a time.
- [ ] Each question included a recommended answer.
- [ ] Available artifacts were inspected instead of asking answerable questions.
- [ ] User, problem, outcome, and MVP workflow are clear.
- [ ] Scope boundaries are explicit.
- [ ] Key risks, assumptions, and open questions are named.
- [ ] The next step is clear: more grilling, project brief, prototype, implementation plan, or build.
