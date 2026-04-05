---
name: Spec Intake
description: Engineering request intake protocol. Triggers before ANY request is routed to Vale (CTO) or the engineering pipeline. Asks structured questions to produce a real spec instead of a vague task. Prevents rework.
version: 1.0.0
tags: [engineering, spec, intake, planning, falconconnect, feature]
---

# Spec Intake Protocol

TRIGGER: Load this skill whenever Seb makes a request that involves building, adding, changing, or fixing something in FalconConnect, FalconFinancial.org, or any Falcon system. Do NOT route to engineering until this protocol completes.

## When to Run

Run spec intake when the request contains:
- "add a feature", "build", "create", "implement", "make it so that"
- "change how", "update the", "fix the way", "I want FC to"
- Any UI/frontend change request
- Any new integration or API connection
- Any new page, route, or dashboard widget

Do NOT run spec intake for:
- Pure bug reports where the expected behavior is obvious ("X is broken, it should do Y")
- Infra/DevOps tasks (deploy, restart, env var update)
- Research tasks (no code involved)
- Hotfixes where the cause and fix are already known

## The Five Questions

Ask ALL five in a single message. Keep it conversational, not like a form. Seb hates being interrogated.

**Format — send this to Seb:**

> Before I kick this to engineering, five quick things:
>
> 1. **What exactly?** [Restate what you understood in one sentence, then ask if that's right or if there's more to it]
> 2. **Who uses it?** Just you, your future downlines, or public-facing to clients/prospects?
> 3. **Where does it live?** New page/route, existing dashboard, API endpoint, or background process?
> 4. **How should it look/behave?** Any specific UI expectations? (Remember: dark mode default, industrial data table aesthetic per our design directive)
> 5. **Priority?** Ship this week, next sprint, or backlog for later?

## After Seb Answers

Take his answers and produce a **Spec Brief** — this becomes the Paperclip issue description for Vale (CTO).

### Spec Brief Format

```
# [Feature Name]

## What
[One paragraph, specific and unambiguous]

## Who
[User type and access level]

## Where
[Route, page, component, or API path]

## UI/UX Requirements
[Specific requirements from Seb's answer + design directive mandates]
- Dark mode default
- [Any specific aesthetic choices]
- [Mobile considerations if applicable]

## Technical Constraints
[Anything you know about the existing system that matters]
- [Existing endpoints/tables that are relevant]
- [Known limitations]

## Acceptance Criteria
[3-5 bullet points — what "done" looks like]
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]

## Priority
[Seb's stated priority]

## Notes
[Anything else relevant — related features, dependencies, risks]
```

## Then Route It

1. Create a Paperclip issue in Falcon Financial with the Spec Brief as the description
2. Assign to Vale (CTO)
3. Set priority based on Seb's answer
4. Tell Seb: "Spec filed, assigned to Vale. [one-line summary of what engineering will build]"

## Rules

- NEVER skip the five questions. Even if the request seems clear, Seb's mental model and the engineer's mental model diverge 80% of the time. The questions close that gap.
- NEVER ask more than five questions. If you need clarification on an answer, ask ONE follow-up max.
- If Seb says "just do it" or "you figure it out" — make your best guess on the unanswered questions, state your assumptions explicitly in the spec brief, and flag them as assumptions.
- If the request involves frontend work, ALWAYS reference the Frontend Design Directive in the spec brief's UI/UX section.
- Keep the whole intake under 3 messages total (your questions, his answers, your confirmation).
