---
name: documentation
description: >
  Drafts or refines product specifications (requirements docs) — WHAT we're building or changing
  and WHY it matters, framed from the user's perspective. Also provides structured templates
  for writing clear, actionable documentation. Use when the user asks to draft requirements,
  write a product spec, capture acceptance criteria, or create documentation for a feature.
  Do not use for technical/architectural planning (see the technical-planning skill).
version: 1.0.0
author: community
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [documentation, requirements, product-spec, acceptance-criteria, specification]
    sources:
      - eqlion/skills-and-agents/skills/requirements
---

# Documentation (Product Spec)

`requirements.md` is the **product spec** for a task. It answers WHAT we're building or changing and WHY it matters — framed from the user's perspective, not the codebase's. The HOW (architecture, layering, tradeoffs) belongs in `plan.md`; use the `technical-planning` skill for that.

This skill is content-only. It produces documentation; it does not move task folders between stages or manage branches.

## When to invoke

The user wants to:
- Draft the initial product spec for a new task
- Refine an existing `requirements.md` after a product conversation
- Capture acceptance criteria, links, or out-of-scope items for the product side of a task
- Write documentation for a feature, API, or system

## Where the file lives

Find the most relevant task folder. If working in a project with a standard layout:
- If there is exactly one active task folder, use it
- Otherwise, look in draft/staging folders
- If multiple candidates exist or no folder is obvious, ask the user which task this is for. Do not guess.

Write or edit `requirements.md` (or the appropriate documentation file). If the file does not exist, create it from the scaffold below. If it does exist, edit it in place — preserve any content the user has already written, and ask before deleting sections.

## Scaffold

```markdown
# <task-id> — <title>

<!--
PRODUCT spec. WHAT we're building or changing and WHY it matters. The HOW
(architecture, layering, tradeoffs) lives in plan.md. Do not put
implementation details here.
-->

## Goal
<!-- One or two sentences: what user-observable outcome are we trying to
     achieve? Frame it from the user's perspective, not the codebase's. -->

## Problem / motivation
<!-- What's the current state, and what's wrong or missing about it? Who is
     affected? What business or product driver makes this worth doing now? -->

## Acceptance criteria
<!-- Bulleted list of user-observable conditions that must hold for the task
     to be considered done. Each bullet should be checkable without reading
     the code. -->

## Out of scope (product)
<!-- Things adjacent stakeholders might assume we're doing but we're
     explicitly not. One line each. -->

## Links
<!-- Jira ticket, Figma frames, BE/API specs, related tickets, Confluence
     docs. -->

## Notes
<!-- Any product/UX constraints, open product questions, edge cases the
     spec hasn't fully resolved yet. -->
```

## Drafting rules

- **User-observable, not implementation-observable.** Acceptance criteria should be things a PM or QA could verify without reading the codebase. "Tapping the deeplink opens the challenge screen and shows a confirmation toast" is good; "the parser registers a new `DeepLinkDestination`" is not.
- **Why, not how.** If you're describing modules, layers, services, types, or files, you're in plan territory. Move it.
- **Link, don't copy.** When the source of truth is Jira/Figma/Confluence, link to it. Quote only the parts that are load-bearing for the task.
- **Surface unknowns, don't hide them.** If the spec has gaps (e.g. error-state copy missing, edge case ambiguous), record them under **Notes** as open questions rather than inventing answers.
- **Scale to complexity.** A two-line bugfix needs Goal + Acceptance criteria + Links. A new feature needs the full scaffold.

## API documentation template

When documenting APIs specifically:

```markdown
# <API Name> Documentation

## Overview
<!-- One paragraph: what this API does and who it's for -->

## Authentication
<!-- How to authenticate, required headers/tokens -->

## Endpoints

### <Method> <Path>

**Description:** <what it does>

**Request:**
- Parameters, body, headers

**Response:**
- Status codes, response body schema

**Example:**
```
<request/response example>
```

## Error handling
<!-- Error codes, error response format -->

## Rate limits
<!-- If applicable -->
```

## General documentation guidelines

1. **Start with the user's perspective** — what problem does this solve for them?
2. **Be specific and checkable** — acceptance criteria and claims that can be verified
3. **Distinguish WHAT from HOW** — the product spec describes outcomes; the plan describes implementation
4. **Surface ambiguities explicitly** — open questions are better than assumed answers
5. **Link to sources** — don't duplicate information that lives elsewhere
6. **Keep it concise** — the spec should be scannable, not exhaustive

## Hand-off

When the product spec is solid enough that someone can start thinking about HOW, point the user at the `technical-planning` skill. It is fine — and common — to iterate on requirements after a planning conversation surfaces ambiguities.
