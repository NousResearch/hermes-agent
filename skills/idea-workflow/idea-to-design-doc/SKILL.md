---
name: idea-to-design-doc
description: "Use when turning a rough idea into a focused product/design Markdown doc through guided questions, without moving into implementation too early."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [idea-workflow, note-taking, product-design, design-doc, brainstorming]
    related_skills: [idea-superpowers-suite, idea-to-implementation-doc]
---
# Idea to Design Doc Workflow

Use this skill when the user says things like:
- "I have an idea for an app"
- "I have an idea for a physical product / 3D-printable product / digital-download design"
- "Help me flesh out this product"
- "Turn this thought into a design doc"
- "Make me a list of ideas and link each one to a note"

## Goal

Convert a rough idea into a structured design document with:
1. A master ideas index
2. One markdown note per idea
3. A guided interview that stops when the user says `stop`, `that's enough`, or `done`
4. The product idea, philosophy, intended experience, and practical technical shape needed to understand the app

This workflow starts with *product thinking* and only moves into technical aspects after the concept/philosophy is clear. The design doc should capture what the user wants and enough technical direction to support a later build plan.

## Important constraints

- Do **not** push into backend, frameworks, database design, or code architecture too early.
- First capture product intent, behavior, audience, layout, features, UX, and philosophy.
- Then, when drafting the design doc, include the relevant technical aspects: platform assumptions, major system components, data needs, integrations, constraints, and likely architecture questions.
- For Full mode or build-bound ideas, recommend sensible technical defaults first, then let the user accept or change them. Cover database/storage, hosting/deployment, backend/runtime, frontend/UI, auth, platform targets, and other technical pieces only after the product direction is clear.
- Ask one question at a time when possible.
- If the user gives a short answer, follow up with a narrower question.
- If the user says to stop, immediately draft the note from what you have.
- For cost-aware runs, cheaper workers may draft, critique, or research supporting material, but the orchestrator remains responsible for interpreting user intent, asking questions, and final synthesis. Follow `idea-superpowers-suite` Cost-Aware Worker Delegation rules when available.

## Default storage layout

Save files locally first, rather than directly into the Obsidian vault.

Recommended local folder structure:

- `./ideas/Ideas Index.md`
- `./ideas/<idea-title>.md`

If the user later wants Obsidian export, treat that as a separate export step.

## Lite vs Full mode

Use **Lite mode** for quick idea capture: ask only enough to produce a useful note. Use **Full mode** when the user wants a durable design doc or eventual implementation handoff.

The user can force the next stage with the exact phrase **GREENLIGHT NEXT STAGE**. If they use it, stop questioning, draft with current information, and record gaps under **Open questions**.

## Interview flow

Ask questions in this order, adapting to the user's answers. For stronger prompts, use `idea-superpowers-suite/references/interview-question-bank.md`.

## Interview flow

Ask questions in this order, adapting to the user's answers. For stronger prompts, use `idea-superpowers-suite/references/interview-question-bank.md`:

1. **Working title**
   - "What should we call this idea for now?"
2. **Problem / purpose**
   - "What problem does it solve, or what is it for?"
3. **Audience**
   - "Who is it for?"
4. **Core behavior**
   - "What should it do when someone uses it?"
5. **Primary user flow**
   - "What is the main thing a user should be able to do from start to finish?"
6. **Key features**
   - "What are the must-have features?"
7. **Layout / organization**
   - "How should the main parts be arranged or grouped?"
8. **Physical/digital-product specifics when relevant**
   - For 3D-printable, hardware, or physical-product ideas, ask about variants/models supported, materials, manufacturing/printing method, fit/tolerance needs, access/serviceability, safety, packaging/deliverables, sales channel, and validation/test-print criteria before moving into app-style hosting/platform questions.
9. **Data location / hosting preference**
   - "Should the data stay local, be self-hosted, go to Cloudflare/AWS/another cloud, or is that undecided?"
10. **Platform targets**
   - "Should this be web-only, or should it also have a Windows app, Mac app, cross-platform desktop app, mobile app, or responsive mobile web?"
11. **Recommended technical defaults**
   - "Based on this idea, I recommend these technical defaults: <short stack summary>. Do you want to accept them or change any part?"
12. **Preferences / feel**
   - "Should it feel simple, playful, serious, fast, calm, etc.?"
13. **Non-goals**
    - "What should this *not* do?"
14. **Success criteria**
    - "How will you know this idea is good enough to move forward?"

## Interview style

Use prompts like:
- "Tell me more about that part."
- "What would happen next?"
- "Can you give an example?"
- "Is that a must-have or a nice-to-have?"
- "Do you want it to be simple or more feature-rich?"

If the user is unsure, offer lightweight options instead of open-ended pressure:
- "Do you want this to be more like a list app, an overview page, or a single-focus tool?"
- "Should users see everything at once, or step through it one section at a time?"

## Stop condition

If the user says any of the following, stop interviewing and draft immediately:
- stop
- that's enough
- enough for now
- done
- draft it
- write it up
- GREENLIGHT NEXT STAGE

## Output format

When drafting, produce a markdown note with this structure:

```markdown
# <Idea Title>

## One-line summary

## Problem / purpose

## Product philosophy

## Target user

## Core concept

## Desired behavior

## Key features

## Layout / information architecture

## UX / product notes

## Technical shape

## Data / integrations / platform needs

## Hosting / data location

## Platform targets

## Recommended technical defaults

## Non-goals

## Open questions

## Next steps
```

## Master index format

Maintain `Ideas Index.md` as a simple bullet list of idea notes:

```markdown
# Ideas Index

- [[Idea Title]] — short one-line summary
- [[Another Idea]] — short one-line summary
```

If the index file exists, append the new idea in alphabetical or newest-first order, but stay consistent.

## Suggested response behavior

During the interview:
- Be conversational and concise.
- Ask a question, then wait for the user's answer.
- Keep the scope on product/design thinking.

When the user stops:
- Summarize the idea clearly.
- Capture uncertainties under **Open questions**.
- Save the note locally.
- Add/update the local index.
- Give the user the local note path.

## Quality bar

A good result should make it easy to revisit the idea later and immediately understand:
- what it is
- who it is for
- how it should behave
- what matters most
- what is intentionally out of scope
