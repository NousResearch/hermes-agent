---
name: meeting-recap-deck
description: Turn meeting notes into a traceable recap outline.
version: 0.1.0
author: Mark (unsupportedpastels), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [meetings, recaps, slides, action-items]
    category: bots
    related_skills: [powerpoint, meeting-action-items, document-to-action-items]
---

# Meeting Recap Deck Workflow Skill

Convert one meeting's supplied notes into a traceable recap outline, preserving decisions, proposals, owners, dates, and exact quotes. A real PPTX is optional and depends on configured `powerpoint` setup; a plain recap is always the honest fallback.

## When to Use

- The user supplies notes/transcript, audience, and preferred recap format.
- The user wants a concise deck outline or plain recap.
- Do not use to merge meetings, invent commitments, send files, or create tickets.

## Prerequisites

The first useful task requires only pasted or local notes readable with `read_file`. If the bundled `powerpoint` skill and its dependencies are configured, it may render a PPTX; otherwise deliver Markdown from `templates/recap-outline.md`. Skill presence alone does not prove document tooling is ready.

## How to Run

1. Copy `templates/recap-outline.md` to a collision-safe path in the owning profile workspace.
2. Try `references/sample-input.md` or use user notes.
3. Apply `references/expected-output-rubric.md` before optional PPTX rendering.

## Quick Reference

| Note evidence | Recap label |
|---|---|
| Explicitly agreed | Decision |
| Suggested but not accepted | Proposal |
| Person and task stated | Action; date only if stated |
| Missing/conflicting attribution | Unresolved |

A post-meeting routine may be recommended. Do not automatically create a scheduled job; explicit activation must approve the note source, meeting scope, cadence, retention, template, and delivery.

## Procedure

### 1. Establish source and audience

Record meeting identity, date, audience, supplied files, template preference, and output mode. Keep one meeting per recap unless the user explicitly asks otherwise. **Done when** source boundaries and intended reader are explicit.

### 2. Build a traceability table

Extract themes, decisions, proposals, open questions, actions, owners, deadlines, quotes, and numbers with note-line or timestamp references. **Done when** every planned slide fact has a source pointer or unresolved label.

### 3. Draft the narrative

Lead with what was heard, then decisions, unresolved questions, and next steps. Never turn a suggestion into an accepted commitment. **Done when** the outline is concise and evidence labels remain intact.

### 4. Choose the artifact honestly

Produce the plain recap by default. Render a PPTX only if `powerpoint` prerequisites are actually available and a template/output path is approved. Do not claim a deck file exists unless it was generated and verified. **Done when** the chosen artifact opens or the Markdown fallback is complete.

### 5. Verify and preserve

Apply `references/expected-output-rubric.md`, preserve notes and templates, and write to a new output path. Do not send or share. **Done when** names, numbers, owners, dates, and quotes trace to the notes and no collision occurred.

## Pitfalls

- Speaker diarization and unattributed notes can make ownership uncertain.
- A proposed action is not a commitment.
- Template matching cannot be promised without the template and working tooling.
- Do not schedule, ticket, send, or share merely because the recap is finished.

## Verification

- [ ] Every quote, number, owner, and deadline traces to supplied notes.
- [ ] Decisions, proposals, and unresolved items are distinct.
- [ ] PPTX delivery is claimed only after generation and open/parse verification.
- [ ] Original notes and template remain unchanged.
