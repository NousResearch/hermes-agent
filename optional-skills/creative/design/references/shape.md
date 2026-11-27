# Shape — Plan Before Building

`shape` is the planning command. Use it before writing any UI code. It returns a confirmed design brief, not a code artifact.

## Inputs

- Brief or user intent.
- Existing design files, if any.
- Target framework (Angular, React, Blazor, or unknown).
- Primary viewport (default: mobile).

## Phase 1: Discovery interview

Ask the two or three questions that most change the result. Do not dump a questionnaire, repeat settled facts, or turn obvious facts into menus.

**Round 1 — purpose, people, outcome**

- What is this surface or feature for, and what problem must it solve?
- Who specifically reaches it, in what situation and state of mind?
- What is the primary thing they must understand or do? What would success look like?
- What is uniquely true here that a neighboring product or generic template could not claim?

**Round 2 — material, behavior, boundaries**

Run only when material decisions are unresolved:

- What real content, evidence, data, and assets must this carry? What are realistic minimum, typical, and maximum ranges?
- Which states and transitions matter: first-run, empty, loading, error, success, permissions, overflow, expert use?
- What is the intended fidelity, breadth, and interactivity: exploration, production-ready screen, full flow, or broader surface?
- What must remain untouched? What would make the result feel wrong even if it looked polished?
- Which platform, framework, performance, accessibility, localization, or delivery constraints are binding?

Never ask for CSS values or canned aesthetic lanes.

## Phase 2: Resolve the direction

For new surfaces or replacement identities, propose 2–3 materially different visual directions ordered by resonance. Inside an established world, inherit it and resolve only the new purpose, hierarchy, states, and interaction.

## Phase 3: Write the brief

Write the smallest useful brief with:

1. **Job and audience:** who arrives, their context, need, and visitor mode.
2. **Outcome and proof:** primary task/action, success, real evidence, and product-specific truth.
3. **Selected direction:** visual authority, structural thesis, sequence, focal moment.
4. **Scope and boundaries:** fidelity, breadth, interactivity, what stays untouched, anti-goals.
5. **States and ranges:** realistic content/data ranges and material states.
6. **Interaction and layout:** hierarchy, topology, responsiveness, affordances, feedback, transitions; intent, not CSS.
7. **Constraints and open decisions:** framework, CSS approach, accessibility, i18n, reusable components.

Use three to five bullets when the task is settled; use the full structure only for ambiguous, multi-screen, or standalone planning.

## Constraints

- No code until the plan is approved.
- No placeholder text like "TBD" or "we will decide later."
- Every decision must be traceable to the brief or a design principle.
- Present the brief for explicit confirmation or one correction round, then stop.
