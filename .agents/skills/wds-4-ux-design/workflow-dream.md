---
name: 'workflow-dream'
description: 'The agent creates a complete scenario flow autonomously, then presents the result for user review.'
---

# [D] Dream Up — Agent Creates Autonomously, User Reviews

**Goal:** The agent creates a complete scenario flow autonomously, then presents the result for user review.

**When to use:** When the user trusts the agent to make good decisions and prefers to review a complete proposal rather than approve each step.

---

## INITIALIZATION

Read design log at `{output_folder}/_progress/00-design-log.md` before starting.

## Entry

Load scenario context from `{output_folder}/C-UX-Scenarios/`.

### Scenario Check (CRITICAL GATE)

Before starting page design, verify that a scenario exists for the selected scenario:

1. Look for scenario files in `{output_folder}/C-UX-Scenarios/[NN-slug]/[NN-slug].md`
2. **If a Phase 3 scenario exists** → Skip to **Process** below. The scenario's 8-question answers, shortest path, and first page specification provide everything needed.
3. **If NO scenario exists** → Do NOT attempt to define the scenario here. Instead:
   - Inform the user: *"Before we design pages, we need a scenario outline. This gives us the user's device, mental state, entry point, and the shortest path — all essential for good page design."*
   - Suggest returning to Phase 3 to outline the scenario using the 8-question dialog
   - The user can then return here with [D] from the Phase 3 post-scenario menu

**Why:** Phase 3's 8-question dialog is the canonical way to define scenarios. It produces richer, more grounded scenarios than trying to shortcut the process here.

### Phase 3 Handover Context

When entering from Phase 3's [D] option (start designing), the scenario file and page folders already exist. Use:
- **Page folders** from `{output_folder}/C-UX-Scenarios/[NN-slug]/pages/[NN].1-[page-slug]/` — each page has a boilerplate `.md` and a `Sketches/` subfolder
- **First page spec** (`[NN].1-*.md`) has full entry context (device, arrival, mental state) from Q4, Q5, Q6
- **Shortest path** from Q8 to know the full page sequence

## Process

The Dream workflow uses the same steps as Suggest (`./steps-s/`) but with **autonomous execution**:

1. **Agent creates all pages** (step-08 through step-15) for each page in the flow
2. **Agent presents the complete result** for user review

### Agent Behavior

- Make reasonable decisions at each step based on Trigger Map, scenario context, and WDS patterns
- Document decisions and rationale as you go
- When uncertain, choose the simpler option
- After completion, present a summary of all decisions made
- User can accept, request changes, or switch to **[S] Suggest** for finer control

### Mode Override Rule (CRITICAL)

Step files in `./steps-s/` contain rules like "ALWAYS halt and wait for user input" and "NEVER generate content without user input." **These rules apply ONLY in Suggest mode.**

In Dream mode:
- **OVERRIDE** all "halt and wait" rules — auto-proceed after completing each step
- **OVERRIDE** "NEVER generate content without user input" — generate based on context and WDS patterns
- **DO NOT** display menus or wait for menu selections between steps
- **DO** still save outputs and update the design log at each step
- **DO** still follow the step's actual instructions for what to generate
- The user can type **"stop"** or **"pause"** at any time to interrupt and switch to Suggest mode

**Reference data:**
- `./data/scenario-init/` — scenario guides and examples
- `./data/page-creation-flows/` — page creation approaches

---

## DESIGN LOG REPORTING

In Dream mode, the agent updates the design log autonomously at each page completion. Append to the Design Loop Status table in `{output_folder}/_progress/00-design-log.md`:

```
| [Scenario slug] | [NN.X] | [Page name] | specified | [YYYY-MM-DD] |
```

Do NOT skip this — even in autonomous mode, the design log must be updated per page.

## AFTER COMPLETION

### Autonomous Mode (all pages at once)

When Dream mode completes all pages in the scenario, present a summary for review:

<output>
**Dream complete! Here's what I created for [Scenario Name]:**

| Step | Page | Status | Key Decisions |
|------|------|--------|---------------|
| [NN.1] | [page name] | specified | [brief summary] |
| [NN.2] | [page name] | specified | [brief summary] |
| ... | ... | ... | ... |

**Shared components extracted:** [list if any]

Review the pages and let me know what to adjust. When you're happy:

1. **Start building** — hand the first page to agentic development
2. **Explore responsive states / storyboard** — if any pages need detail work
</output>

### Per-Page Mode (user interrupted or reviewing one at a time)

Present the same two-option transition as Discuss and Suggest:

**If complexity exists on this page:**

<output>
**Specification for "[page name]" is complete.**

This page has [responsive states / storyboard items / complex functionality] that need exploring.

1. **Explore [responsive states / storyboard / functionality]** — define the details
2. **Explore the next scenario step** — [next page name]
</output>

**If simple page:**

<output>
**Specification for "[page name]" is complete. Ready to build.**

1. **Build it** — start agentic development
2. **Explore the next scenario step** — [next page name]
</output>

### Component Extraction (Dream Mode)

In Dream mode, component extraction runs automatically:
1. Scan completed page specs silently after each page
2. If shared elements found, auto-extract as shared components (log decisions)
3. Reference shared components in subsequent page specs instead of duplicating
4. Include extraction summary in the final review presentation

### Execution Rules

- ALWAYS halt and wait for user input after presenting review/transition
- User can type "stop" or "pause" to interrupt autonomous mode
- The user can always say "stop" to pause and return later — log current status
