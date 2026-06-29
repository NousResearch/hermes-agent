---
name: wds-3-scenarios
description: Create UX scenario outlines from Trigger Map through structured micro-steps
---

# Phase 3: UX Scenarios

**Goal:** Transform the Trigger Map into concrete UX scenario outlines — linear sunshine paths that expose all pages for design scrutiny.

**Your Role:** UX Scenario Facilitator collaborating with the project owner — you ASK, the user DECIDES. You bring scenario thinking and user journey expertise. Work together as equals.

---

## WORKFLOW ARCHITECTURE

This uses **step-file architecture** for disciplined execution:

### Core Principles

- **Micro-file Design**: Each step is a self-contained instruction file
- **Just-In-Time Loading**: Only current step file is in memory
- **Sequential Enforcement**: Steps must be completed in order
- **User Checkpoints**: Steps 02 and 04 require user approval before proceeding
- **Quality Validation**: Step 07 validates all scenarios against rubric

### Step Processing Rules

1. **READ COMPLETELY**: Always read entire step file before taking any action
2. **FOLLOW SEQUENCE**: Execute all sections in order, never deviate
3. **LOAD NEXT**: When directed, load, read entire file, then execute next step
4. **CHECKPOINT**: When a step says "wait for user", do NOT auto-proceed

### Critical Rules (NO EXCEPTIONS)

- **NEVER** load multiple step files simultaneously
- **ALWAYS** read entire step file before execution
- **NEVER** skip steps or optimize the sequence
- **ALWAYS** follow the exact instructions in step file

### Prerequisites

- Phase 1: Product Brief (required)
- Phase 2: Trigger Map (required)

---

## Steps

| # | File | Purpose |
|---|------|---------|
| 01 | [Load Context](steps-c/step-01-load-context.md) | Read all prerequisite artifacts, detect project state |
| 02 | [Analyze Scope](steps-c/step-02-analyze-scope.md) | Determine site type, pages, scale strategy (user checkpoint) |
| 03 | [Build Strategic Context](steps-c/step-03-build-strategic-context.md) | Extract strategic context from Trigger Map |
| 04 | [Suggest Scenarios](steps-c/step-04-suggest-scenarios.md) | Present scenario plan for approval (user checkpoint) |
| 05 | [Outline Scenario](steps-c/step-05-outline-scenario.md) | Detail ONE scenario (loops for each) |
| 06 | [Generate Overview](steps-c/step-06-generate-overview.md) | Create 00-ux-scenarios.md index file |
| 07 | [Quality Review](steps-c/step-07-quality-review.md) | Self-review against rubric |
| 08 | [Update Design Log](steps-c/step-08-update-design-log.md) | Document phase completion in project log |
| 09 | [Handover](steps-c/step-09-handover.md) | Complete Phase 3, prepare Phase 4 |

---

## INITIALIZATION

### 1. Configuration Loading

Load and read full config from `{project-root}/_bmad/wds/config.yaml` and resolve:

- `project_name`, `output_folder`, `user_name`, `communication_language`, `document_output_language`

### 2. Design Log

Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.

### 3. Mode Determination

**Check invocation:**
- "validate" / -v → Load and execute `./workflow-validate.md`
- Default (create) → Continue to step 3

### 4. First Step

Load and execute `./steps-c/step-01-load-context.md` to begin.

---

## REFERENCE CONTENT

| Location | Purpose |
|----------|---------|
| `data/quality-checklist.md` | Scenario quality checklist |
| `data/scenario-outline-template.md` | Scenario outline template |
| `data/validation-standards.md` | Validation standards |

---

## OUTPUT

- `{output_folder}/C-UX-Scenarios/00-ux-scenarios.md` — Scenario index with coverage matrix
- `{output_folder}/C-UX-Scenarios/NN-[scenario-name]/NN-[scenario-name].md` — Individual scenario outlines

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action or proceed to Phase 4: UX Design
