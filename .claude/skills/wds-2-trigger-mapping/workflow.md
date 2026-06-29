---
name: wds-2-trigger-mapping
description: Map business goals to user psychology through structured workshops
---

# Phase 2: Trigger Mapping

**Goal:** Connect business goals to user psychology through structured workshops, creating a strategic reference that coordinates all teams.

**Your Role:** In addition to your name, communication_style, and persona, you are also a Strategic Analyst facilitating Effect Mapping workshops. This is a partnership, not a client-vendor relationship. You bring structured facilitation and pattern recognition, while the user brings business knowledge and user insight. Work together as equals.

---

## WORKFLOW ARCHITECTURE

Based on Effect Mapping by Mijo Balic & Ingrid Domingues (inUse). Adapted by WDS: simplified (no features), enhanced with negative driving forces.

This uses **step-file architecture** for disciplined execution:

### Step Processing Rules

1. **READ COMPLETELY**: Always read entire step file before taking any action
2. **FOLLOW SEQUENCE**: Execute all sections in order
3. **LOAD NEXT**: When directed, load and execute next step
4. **CHECKPOINT**: When a step says "wait for user", do NOT auto-proceed

### Critical Rules (NO EXCEPTIONS)

- **NEVER** load multiple step files simultaneously
- **ALWAYS** read entire step file before execution
- **NEVER** skip steps or optimize the sequence
- **ALWAYS** follow the exact instructions in step file

### Two Paths

- **From scratch** → step-01-overview.md (Workshop/Suggest/Dream modes)
- **From existing documentation** → step-00a-documentation-synthesis.md

### Prerequisites

- Phase 1: Product Brief (required)

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
- "existing" / from docs → Load and execute `./steps-c/step-00a-documentation-synthesis.md`
- Default (create from scratch) → Load and execute `./steps-c/step-01-overview.md`

---

## REFERENCE CONTENT

| Location | Purpose |
|----------|---------|
| `data/business-goals-template.md` | Business goals template |
| `data/key-insights-structure.md` | Key insights structure |
| `data/mermaid-formatting-guide.md` | Mermaid diagram formatting |
| `data/quality-checklist.md` | Quality checklist |

---

## OUTPUT

- `{output_folder}/B-Trigger-Map/trigger-map.md`
- `{output_folder}/B-Trigger-Map/personas/`
- `{output_folder}/B-Trigger-Map/feature-impact-analysis.md`

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action or proceed to Phase 3: UX Scenarios
