---
name: 'step-01b-resume'
description: 'Resume interrupted workflow from last completed step'
outputFile: '{test_artifacts}/test-design-progress.md'
---

# Step 1b: Resume Workflow

## STEP GOAL

Resume an interrupted workflow by loading the existing output document, displaying progress, and routing to the next incomplete step.

## MANDATORY EXECUTION RULES

- 📖 Read the entire step file before acting
- ✅ Speak in `{communication_language}`

---

## EXECUTION PROTOCOLS:

- 🎯 Follow the MANDATORY SEQUENCE exactly
- 📖 Load the next step only when instructed

## CONTEXT BOUNDARIES:

- Available context: Output document with progress frontmatter
- Focus: Load progress and route to next step
- Limits: Do not re-execute completed steps
- Dependencies: Output document must exist from a previous run

## MANDATORY SEQUENCE

**CRITICAL:** Follow this sequence exactly. Do not skip, reorder, or improvise.

### 1. Load Output Document

Read `{outputFile}` and parse YAML frontmatter for:

- `workflowStatus` — overall workflow state (`in-progress` or `completed`)
- `totalSteps` — total number of create-mode workflow steps
- `stepsCompleted` — array of completed step names
- `lastStep` — last completed step name
- `nextStep` — next step file to execute
- `lastSaved` — timestamp of last save

If `workflowStatus`, `totalSteps`, or `nextStep` are missing (legacy progress file), infer them from `lastStep` using this mapping:

- `'step-01-detect-mode'` → `workflowStatus: 'in-progress'`, `totalSteps: 5`, `nextStep: './step-02-load-context.md'`
- `'step-02-load-context'` → `workflowStatus: 'in-progress'`, `totalSteps: 5`, `nextStep: './step-03-risk-and-testability.md'`
- `'step-03-risk-and-testability'` → `workflowStatus: 'in-progress'`, `totalSteps: 5`, `nextStep: './step-04-coverage-plan.md'`
- `'step-04-coverage-plan'` → `workflowStatus: 'in-progress'`, `totalSteps: 5`, `nextStep: './step-05-generate-output.md'`
- `'step-05-generate-output'` → `workflowStatus: 'completed'`, `totalSteps: 5`, `nextStep: ''`

**If `{outputFile}` does not exist**, display:

"⚠️ **No previous progress found.** There is no output document to resume from. Please use **[C] Create** to start a fresh workflow run."

**THEN:** Halt. Do not proceed.

---

### 2. Display Progress Dashboard

Display:

"📋 **Workflow Resume — Test Design and Risk Assessment**

**Workflow status:** {workflowStatus}
**Last saved:** {lastSaved}
**Last completed step:** {lastStep}
**Next step:** {nextStep || 'None'}
**Steps completed:** {stepsCompleted.length} of {totalSteps}"

---

### 3. Route to Next Step

If `workflowStatus` is `'completed'`, display:
"✅ **All steps completed.** Use **[V] Validate** to review outputs or **[E] Edit** to make revisions."

**THEN:** Halt.

If `nextStep` is one of the known create-mode step files below, load it, read completely, and execute:

- `./step-02-load-context.md`
- `./step-03-risk-and-testability.md`
- `./step-04-coverage-plan.md`
- `./step-05-generate-output.md`

**If `nextStep` is empty or does not match a known step file**, display:
"⚠️ **Unknown progress state** (`workflowStatus`: {workflowStatus}, `lastStep`: {lastStep}, `nextStep`: {nextStep}). Please use **[C] Create** to start fresh."

**THEN:** Halt.

The existing content in `{outputFile}` provides context from previously completed steps.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Output document loaded and parsed correctly
- Explicit or legacy progress state resolved correctly
- Progress dashboard displayed accurately
- Routed to correct next step

### ❌ SYSTEM FAILURE:

- Not loading output document
- Incorrect progress display
- Routing to wrong step
- Re-executing completed steps

**Master Rule:** Resume MUST route to the exact next incomplete step. Never re-execute completed steps.
