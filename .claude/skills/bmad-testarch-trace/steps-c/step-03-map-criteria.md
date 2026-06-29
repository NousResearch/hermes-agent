---
name: 'step-03-map-criteria'
description: 'Map coverage oracle items to tests and build traceability matrix'
nextStepFile: '{skill-root}/steps-c/step-04-analyze-gaps.md'
outputFile: '{test_artifacts}/traceability-matrix.md'
---

# Step 3: Map Coverage Oracle to Tests

## STEP GOAL

Create the traceability matrix linking the resolved oracle items to tests.

## MANDATORY EXECUTION RULES

- 📖 Read the entire step file before acting
- ✅ Speak in `{communication_language}`

---

## EXECUTION PROTOCOLS:

- 🎯 Follow the MANDATORY SEQUENCE exactly
- 💾 Record outputs before proceeding
- 📖 Load the next step only when instructed

## CONTEXT BOUNDARIES:

- Available context: config, loaded artifacts, and knowledge fragments
- Focus: this step's goal only
- Limits: do not execute future steps
- Dependencies: prior steps' outputs (if any)

## MANDATORY SEQUENCE

**CRITICAL:** Follow this sequence exactly. Do not skip, reorder, or improvise.

## 1. Build Matrix

For each resolved oracle item (formal requirement, endpoint/spec item, or synthetic journey):

- Map to matching tests
- Mark coverage status: FULL / PARTIAL / NONE / UNIT-ONLY / INTEGRATION-ONLY
- Record test level and priority
- Preserve each mapped test's stable identity fields (`id`, `title`, `file`, `line`, `level`, status flags) so Phase 1 can deduplicate unique tests before JSON export
- Record heuristic signals:
  - Endpoint coverage present/missing (for API-impacting items)
  - Auth/authz coverage present/missing (positive and negative paths)
  - Error-path coverage present/missing (validation, timeout, network/server failures)
  - UI journey E2E coverage present/missing (for source-derived journeys)
  - UI state coverage present/missing (loading, empty, validation, error, permission-denied)

---

## 2. Validate Coverage Logic

Ensure:

- P0/P1 items have coverage
- No duplicate coverage across levels without justification
- Items are not happy-path-only when the oracle implies error handling or alternate states
- API items are not marked FULL if endpoint-level checks are missing
- Auth/authz items include at least one denied/invalid-path test where applicable
- Synthetic UI journeys are not marked FULL when no E2E or component test asserts the critical path and key failure states

---

### 3. Save Progress

**Save this step's accumulated work to `{outputFile}`.**

- **If `{outputFile}` does not exist** (first save), create it using the workflow template (if available) with YAML frontmatter:

  ```yaml
  ---
  stepsCompleted: ['step-03-map-criteria']
  lastStep: 'step-03-map-criteria'
  lastSaved: '{date}'
  ---
  ```

  Then write this step's output below the frontmatter.

- **If `{outputFile}` already exists**, update:
  - Add `'step-03-map-criteria'` to `stepsCompleted` array (only if not already present)
  - Set `lastStep: 'step-03-map-criteria'`
  - Set `lastSaved: '{date}'`
  - Append this step's output to the appropriate section of the document.

Load next step: `{nextStepFile}`

## 🚨 SYSTEM SUCCESS/FAILURE METRICS:

### ✅ SUCCESS:

- Step completed in full with required outputs

### ❌ SYSTEM FAILURE:

- Skipped sequence steps or missing outputs
  **Master Rule:** Skipping steps is FORBIDDEN.
