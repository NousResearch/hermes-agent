---
name: 'step-03-risk-and-testability'
description: 'Perform testability review (system-level) and risk assessment'
nextStepFile: '{skill-root}/steps-c/step-04-coverage-plan.md'
outputFile: '{test_artifacts}/test-design-progress.md'
---

# Step 3: Testability & Risk Assessment

## STEP GOAL

Produce a defensible testability review (system-level) and a risk assessment matrix (all modes).

## MANDATORY EXECUTION RULES

- 📖 Read the entire step file before acting
- ✅ Speak in `{communication_language}`
- 🎯 Base conclusions on evidence from loaded artifacts

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

## 1. System-Level Mode: Testability Review

If **system-level**, evaluate architecture for:

- **Controllability** (state seeding, mockability, fault injection)
- **Observability** (logs, metrics, traces, deterministic assertions)
- **Reliability** (isolation, reproducibility, parallel safety)

**Structure output as:**

1. **🚨 Testability Concerns** (actionable issues first)
2. **✅ Testability Assessment Summary** (what is already strong)

Also identify **ASRs** (Architecturally Significant Requirements):

- Mark each as **ACTIONABLE** or **FYI**

---

## 2. All Modes: Risk Assessment

Using `risk-governance.md` and `probability-impact.md` (if loaded):

- Identify real risks (not just features)
- Classify by category: TECH / SEC / PERF / DATA / BUS / OPS
- Score Probability (1–3) and Impact (1–3)
- Calculate Risk Score (P × I)
- Flag high risks (score ≥ 6)
- Define mitigation, owner, and timeline

---

## 3. NFR Planning Assessment

Using `nfr-criteria.md` when loaded:

- Identify NFR categories in scope: security, performance, reliability, scalability, maintainability, compliance, and any project-specific categories
- Extract measurable thresholds from PRD, architecture, ADRs, epics, or stories
- Mark missing thresholds as **UNKNOWN** and convert them into clarification items or risks; do not guess values
- Define planned evidence sources for later validation (tests, scans, metrics, logs, monitoring, CI reports)
- Convert NFR gaps into the existing risk register using SEC / PERF / OPS / TECH / DATA categories

**Boundary:** This workflow plans NFR validation. It does not assess final PASS/CONCERNS/FAIL from implementation evidence. Use `nfr-assess` after implementation evidence exists.

---

## 4. Summarize Risk Findings

Summarize the highest risks and their mitigation priorities.

---

### 5. Save Progress

**Save this step's accumulated work to `{outputFile}`.**

- **If `{outputFile}` does not exist** (first save), create it with YAML frontmatter:

  ```yaml
  ---
  workflowStatus: 'in-progress'
  totalSteps: 5
  stepsCompleted: ['step-03-risk-and-testability']
  lastStep: 'step-03-risk-and-testability'
  nextStep: '{nextStepFile}'
  lastSaved: '{date}'
  ---
  ```

  Then write this step's output below the frontmatter.

- **If `{outputFile}` already exists**, update:
  - Set `workflowStatus: 'in-progress'`
  - Set `totalSteps: 5`
  - Add `'step-03-risk-and-testability'` to `stepsCompleted` array (only if not already present)
  - Set `lastStep: 'step-03-risk-and-testability'`
  - Set `nextStep: '{nextStepFile}'`
  - Set `lastSaved: '{date}'`
  - Append this step's output to the appropriate section of the document.

Load next step: `{nextStepFile}`

## 🚨 SYSTEM SUCCESS/FAILURE METRICS:

### ✅ SUCCESS:

- Step completed in full with required outputs

### ❌ SYSTEM FAILURE:

- Skipped sequence steps or missing outputs
  **Master Rule:** Skipping steps is FORBIDDEN.
