# NFR Evidence Audit

**Workflow:** `bmad-testarch-nfr`
**Version:** 5.0 (Step-File Architecture)

---

## Overview

Audit non-functional requirement evidence (performance, security, reliability, maintainability) with deterministic PASS/CONCERNS/FAIL outcomes.

Use this workflow after implementation evidence exists: tests, scans, metrics, logs, monitoring data, CI results, or other release evidence. Use `test-design` earlier to define NFR thresholds, planned evidence, and NFR-derived risks.

---

## WORKFLOW ARCHITECTURE

This workflow uses **step-file architecture**:

- **Micro-file Design**: Each step is self-contained
- **JIT Loading**: Only the current step file is in memory
- **Sequential Enforcement**: Execute steps in order

---

## INITIALIZATION SEQUENCE

### 1. Configuration Loading

From `workflow.yaml`, resolve:

- `config_source`, `test_artifacts`, `user_name`, `communication_language`, `document_output_language`, `date`
- `custom_nfr_categories`

### 2. First Step

Load, read completely, and execute:
`{skill-root}/steps-c/step-01-load-context.md`

### 3. Resume Support

If the user selects **Resume** mode, load, read completely, and execute:
`{skill-root}/steps-c/step-01b-resume.md`

This checks the output document for progress tracking frontmatter and routes to the next incomplete step.
