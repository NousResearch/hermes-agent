---
name: 'step-06-platform-requirements'
description: 'Verify platform requirements and compile final validation report'

# File References
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-validate.md'
---

# Validation Step 06: Platform Requirements

## STEP GOAL:
Verify technical platform requirements are specified and consistent with project scope, then compile the final validation report.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst completing the final validation of Phase 1
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Tech stack, integrations, contact strategy, multilingual, final validation report
- FORBIDDEN: Do not skip compiling the final validation report across all 6 steps
- Approach: Check prerequisites, validate platform sections, compile final report, save

## EXECUTION PROTOCOLS:
- Primary goal: Platform requirements validated and final validation report compiled
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Platform Requirements document, all previous validation results
- Focus: Platform validation and final report compilation
- Limits: Validation only, not modification
- Dependencies: Steps 01-05 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 0. Prerequisites

Check if Platform Requirements document exists at `{output_folder}/A-Product-Brief/`. If not, note as "Platform Requirements not defined" and skip to final report.

### 1. Tech Stack

- [ ] CMS/framework specified
- [ ] Hosting approach noted
- [ ] Technical constraints documented

### 2. Integrations

- [ ] Third-party integrations listed
- [ ] Each integration has purpose documented
- [ ] No conflicting integration choices

### 3. Contact Strategy

- [ ] Contact form requirements documented
- [ ] Communication channels specified (email, chat, phone)

### 4. Multilingual (if applicable)

- [ ] Language switching approach defined
- [ ] URL structure for languages specified
- [ ] Translation workflow noted

### 5. Platform Report

```
## Platform Requirements Report

**Status:** [Defined / Not defined]
**Tech stack:** [Specified / Not specified]
**Integrations:** [N]
**Multilingual:** [Yes/No/N/A]
**Issues:** [N]

[List any unresolved technical decisions]
```

### 6. Final Validation Report

Compile results from all 6 validation steps:

```
## Phase 1 Validation Report

**Project:** {project_name}
**Date:** [date]
**Brief level:** [complete/simplified]

### Results Summary
| Check | Status | Issues |
|-------|--------|--------|
| Brief Completeness | pass/warn/fail | [summary] |
| Trigger Map Consistency | pass/warn/fail | [summary] |
| SEO Strategy | pass/warn/fail | [summary] |
| Content & Language | pass/warn/fail | [summary] |
| Visual Direction | pass/warn/fail | [summary] |
| Platform Requirements | pass/warn/fail | [summary] |

### Critical Issues (must fix)
[list or "None"]

### Warnings (should fix)
[list or "None"]

### Recommendations
[list or "All clear"]
```

Save report to `{output_folder}/A-Product-Brief/validation-report.md`

### N. Present MENU OPTIONS
Display: "**Select an Option:** [M] Return to workflow menu"

#### Menu Handling Logic:
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE
This is the FINAL step of the Phase 1 Validation workflow. Validation is complete.

---

## SYSTEM SUCCESS/FAILURE METRICS

### SUCCESS:
- Prerequisites checked
- Platform requirements validated
- Final validation report compiled across all 6 steps
- Report saved to output folder
- User informed of validation results

### FAILURE:
- Skipped prerequisite check
- Did not compile final validation report
- Left platform sections unverified
- Did not save report

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
